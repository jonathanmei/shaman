"""Proof of concept: polish one tile of stored sign bits of a late, sensitive layer with warm-started RQAOA.

Stages (``python scripts/qaoa_sign_polish.py <stage> <config.json>``)::

    pick-layer   rank the layers of the last third of blocks by depth-adjusted measured sensitivity (cluster)
    dump         calibrate, run the layer's ADMM, select the tile, write instance.json + layer_state.pt (cluster)
    solve        warm-started RQAOA at every depth in ``depths``, write solution_p{p}.json
    verify       re-apply the solved bits, report the curvature-weighted error before / after (cluster)
    all          dump, then solve and verify per depth, in one process
    crosscheck   statevector vs Pauli propagation vs quimb MPS on the dumped instance (and a 16-bit sub-instance)

The ``dump`` stage factorises the chosen layer against the calibration curvature of the FP model (exactly what the
rank probe does), not against the error-fed inputs of the block loop nor the fresh input factor; the instance is
therefore a faithful *stand-alone* layer problem, not the one the full pipeline solved at that position.

Backends: ``statevector`` (exact, ``n <= 22``), ``pauli`` (Pauli propagation: exact at ``p = 1``, truncated at
order ``prop_weight`` beyond; any ``n``), ``ionq`` (counts from an IonQ backend through qiskit-ionq).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel

from nanoquant.core.compress_block import mahalanobis_weight_error
from nanoquant.core.importance import collect_stats, get_shrunk_stats, register_stats
from nanoquant.core.pipeline import collect_stats_kwargs
from nanoquant.core.rank_probe import PROBE_KIND, deployed_matrix, measure_sensitivity
from nanoquant.main import load_quant_config
from nanoquant.quantum.ising import IsingInstance, local_search, simulated_annealing
from nanoquant.quantum.layer_admm import admm_for_layer, layer_curvature
from nanoquant.quantum.pick_layer import rank_late_layers
from nanoquant.quantum.rqaoa import (
    MAX_STATEVECTOR_N,
    Correlator,
    PauliPropCorrelator,
    RQAOAResult,
    StatevectorCorrelator,
    normalized_angles,
    optimize_angles,
    rqaoa,
    warm_start_angles,
)
from nanoquant.quantum.sign_tile import Tile, apply_tile, select_tile, tile_ising
from nanoquant.utils.cache import ArtifactCache, probe_key, stats_key
from nanoquant.utils.data_utils import get_calib_loader, prepare_dataset
from nanoquant.utils.load_utils import load_model, load_tokenizer
from nanoquant.utils.utils import (
    calculate_ranks,
    find_layers,
    get_decoder_layers,
    get_layers_to_factorize,
    set_seed,
)

PICK = "pick_layer.json"
INSTANCE = "instance.json"
LAYER_STATE = "layer_state.pt"
SOLUTION = "solution.json"
VERIFY = "verify.json"
CROSSCHECK = "crosscheck.json"


def solution_name(p: int) -> str:
    """``solution.json`` at depth 1 (the original proof-of-concept name), ``solution_p{p}.json`` otherwise."""
    return SOLUTION if p == 1 else f"solution_p{p}.json"


def verify_name(p: int) -> str:
    """``verify.json`` at depth 1, ``verify_p{p}.json`` otherwise."""
    return VERIFY if p == 1 else f"verify_p{p}.json"


class PolishConfig(BaseModel):
    """Settings of one proof-of-concept run.

    Parameters
    ----------
    pipeline_config : str
        Path of the pipeline JSON config whose cache (calibration statistics, rank probe) and ADMM settings are used.
    out_dir : str
        Where the run's files go.
    block, module : int, str
        The layer (``"<block>.<module>"``) to polish. Leave both ``null`` to let ``dump`` take the top of the
        ``pick-layer`` ranking (it writes ``pick_layer.json``); pin them afterwards to document the run.
    n : int
        Tile size (number of qubits).
    depths : list of int
        QAOA depths ``p`` to solve (one solution / verify file per depth).
    shots : int
        Shots per RQAOA step (counts backends only).
    seed : int
        Seeds the angle multistart and the statevector sampler.
    n_stop : int
        Spins brute-forced at the end of the recursion.
    eps : float
        Warm-start regularisation (probability floor of flipping a fully confident bit).
    backend : {"statevector", "pauli", "ionq"}
        Correlator.
    prop_weight, max_active, pair_top : int
        Pauli propagation: perturbative order, eligible non-root sites per observable, and pairs evaluated per spin
        (``null`` = all).
    n_starts, reoptimize_every : int
        Random angle starts at the first recursion step; re-optimisation period (carried angles in between).
    angles_from : str, optional
        A solution JSON whose first-step angles (normalised units) are used at every step without optimisation.
    ionq_target, ionq_noise_model : str
        IonQ backend (``"simulator"``, ``"qpu.aria-1"``, ...) and simulator noise model (``"aria-1"``, ...).
    device : str
        Torch device for the cluster stages.
    instance_path : str, optional
        Solve this dumped ``instance.json`` instead of ``out_dir/instance.json`` (other backend, same tile).
    mps_bonds : list of int
        Bond dimensions for the ``crosscheck`` stage.
    sub_tile : int
        Size of the statevector-checkable sub-instance in ``crosscheck``.
    angles_sub_tile : int, optional
        Optimise the angles once on the sub-instance of this many spins (same backend) and hold them fixed for the
        whole recursion (the transfer strategy for deep / large runs).
    opt_maxiter : int
        Nelder-Mead iteration cap per start.
    depth_overrides : dict
        Per-depth overrides of any of the above, keyed by the depth as a string (``{"3": {"pair_top": 4}}``).
    """

    pipeline_config: str
    out_dir: str = "runs/qaoa_sign_polish"
    block: int | None = None
    module: str | None = None
    n: int = 9
    depths: list[int] = [1]
    shots: int = 2000
    seed: int = 0
    n_stop: int = 4
    eps: float = 0.1
    backend: Literal["statevector", "pauli", "ionq"] = "statevector"
    prop_weight: int = 1
    max_active: int | None = None
    pair_top: int | None = None
    n_starts: int = 8
    reoptimize_every: int = 1
    angles_from: str | None = None
    ionq_target: str = "simulator"
    ionq_noise_model: str | None = None
    device: str = "cuda"
    instance_path: str | None = None
    mps_bonds: list[int] = [16, 64]
    sub_tile: int = 16
    angles_sub_tile: int | None = None
    opt_maxiter: int = 400
    depth_overrides: dict[str, dict] = {}

    def for_depth(self, p: int) -> PolishConfig:
        """This config with the overrides registered for depth ``p`` applied."""
        return self.model_copy(update=self.depth_overrides.get(str(p), {}))

    @property
    def instance_file(self) -> Path:
        """``instance_path`` if set (re-solving a dumped instance with another backend), else ``out_dir/instance.json``."""
        return Path(self.instance_path) if self.instance_path else Path(self.out_dir) / INSTANCE

    @classmethod
    def from_json(cls, path: str | Path) -> PolishConfig:
        """Read a config file."""
        return cls.model_validate(json.loads(Path(path).read_text()))

    @property
    def layer_key(self) -> str:
        """``"<block>.<module>"``."""
        if self.block is None or self.module is None:
            raise ValueError("block and module must be set (run pick-layer first and pin its choice)")
        return f"{self.block}.{self.module}"


# ----------------------------------------------------------------------------------------------------
# Cluster side: calibrated model, ranks, ADMM of one layer
# ----------------------------------------------------------------------------------------------------
def calibrated_model(qc: dict, dev: str):
    """Load the FP model, register its (cached) calibration curvature and return the rank-probe artifact and ranks.

    Returns
    -------
    tuple
        ``(model, layers, sensitivity, ranks)``.
    """
    cache = ArtifactCache(qc.get("cache_dir", ""))
    model = load_model(qc["model_id"], qc["seqlen"], device_map=qc.get("device_map", "cpu"))
    layers = get_layers_to_factorize(model.config.model_type)

    def stats():
        data = prepare_dataset(qc["model_id"], qc)
        loader = get_calib_loader(data, load_tokenizer(qc["model_id"]), qc["num_calib_samples"], qc["seed"],
                                  qc["seqlen"])
        return collect_stats(model, loader, dev, **collect_stats_kwargs(qc))

    raw = cache.load_or_compute("stats", stats_key(qc), stats)
    model = register_stats(model, get_shrunk_stats(raw, shrinkage=qc["calib_shrinkage"]))
    sensitivity = None
    if (qc.get("rank_sensitivity", "none") or "none") != "none":
        sensitivity = cache.load_or_compute(PROBE_KIND, probe_key(qc),
                                            lambda: measure_sensitivity(model, layers, qc, dev))
    ranks = calculate_ranks(model, layers, qc, sensitivity=sensitivity)
    return model, layers, sensitivity, ranks


def _ranking(cfg: PolishConfig, qc: dict, model, sensitivity: dict | None, ranks: dict) -> list[dict]:
    """Print and return the late-block sensitivity ranking; also written to ``out_dir/pick_layer.json``."""
    if sensitivity is None:
        raise ValueError("picking a layer needs a measured rank sensitivity (rank_sensitivity != 'none')")
    n_blocks = len(get_decoder_layers(model))
    scores = rank_late_layers(sensitivity["curves"], ranks, n_blocks, float(qc.get("rank_depth_ramp", 0.0) or 0.0))
    print(f"{'layer':28s} {'rank':>5s} {'level':>12s} {'score':>12s}")
    for s in scores:
        print(f"{s.key:28s} {s.rank:5d} {s.level:12.4e} {s.score:12.4e}")
    if scores:
        print(f"\npick: block {scores[0].block}, module {scores[0].name} (pin these in the config as block / module)")
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [s.model_dump() for s in scores]
    (out / PICK).write_text(json.dumps(rows, indent=1))
    return rows


def pick_layer(cfg: PolishConfig) -> list[dict]:
    """Rank the late-block layers by depth-adjusted measured sensitivity."""
    qc = load_quant_config(cfg.pipeline_config)
    model, _, sensitivity, ranks = calibrated_model(qc, cfg.device)
    return _ranking(cfg, qc, model, sensitivity, ranks)


def dump_from_layer_state(W: torch.Tensor, L: torch.Tensor | None, R: torch.Tensor | None, result: dict,
                          cfg: PolishConfig, layer_key: str, rank: int) -> IsingInstance:
    """Select the tile, build the Ising instance and write ``instance.json`` + ``layer_state.pt``."""
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tile = select_tile(result, cfg.n, layer=layer_key)
    inst = tile_ising(W, L, R, result, tile)
    inst.meta.update({"rank": rank, "admm_energy": inst.energy(tile.signs),
                      "layer_error_fp32": mahalanobis_weight_error(W, deployed_matrix(result), L, R)})
    inst.to_json(out / INSTANCE)
    torch.save({"W": W.detach().cpu(), "L": None if L is None else L.detach().cpu(),
                "R": None if R is None else R.detach().cpu(),
                "result": {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in result.items()},
                "tile": tile.model_dump(), "layer": layer_key, "rank": rank}, out / LAYER_STATE)
    J, h = inst.J_array(), inst.h_array()
    print(f"[dump] {layer_key}: rank {rank}, tile of {tile.n} bits, ADMM energy {inst.meta['admm_energy']:.6e}, "
          f"|J| max {np.abs(J).max():.3e}, |h| max {np.abs(h).max():.3e}, "
          f"median |J|/|h| {np.median(np.abs(J[np.triu_indices(tile.n, 1)])) / np.median(np.abs(h)):.3f} "
          f"-> {out / INSTANCE}")
    return inst


def dump(cfg: PolishConfig) -> IsingInstance:
    """Calibrate, factorise the chosen layer at its allocated rank, and write the tile instance."""
    qc = load_quant_config(cfg.pipeline_config)
    model, _, sensitivity, ranks = calibrated_model(qc, cfg.device)
    if cfg.block is None or cfg.module is None:
        # unpinned config: take the top of the ranking (recorded in pick_layer.json and in the instance meta)
        top = _ranking(cfg, qc, model, sensitivity, ranks)[0]
        cfg = cfg.model_copy(update={"block": top["block"], "module": top["name"]})
    key = cfg.layer_key
    if key not in ranks:
        raise KeyError(f"{key} is not a factorised layer; known keys look like {next(iter(ranks))}")
    lx = find_layers(get_decoder_layers(model)[cfg.block])[cfg.module]
    W, i_norm, o_norm, i_cov, o_cov, L, R = layer_curvature(lx, cfg.device)
    set_seed(qc["seed"])
    t0 = time.time()
    result = admm_for_layer(W, i_norm, o_norm, i_cov, o_cov, ranks[key], qc,
                            eigh_dtype=getattr(torch, qc.get("kron_eigh_dtype", "float64")))
    print(f"[dump] ADMM of {key} at rank {ranks[key]} in {time.time() - t0:.0f}s")
    return dump_from_layer_state(W, L, R, result, cfg, key, ranks[key])


# ----------------------------------------------------------------------------------------------------
# Solve
# ----------------------------------------------------------------------------------------------------
def make_correlator(cfg: PolishConfig) -> Correlator:
    """The configured correlator."""
    if cfg.backend == "statevector":
        return StatevectorCorrelator()
    if cfg.backend == "pauli":
        return PauliPropCorrelator(k=cfg.prop_weight, max_active=cfg.max_active, pair_top=cfg.pair_top)
    from nanoquant.quantum.ionq_backend import IonQSampler

    return StatevectorCorrelator(IonQSampler(target=cfg.ionq_target, noise_model=cfg.ionq_noise_model))


def classical_references(inst: IsingInstance, admm_signs: list[int], seed: int) -> dict:
    """Brute force below the statevector limit; otherwise 1-flip descent from the ADMM bits and annealing."""
    if inst.n <= MAX_STATEVECTOR_N:
        s, e = inst.brute_force()
        return {"reference": "brute_force", "reference_energy": e, "reference_spins": s}
    t0 = time.time()
    s_ls, e_ls = local_search(inst, admm_signs)
    s_sa, e_sa = simulated_annealing(inst, seed=seed, restarts=8, sweeps=200, s0=admm_signs)
    best_s, best_e = (s_sa, e_sa) if e_sa <= e_ls else (s_ls, e_ls)
    return {"reference": "simulated_annealing", "reference_energy": best_e, "reference_spins": best_s,
            "local_search_energy": e_ls, "annealing_energy": e_sa, "reference_seconds": time.time() - t0}


def solve_depth(cfg: PolishConfig, inst: IsingInstance, p: int, refs: dict) -> dict:
    """Run warm-started RQAOA at depth ``p`` and write ``solution_p{p}.json``."""
    cfg = cfg.for_depth(p)
    out = Path(cfg.out_dir)
    signs, conf = inst.meta["admm_signs"], inst.meta["confidence"]
    theta = warm_start_angles(signs, conf, eps=cfg.eps)
    correlator = make_correlator(cfg)
    fixed, angle_source = None, "optimised in the recursion"
    if cfg.angles_from:
        src = json.loads(Path(cfg.angles_from).read_text())
        step0 = src["steps"][0]
        if len(step0["gammas"]) != p:
            raise ValueError(f"{cfg.angles_from} has depth {len(step0['gammas'])}, config asks for {p}")
        fixed = src.get("normalized_angles_step0") or normalized_angles(
            IsingInstance.from_json(Path(cfg.angles_from).parent / INSTANCE), step0["gammas"], step0["betas"]).tolist()
        angle_source = f"transferred from {cfg.angles_from}"
    elif cfg.angles_sub_tile and inst.n > cfg.angles_sub_tile:
        sub = _sub_instance(inst, cfg.angles_sub_tile)
        theta_sub = warm_start_angles(sub.meta["admm_signs"], sub.meta["confidence"], eps=cfg.eps)
        t0 = time.time()
        g_sub, b_sub, e_sub = optimize_angles(sub, theta_sub, p=p, seed=cfg.seed, n_starts=cfg.n_starts,
                                              correlator=correlator, maxiter=cfg.opt_maxiter)
        fixed = normalized_angles(sub, g_sub, b_sub).tolist()
        angle_source = f"optimised on the first {sub.n} spins in {time.time() - t0:.0f}s (sub energy {e_sub:.6e})"
        print(f"[solve] p={p}: angles {angle_source}")
    t0 = time.time()
    res: RQAOAResult = rqaoa(inst, theta, correlator, shots=cfg.shots, seed=cfg.seed, n_stop=cfg.n_stop, p=p,
                             n_starts=cfg.n_starts, reoptimize_every=cfg.reoptimize_every, fixed_angles=fixed,
                             log=print, maxiter=cfg.opt_maxiter)
    seconds = time.time() - t0
    admm_e = inst.energy(signs)
    ref_e = refs["reference_energy"]
    gain_available = admm_e - ref_e
    captured = (admm_e - res.energy) / gain_available if gain_available > 0 else 1.0
    from nanoquant.quantum.ionq_backend import two_qubit_gate_count

    _, e_ls_from_rqaoa = local_search(inst, res.spins)
    summary = {
        "layer": inst.meta.get("layer"), "n": inst.n, "p": p, "backend": cfg.backend,
        "prop_weight": cfg.prop_weight if cfg.backend == "pauli" else None,
        "max_active": cfg.max_active, "pair_top": cfg.pair_top,
        "ionq_target": cfg.ionq_target if cfg.backend == "ionq" else None,
        "shots": cfg.shots, "seed": cfg.seed, "eps": cfg.eps, "n_stop": cfg.n_stop,
        "n_starts": cfg.n_starts, "reoptimize_every": cfg.reoptimize_every, "angles_from": cfg.angles_from,
        "angle_source": angle_source, "opt_maxiter": cfg.opt_maxiter,
        "admm_energy": admm_e, "rqaoa_energy": res.energy, "delta_vs_admm": admm_e - res.energy,
        **refs, "gain_available": gain_available, "fraction_captured": captured,
        "hit_reference": bool(res.energy <= ref_e + 1e-9 * max(1.0, abs(ref_e))),
        "admm_bits_optimal": bool(gain_available <= 1e-9 * max(1.0, abs(ref_e))),
        "rqaoa_is_1flip_local_min": bool(e_ls_from_rqaoa >= res.energy - 1e-12 * max(1.0, abs(res.energy))),
        "bits_flipped_vs_admm": int(sum(a != b for a, b in zip(res.spins, signs))),
        "two_qubit_gates_per_layer": two_qubit_gate_count(inst, p=1),
        "hardware_jobs": len(res.steps), "seconds": seconds,
        "seconds_per_step": [s.seconds for s in res.steps],
        "normalized_angles_step0": normalized_angles(inst, res.steps[0].gammas, res.steps[0].betas).tolist()
        if res.steps else None,
        "job_ids": list(getattr(getattr(correlator, "sampler", None), "job_ids", [])),
        "spins": res.spins, "admm_spins": list(signs),
        "steps": [s.model_dump() for s in res.steps],
    }
    (out / solution_name(p)).write_text(json.dumps(summary, indent=1))
    print(f"[solve] p={p}, {inst.n} spins on {cfg.backend}: ADMM {admm_e:.6e} -> RQAOA {res.energy:.6e} "
          f"({refs['reference']} {ref_e:.6e}, captured {captured:.1%} of the available gain, hit: "
          f"{summary['hit_reference']}, {summary['bits_flipped_vs_admm']} bits flipped) in {seconds:.1f}s, "
          f"{len(res.steps)} steps -> {out / solution_name(p)}")
    return summary


def solve(cfg: PolishConfig) -> list[dict]:
    """Warm-started RQAOA on the dumped instance at every configured depth."""
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    inst = IsingInstance.from_json(cfg.instance_file)
    refs = classical_references(inst, inst.meta["admm_signs"], cfg.seed)
    print(f"[solve] reference ({refs['reference']}): {refs['reference_energy']:.6e}")
    return [solve_depth(cfg, inst, p, refs) for p in cfg.depths]


# ----------------------------------------------------------------------------------------------------
# Verify
# ----------------------------------------------------------------------------------------------------
def verify_depth(cfg: PolishConfig, p: int) -> dict:
    """Re-apply the depth-``p`` solution to the saved layer and report the error before / after."""
    out = Path(cfg.out_dir)
    state = torch.load(out / LAYER_STATE, weights_only=False)
    sol = json.loads((out / solution_name(p)).read_text())
    tile = Tile.model_validate(state["tile"])
    W, L, R, result = state["W"], state["L"], state["R"], state["result"]
    before = mahalanobis_weight_error(W, deployed_matrix(result), L, R)
    after = mahalanobis_weight_error(W, deployed_matrix(apply_tile(result, tile, sol["spins"])), L, R)
    inst = IsingInstance.from_json(cfg.instance_file)
    report = {"layer": state["layer"], "rank": state["rank"], "n": tile.n, "p": p,
              "error_before": before, "error_after": after, "delta_measured": before - after,
              "delta_predicted": inst.energy(tile.signs) - inst.energy(sol["spins"]),
              "relative_change": (after - before) / before if before else 0.0}
    (out / verify_name(p)).write_text(json.dumps(report, indent=1))
    print(f"[verify] p={p} {state['layer']}: error {before:.6e} -> {after:.6e} "
          f"(measured delta {report['delta_measured']:.3e}, predicted {report['delta_predicted']:.3e}, "
          f"{report['relative_change']:+.2e} relative) -> {out / verify_name(p)}")
    return report


def verify(cfg: PolishConfig) -> list[dict]:
    """Verify every configured depth."""
    return [verify_depth(cfg, p) for p in cfg.depths]


def run_all(cfg: PolishConfig) -> list[dict]:
    """``dump`` once, then ``solve`` and ``verify`` per depth."""
    dump(cfg)
    inst = IsingInstance.from_json(cfg.instance_file)
    refs = classical_references(inst, inst.meta["admm_signs"], cfg.seed)
    print(f"[solve] reference ({refs['reference']}): {refs['reference_energy']:.6e}")
    reports = []
    for p in cfg.depths:
        solve_depth(cfg, inst, p, refs)
        reports.append(verify_depth(cfg, p))
    return reports


# ----------------------------------------------------------------------------------------------------
# Cross-check of the simulators
# ----------------------------------------------------------------------------------------------------
def _sub_instance(inst: IsingInstance, m: int) -> IsingInstance:
    idx = list(range(min(m, inst.n)))
    J = inst.J_array()[np.ix_(idx, idx)]
    return IsingInstance(h=[inst.h[i] for i in idx], J=J.tolist(), const=inst.const,
                         meta={"admm_signs": [inst.meta["admm_signs"][i] for i in idx],
                               "confidence": [inst.meta["confidence"][i] for i in idx]})


def _errors(ref: tuple, got: tuple) -> dict:
    e0, z0, zz0 = ref
    e1, z1, zz1 = got
    return {"energy_abs_err": abs(e1 - e0), "z_max_abs_err": float(np.abs(z1 - z0).max()),
            "zz_max_abs_err": float(np.abs(zz1 - zz0).max())}


def crosscheck(cfg: PolishConfig) -> dict:
    """Compare statevector (sub-instance), Pauli propagation (k = 0, 1, 2) and quimb MPS (``mps_bonds``).

    Angles per depth come from a short statevector optimisation on the sub-instance and are reused (in normalised
    units) on the full dumped instance, where Pauli propagation and MPS are compared against each other.
    """
    from nanoquant.quantum.ising import cost_scale, spin_table
    from nanoquant.quantum.mps_check import mps_expectations
    from nanoquant.quantum.pauli_prop import expectations
    from nanoquant.quantum.rqaoa import expectation, probabilities, qaoa_state

    out = Path(cfg.out_dir)
    if not cfg.instance_file.exists():
        dump(cfg)
    full = IsingInstance.from_json(cfg.instance_file)
    sub = _sub_instance(full, cfg.sub_tile)
    report: dict = {"n_full": full.n, "n_sub": sub.n, "depths": {}}
    for p in cfg.depths:
        theta_sub = warm_start_angles(sub.meta["admm_signs"], sub.meta["confidence"], eps=cfg.eps)
        gammas, betas, _ = optimize_angles(sub, theta_sub, p=p, seed=cfg.seed, n_starts=4)
        probs = probabilities(qaoa_state(sub, theta_sub, gammas, betas))
        S = spin_table(sub.n).astype(float)
        zz_ref = (S * probs[:, None]).T @ S
        np.fill_diagonal(zz_ref, 0.0)
        ref = (expectation(sub, theta_sub, gammas, betas), probs @ S, zz_ref)
        entry: dict = {"gammas": gammas, "betas": betas, "sub": {}, "full": {}}
        for k in (0, 1, 2):
            t0 = time.time()
            got = expectations(sub, theta_sub, gammas, betas, k=k)
            entry["sub"][f"pauli_k{k}"] = {**_errors(ref, got), "seconds": time.time() - t0}
        for chi in cfg.mps_bonds:
            t0 = time.time()
            e, z, zz, reached = mps_expectations(sub, theta_sub, gammas, betas, max_bond=chi)
            entry["sub"][f"mps_chi{chi}"] = {**_errors(ref, (e, z, zz)), "seconds": time.time() - t0,
                                             "bond_reached": reached}
        # full instance: same normalised angles, methods against each other (no exact reference)
        norm = normalized_angles(sub, gammas, betas)
        g_full, b_full = (norm[:p] / cost_scale(full)).tolist(), norm[p:].tolist()
        theta_full = warm_start_angles(full.meta["admm_signs"], full.meta["confidence"], eps=cfg.eps)
        results: dict[str, tuple] = {}
        # second order at depth >= 3 materialises ~10^5 terms per pair on 32 bits (many minutes); first order there
        ks = (0, 1, 2) if p <= 2 else (0, 1)
        for k in ks:
            t0 = time.time()
            results[f"pauli_k{k}"] = expectations(full, theta_full, g_full, b_full, k=k)
            entry["full"][f"pauli_k{k}"] = {"energy": results[f"pauli_k{k}"][0], "seconds": time.time() - t0}
        for chi in cfg.mps_bonds:
            t0 = time.time()
            e, z, zz, reached = mps_expectations(full, theta_full, g_full, b_full, max_bond=chi)
            results[f"mps_chi{chi}"] = (e, z, zz)
            entry["full"][f"mps_chi{chi}"] = {"energy": e, "seconds": time.time() - t0, "bond_reached": reached}
        ref_name = f"pauli_k{max(ks)}"
        best = results[ref_name]
        entry["full_reference"] = ref_name
        for name, got in results.items():
            entry["full"][name].update({f"vs_{ref_name}_{k}": v for k, v in _errors(best, got).items()})
        report["depths"][str(p)] = entry
        print(f"[crosscheck] p={p}: sub-instance errors vs statevector "
              + ", ".join(f"{m}: zz {v['zz_max_abs_err']:.2e}" for m, v in entry["sub"].items())
              + " | full: " + ", ".join(f"{m}: E {v['energy']:.6e} ({v['seconds']:.1f}s)"
                                       for m, v in entry["full"].items()))
    (out / CROSSCHECK).write_text(json.dumps(report, indent=1))
    return report


STAGES = {"pick-layer": pick_layer, "dump": dump, "solve": solve, "verify": verify, "all": run_all,
          "crosscheck": crosscheck}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in STAGES:
        print(f"usage: {Path(__file__).name} {{{'|'.join(STAGES)}}} <config.json>")
        raise SystemExit(2)
    STAGES[argv[0]](PolishConfig.from_json(argv[1]))


if __name__ == "__main__":
    main()
