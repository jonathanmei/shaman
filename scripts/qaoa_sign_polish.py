"""Proof of concept: polish one tile of stored sign bits of a late, sensitive layer with warm-started RQAOA.

Stages (``python scripts/qaoa_sign_polish.py <stage> <config.json>``)::

    pick-layer   rank the layers of the last third of blocks by depth-adjusted measured sensitivity (cluster)
    dump         calibrate, run the layer's ADMM, select the tile, write instance.json + layer_state.pt (cluster)
    solve        warm-started RQAOA on the chosen backend, write solution.json (local)
    verify       re-apply the solved bits, report the curvature-weighted error before / after (cluster)

The ``dump`` stage factorises the chosen layer against the calibration curvature of the FP model (exactly what the
rank probe does), not against the error-fed inputs of the block loop nor the fresh input factor; the instance is
therefore a faithful *stand-alone* layer problem, not the one the full pipeline solved at that position.
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
from nanoquant.quantum.ising import IsingInstance
from nanoquant.quantum.layer_admm import admm_for_layer, layer_curvature
from nanoquant.quantum.pick_layer import rank_late_layers
from nanoquant.quantum.rqaoa import (
    RQAOAResult,
    Sampler,
    StatevectorSampler,
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


class PolishConfig(BaseModel):
    """Settings of one proof-of-concept run.

    Parameters
    ----------
    pipeline_config : str
        Path of the pipeline JSON config whose cache (calibration statistics, rank probe) and ADMM settings are used.
    out_dir : str
        Where ``instance.json``, ``layer_state.pt``, ``solution.json`` and ``verify.json`` go.
    block, module : int, str
        The layer (``"<block>.<module>"``) to polish. Leave both ``null`` to let ``dump`` take the top of the
        ``pick-layer`` ranking (it writes ``pick_layer.json``); pin them afterwards to document the run.
    n : int
        Tile size (number of qubits).
    shots : int
        Shots per RQAOA step.
    seed : int
        Seeds the angle multistart and the statevector sampler.
    n_stop : int
        Spins brute-forced at the end of the recursion.
    eps : float
        Warm-start regularisation (probability floor of flipping a fully confident bit).
    backend : {"statevector", "ionq"}
        Sampler.
    ionq_target, ionq_noise_model : str
        IonQ backend (``"simulator"``, ``"qpu.aria-1"``, ...) and simulator noise model (``"aria-1"``, ...).
    device : str
        Torch device for the cluster stages.
    instance_path : str, optional
        Solve this dumped ``instance.json`` instead of ``out_dir/instance.json`` (other backend, same tile).
    """

    pipeline_config: str
    out_dir: str = "runs/qaoa_sign_polish"
    block: int | None = None
    module: str | None = None
    n: int = 9
    shots: int = 2000
    seed: int = 0
    n_stop: int = 4
    eps: float = 0.1
    backend: Literal["statevector", "ionq"] = "statevector"
    ionq_target: str = "simulator"
    ionq_noise_model: str | None = None
    device: str = "cuda"
    instance_path: str | None = None

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
    print(f"[dump] {layer_key}: rank {rank}, tile of {tile.n} bits, ADMM energy {inst.meta['admm_energy']:.6e}, "
          f"|J| max {np.abs(inst.J_array()).max():.3e}, |h| max {np.abs(inst.h_array()).max():.3e} -> {out / INSTANCE}")
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
# Local side: solve
# ----------------------------------------------------------------------------------------------------
def make_sampler(cfg: PolishConfig) -> Sampler:
    """The configured sampler."""
    if cfg.backend == "statevector":
        return StatevectorSampler()
    from nanoquant.quantum.ionq_backend import IonQSampler

    return IonQSampler(target=cfg.ionq_target, noise_model=cfg.ionq_noise_model)


def solve(cfg: PolishConfig) -> dict:
    """Run warm-started RQAOA on ``instance.json`` and write ``solution.json``."""
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inst = IsingInstance.from_json(cfg.instance_file)
    signs, conf = inst.meta["admm_signs"], inst.meta["confidence"]
    theta = warm_start_angles(signs, conf, eps=cfg.eps)
    sampler = make_sampler(cfg)
    t0 = time.time()
    res: RQAOAResult = rqaoa(inst, theta, sampler, shots=cfg.shots, seed=cfg.seed, n_stop=cfg.n_stop)
    seconds = time.time() - t0
    best_s, best_e = inst.brute_force()
    admm_e = inst.energy(signs)
    gain_available = admm_e - best_e
    captured = (admm_e - res.energy) / gain_available if gain_available > 0 else 1.0
    from nanoquant.quantum.ionq_backend import two_qubit_gate_count

    summary = {
        "layer": inst.meta.get("layer"), "n": inst.n, "backend": cfg.backend,
        "ionq_target": cfg.ionq_target if cfg.backend == "ionq" else None,
        "shots": cfg.shots, "seed": cfg.seed, "eps": cfg.eps, "n_stop": cfg.n_stop,
        "admm_energy": admm_e, "brute_force_energy": best_e, "rqaoa_energy": res.energy,
        "delta_vs_admm": admm_e - res.energy, "gain_available": gain_available, "fraction_captured": captured,
        "hit_optimum": bool(abs(res.energy - best_e) <= 1e-9 * max(1.0, abs(best_e))),
        "admm_bits_optimal": bool(gain_available <= 1e-9 * max(1.0, abs(best_e))),
        "two_qubit_gates_first_step": two_qubit_gate_count(inst, p=1),
        "hardware_jobs": len(res.steps), "seconds": seconds,
        "job_ids": list(getattr(sampler, "job_ids", [])),
        "spins": res.spins, "brute_force_spins": best_s, "admm_spins": list(signs),
        "steps": [s.model_dump() for s in res.steps],
    }
    (out / SOLUTION).write_text(json.dumps(summary, indent=1))
    print(f"[solve] {inst.n} spins on {cfg.backend}: ADMM {admm_e:.6e} -> RQAOA {res.energy:.6e} "
          f"(optimum {best_e:.6e}, captured {captured:.1%} of the available gain, hit optimum: "
          f"{summary['hit_optimum']}) in {seconds:.1f}s, {len(res.steps)} steps -> {out / SOLUTION}")
    return summary


# ----------------------------------------------------------------------------------------------------
# Cluster side: verify
# ----------------------------------------------------------------------------------------------------
def verify(cfg: PolishConfig) -> dict:
    """Re-apply the solved bits to the saved layer and report the curvature-weighted error before / after."""
    out = Path(cfg.out_dir)
    state = torch.load(out / LAYER_STATE, weights_only=False)
    sol = json.loads((out / SOLUTION).read_text())
    tile = Tile.model_validate(state["tile"])
    W, L, R, result = state["W"], state["L"], state["R"], state["result"]
    before = mahalanobis_weight_error(W, deployed_matrix(result), L, R)
    after = mahalanobis_weight_error(W, deployed_matrix(apply_tile(result, tile, sol["spins"])), L, R)
    inst = IsingInstance.from_json(cfg.instance_file)
    report = {"layer": state["layer"], "rank": state["rank"], "n": tile.n,
              "error_before": before, "error_after": after, "delta_measured": before - after,
              "delta_predicted": inst.energy(tile.signs) - inst.energy(sol["spins"]),
              "relative_change": (after - before) / before if before else 0.0}
    (out / VERIFY).write_text(json.dumps(report, indent=1))
    print(f"[verify] {state['layer']}: error {before:.6e} -> {after:.6e} "
          f"(measured delta {report['delta_measured']:.3e}, predicted {report['delta_predicted']:.3e}, "
          f"{report['relative_change']:+.2e} relative) -> {out / VERIFY}")
    return report


STAGES = {"pick-layer": pick_layer, "dump": dump, "solve": solve, "verify": verify}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in STAGES:
        print(f"usage: {Path(__file__).name} {{{'|'.join(STAGES)}}} <config.json>")
        raise SystemExit(2)
    STAGES[argv[0]](PolishConfig.from_json(argv[1]))


if __name__ == "__main__":
    main()
