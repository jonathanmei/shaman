"""End-to-end stages of the proof-of-concept script on a synthetic 32 x 32 layer (no model)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from nanoquant.core import admm_nq

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qaoa_sign_polish.py"
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _load_script():
    if "qaoa_sign_polish" in sys.modules:
        return sys.modules["qaoa_sign_polish"]
    spec = importlib.util.spec_from_file_location("qaoa_sign_polish", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # pydantic resolves the postponed annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _layer(seed: int = 0, corr: float = 0.6):
    torch.manual_seed(seed)
    n_out, n_in = 32, 32
    W = torch.randn(n_out, n_in)
    i_norm, o_norm = torch.rand(n_in) + 0.5, torch.rand(n_out) + 0.5
    C_o = (1 - corr) * torch.eye(n_out) + corr * torch.ones(n_out, n_out)
    C_i = (1 - corr) * torch.eye(n_in) + corr * torch.ones(n_in, n_in)
    o_cov = C_o * o_norm.sqrt().unsqueeze(1) * o_norm.sqrt().unsqueeze(0)
    i_cov = C_i * i_norm.sqrt().unsqueeze(1) * i_norm.sqrt().unsqueeze(0)
    res = admm_nq.factorize_admm_nanoquant(W, i_norm, o_norm, mid_rank=8, outer_iters=30, print_admm_steps=False,
                                           rho_scheduler="linear", i_cov=i_cov, o_cov=o_cov)
    return W, o_cov, i_cov, res


def test_dump_solve_verify_round_trip_statevector(tmp_path):
    S = _load_script()
    cfg = S.PolishConfig(pipeline_config="unused.json", out_dir=str(tmp_path / "run"), block=3, module="mlp.down_proj",
                         n=8, shots=1500, seed=0, n_stop=4)
    W, L, R, res = _layer()
    inst = S.dump_from_layer_state(W, L, R, res, cfg, cfg.layer_key, rank=8)
    assert (tmp_path / "run" / S.INSTANCE).exists() and (tmp_path / "run" / S.LAYER_STATE).exists()
    assert inst.meta["layer"] == "3.mlp.down_proj" and inst.meta["rank"] == 8
    assert inst.meta["admm_energy"] == pytest.approx(inst.meta["layer_error_fp32"], rel=1e-3)

    summaries = S.solve(cfg)
    assert len(summaries) == 1
    sol = json.loads((tmp_path / "run" / S.SOLUTION).read_text())
    assert sol["n"] == 8 and sol["p"] == 1 and len(sol["spins"]) == 8 and sol["backend"] == "statevector"
    assert sol["reference"] == "brute_force"
    assert sol["rqaoa_energy"] == pytest.approx(inst.energy(sol["spins"]))
    assert sol["reference_energy"] <= sol["rqaoa_energy"] + 1e-9
    assert sol["fraction_captured"] <= 1.0 + 1e-12
    assert sol["delta_vs_admm"] == pytest.approx(sol["admm_energy"] - sol["rqaoa_energy"])
    assert sol["hardware_jobs"] == 4 and sol["two_qubit_gates_per_layer"] == 28
    assert len(sol["seconds_per_step"]) == 4 and len(sol["normalized_angles_step0"]) == 2

    reports = S.verify(cfg)
    report = reports[0]
    assert report["error_after"] <= report["error_before"] * (1 + 1e-6)
    assert report["delta_measured"] == pytest.approx(report["delta_predicted"], abs=1e-3 * report["error_before"])


def test_all_with_pauli_backend_above_the_statevector_limit(tmp_path):
    """24 bits, depths 1 and 2: classical references instead of brute force, per-depth files, angle transfer."""
    S = _load_script()
    out = tmp_path / "run24"
    cfg = S.PolishConfig(pipeline_config="unused.json", out_dir=str(out), block=3, module="mlp.down_proj", n=24,
                         depths=[1, 2], backend="pauli", prop_weight=1, max_active=6, pair_top=4, n_starts=1,
                         reoptimize_every=100, shots=0, seed=0, n_stop=4)
    W, L, R, res = _layer(seed=1)
    S.dump_from_layer_state(W, L, R, res, cfg, cfg.layer_key, rank=8)
    inst = S.IsingInstance.from_json(out / S.INSTANCE)
    refs = S.classical_references(inst, inst.meta["admm_signs"], seed=0)
    assert refs["reference"] == "simulated_annealing"
    assert refs["reference_energy"] <= inst.energy(inst.meta["admm_signs"]) + 1e-12
    for p in (1, 2):
        s = S.solve_depth(cfg, inst, p, refs)
        assert (out / S.solution_name(p)).exists() and s["p"] == p
        assert s["rqaoa_energy"] == pytest.approx(inst.energy(s["spins"]))
        assert s["steps"][0]["reoptimized"] and not any(st["reoptimized"] for st in s["steps"][1:])
        r = S.verify_depth(cfg, p)
        assert r["delta_measured"] == pytest.approx(r["delta_predicted"], abs=1e-3 * r["error_before"])
    # transfer the depth-2 angles into a fixed-angle re-solve
    cfg2 = cfg.model_copy(update={"angles_from": str(out / S.solution_name(2)), "depths": [2],
                                  "out_dir": str(tmp_path / "run24_fixed"), "instance_path": str(out / S.INSTANCE)})
    (tmp_path / "run24_fixed").mkdir()
    s2 = S.solve_depth(cfg2, inst, 2, refs)
    assert not any(st["reoptimized"] for st in s2["steps"])
    assert s2["steps"][0]["gammas"] == pytest.approx(json.loads((out / S.solution_name(2)).read_text())["steps"][0]["gammas"])


def test_crosscheck_stage_on_a_synthetic_instance(tmp_path):
    pytest.importorskip("quimb")
    S = _load_script()
    out = tmp_path / "run12"
    cfg = S.PolishConfig(pipeline_config="unused.json", out_dir=str(out), block=3, module="mlp.down_proj", n=12,
                         depths=[1, 2], sub_tile=8, mps_bonds=[4, 16], seed=0)
    W, L, R, res = _layer(seed=2)
    S.dump_from_layer_state(W, L, R, res, cfg, cfg.layer_key, rank=8)
    report = S.crosscheck(cfg)
    assert report["n_full"] == 12 and report["n_sub"] == 8
    for p in ("1", "2"):
        sub = report["depths"][p]["sub"]
        assert sub["pauli_k0"]["zz_max_abs_err"] < (1e-9 if p == "1" else 1.0)  # exact at p = 1 only
        assert sub["mps_chi16"]["zz_max_abs_err"] < 1e-4  # chi = 16 is exact on 8 qubits up to the SVD cutoff
        assert sub["mps_chi4"]["zz_max_abs_err"] > sub["mps_chi16"]["zz_max_abs_err"]
        assert "vs_pauli_k2_zz_max_abs_err" in report["depths"][p]["full"]["mps_chi16"]
    p2 = report["depths"]["2"]["sub"]
    assert p2["pauli_k0"]["zz_max_abs_err"] > p2["pauli_k1"]["zz_max_abs_err"] > p2["pauli_k2"]["zz_max_abs_err"]
    assert (out / S.CROSSCHECK).exists()


def test_config_requires_a_pinned_layer_and_ships_configs():
    S = _load_script()
    cfg = S.PolishConfig(pipeline_config="x.json")
    with pytest.raises(ValueError):
        _ = cfg.layer_key
    for name, n, depths in [("qaoa_sign_polish_0p6b.json", 9, [1]), ("qaoa_sign_polish_0p6b_n64_p123.json", 64, [1, 2, 3]),
                            ("qaoa_sign_polish_0p6b_n256_p12.json", 256, [1, 2]),
                            ("qaoa_sign_polish_0p6b_n32_crosscheck.json", 32, [1, 2, 3])]:
        c = S.PolishConfig.from_json(CONFIGS / name)
        assert c.n == n and c.depths == depths and c.layer_key == "27.mlp.down_proj"


def test_cli_usage_error():
    S = _load_script()
    with pytest.raises(SystemExit):
        S.main(["nonsense"])
