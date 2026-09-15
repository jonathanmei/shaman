"""End-to-end dump -> solve -> verify of the proof-of-concept script on a synthetic 32 x 32 layer (no model)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from nanoquant.core import admm_nq

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qaoa_sign_polish.py"


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


def test_dump_solve_verify_round_trip(tmp_path):
    S = _load_script()
    cfg = S.PolishConfig(pipeline_config="unused.json", out_dir=str(tmp_path / "run"), block=3, module="mlp.down_proj",
                         n=8, shots=1500, seed=0, n_stop=4)
    W, L, R, res = _layer()
    inst = S.dump_from_layer_state(W, L, R, res, cfg, cfg.layer_key, rank=8)
    assert (tmp_path / "run" / S.INSTANCE).exists() and (tmp_path / "run" / S.LAYER_STATE).exists()
    assert inst.meta["layer"] == "3.mlp.down_proj" and inst.meta["rank"] == 8
    assert inst.meta["admm_energy"] == pytest.approx(inst.meta["layer_error_fp32"], rel=1e-3)

    summary = S.solve(cfg)
    sol = json.loads((tmp_path / "run" / S.SOLUTION).read_text())
    assert sol["n"] == 8 and len(sol["spins"]) == 8 and sol["backend"] == "statevector"
    assert sol["rqaoa_energy"] == pytest.approx(inst.energy(sol["spins"]))
    assert sol["brute_force_energy"] <= sol["rqaoa_energy"] + 1e-9
    assert sol["fraction_captured"] <= 1.0 + 1e-12
    assert sol["delta_vs_admm"] == pytest.approx(sol["admm_energy"] - sol["rqaoa_energy"])
    assert sol["hardware_jobs"] == 4 and sol["two_qubit_gates_first_step"] == 28
    assert summary["hit_optimum"] in (True, False)

    report = S.verify(cfg)
    assert report["error_after"] <= report["error_before"] * (1 + 1e-6)
    assert report["delta_measured"] == pytest.approx(report["delta_predicted"], abs=1e-3 * report["error_before"])


def test_config_requires_a_pinned_layer():
    S = _load_script()
    cfg = S.PolishConfig(pipeline_config="x.json")
    with pytest.raises(ValueError):
        _ = cfg.layer_key
    assert S.PolishConfig.from_json(Path(__file__).resolve().parents[1] / "configs" / "qaoa_sign_polish_0p6b.json").n == 9


def test_cli_usage_error():
    S = _load_script()
    with pytest.raises(SystemExit):
        S.main(["nonsense"])
