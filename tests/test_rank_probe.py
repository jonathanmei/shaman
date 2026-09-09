"""Tests for the measured-sensitivity rank allocation: power-law curve fit, the marginal-loss-per-bit allocator,
the calibration-time probe (short ADMM at candidate ranks) and the config / cache-key plumbing."""

import math
from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nanoquant.core import pipeline, rank_probe
from nanoquant.modules.quant_config import NanoQuantConfig
from nanoquant.utils import bits as B
from nanoquant.utils import cache as C
from nanoquant.utils import utils as U

QWEN3_0P6B = {"hidden": 1024, "q": 2048, "kv": 1024, "inter": 3072}
# small enough for CPU ADMM, large enough that the 0.5 / 1.0 / 1.5 probe multiples land on distinct 32-grid ranks
TINY = {"hidden": 256, "q": 384, "kv": 256, "inter": 512}
NAMES = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj',
         'mlp.up_proj', 'mlp.down_proj']


def _block(s):
    b = nn.Module()
    b.self_attn = nn.Module()
    b.mlp = nn.Module()
    b.self_attn.q_proj = nn.Linear(s["hidden"], s["q"], bias=False)
    b.self_attn.k_proj = nn.Linear(s["hidden"], s["kv"], bias=False)
    b.self_attn.v_proj = nn.Linear(s["hidden"], s["kv"], bias=False)
    b.self_attn.o_proj = nn.Linear(s["q"], s["hidden"], bias=False)
    b.mlp.gate_proj = nn.Linear(s["hidden"], s["inter"], bias=False)
    b.mlp.up_proj = nn.Linear(s["hidden"], s["inter"], bias=False)
    b.mlp.down_proj = nn.Linear(s["inter"], s["hidden"], bias=False)
    return b


def _model(n_blocks=4, shapes=QWEN3_0P6B):
    return SimpleNamespace(config=SimpleNamespace(model_type="qwen3"),
                           model=SimpleNamespace(layers=[_block(shapes) for _ in range(n_blocks)]))


def _cfg(**over):
    cfg = {"bits": 1.0, "admm_type": "nanoquant", "admm_mid_scale": False, "seed": 0}
    cfg.update(over)
    return cfg


def _shapes(model, names=NAMES):
    out = {}
    for i, blk in enumerate(model.model.layers):
        sub = U.find_layers(blk)
        for n in names:
            out[f"{i}.{n}"] = (sub[n].in_features, sub[n].out_features)
    return out


def _total_bits(shapes, ranks, num_scales=2):
    return sum(U.layer_bits(a, b, ranks[k], num_scales) for k, (a, b) in shapes.items())


def _flat_curves(shapes, log_a=0.0, beta=1.0):
    return {k: (log_a, beta) for k in shapes}


# --------------------------------------------------------------------------------------
# power-law fit
# --------------------------------------------------------------------------------------
def test_fit_power_law_recovers_exponent_and_clamps():
    log_a, beta = 3.0, 1.7
    probes = {r: math.exp(log_a) * r ** (-beta) for r in (128, 256, 384)}
    a_hat, b_hat = U.fit_power_law(probes)
    assert a_hat == pytest.approx(log_a, abs=1e-6)
    assert b_hat == pytest.approx(beta, abs=1e-6)
    assert U.predicted_loss((a_hat, b_hat), 256) == pytest.approx(probes[256], rel=1e-6)
    # a curve that (through noise) increases with rank is clamped to the minimum slope, never negative
    _, b_up = U.fit_power_law({128: 1.0, 256: 2.0})
    assert b_up == U.BETA_RANGE[0]
    # a single probe cannot determine the slope: default exponent, level from the point
    a1, b1 = U.fit_power_law({256: 5.0})
    assert b1 == U.BETA_DEFAULT and U.predicted_loss((a1, b1), 256) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        U.fit_power_law({})
    with pytest.raises(ValueError):
        U.fit_power_law({128: 0.0, 256: -1.0})


# --------------------------------------------------------------------------------------
# allocator
# --------------------------------------------------------------------------------------
def test_measured_allocation_meets_the_parity_budget_and_the_grid():
    model = _model(3)
    shapes = _shapes(model)
    legacy = U.calculate_ranks(model, NAMES, _cfg())
    ranks = U.allocate_ranks_measured(shapes, _flat_curves(shapes), 1.0, 2, "parity", legacy)
    assert set(ranks) == set(shapes)
    assert all(r % 32 == 0 and 32 <= r <= min(a, b) for (k, (a, b)), r in zip(shapes.items(), [ranks[k] for k in shapes]))
    tot, tot_legacy = _total_bits(shapes, ranks), _total_bits(shapes, legacy)
    assert tot <= tot_legacy
    assert tot_legacy - tot <= 32 * (3072 + 1024)  # within one step of the widest layer


def test_measured_allocation_full_budget_spends_the_remainder():
    model = _model(2)
    shapes = _shapes(model)
    legacy = U.calculate_ranks(model, NAMES, _cfg())
    ranks = U.allocate_ranks_measured(shapes, _flat_curves(shapes), 1.0, 2, "full", legacy)
    weights = sum(a * b for a, b in shapes.values())
    tot = _total_bits(shapes, ranks)
    assert _total_bits(shapes, legacy) < tot <= weights
    assert weights - tot <= 32 * (3072 + 1024)


def test_steeper_curve_wins_rank_at_identical_bits():
    model = _model(2)
    shapes = _shapes(model)
    legacy = U.calculate_ranks(model, NAMES, _cfg())
    curves = _flat_curves(shapes, log_a=0.0, beta=1.0)
    # same loss at the uniform rank (736) but three times the slope there: more to gain per rank step
    curves["1.mlp.down_proj"] = (2.0 * math.log(736), 3.0)
    ranks = U.allocate_ranks_measured(shapes, curves, 1.0, 2, "parity", legacy)
    assert ranks["1.mlp.down_proj"] > ranks["0.mlp.down_proj"]
    # a higher level (same slope) also wins: the loss at stake is larger
    curves = _flat_curves(shapes)
    curves["0.self_attn.v_proj"] = (4.0, 1.0)
    ranks = U.allocate_ranks_measured(shapes, curves, 1.0, 2, "parity", legacy)
    assert ranks["0.self_attn.v_proj"] > ranks["1.self_attn.v_proj"]


def test_measured_allocation_respects_ceiling_and_missing_curves():
    model = _model(2)
    shapes = _shapes(model)
    legacy = U.calculate_ranks(model, NAMES, _cfg())
    curves = _flat_curves(shapes, beta=0.2)
    curves["1.mlp.up_proj"] = (30.0, 1.0)  # e^30 / r: its marginal gain dominates every other layer at any rank
    capped = U.allocate_ranks_measured(shapes, curves, 1.0, 2, "parity", legacy)
    assert capped["1.mlp.up_proj"] == 1024  # min(1024, 3072)
    lifted = U.allocate_ranks_measured(shapes, curves, 1.0, 2, "parity", legacy, max_ratio=2.0)
    assert 1024 < lifted["1.mlp.up_proj"] <= 2048 and lifted["1.mlp.up_proj"] % 32 == 0
    # a layer without a curve keeps its legacy rank and its bits stay reserved
    del curves["0.self_attn.k_proj"]
    ranks = U.allocate_ranks_measured(shapes, curves, 1.0, 2, "parity", legacy)
    assert ranks["0.self_attn.k_proj"] == legacy["0.self_attn.k_proj"]
    assert _total_bits(shapes, ranks) <= _total_bits(shapes, legacy)
    with pytest.raises(ValueError):
        U.allocate_ranks_measured(shapes, curves, 1.0, 2, "uniform", legacy)


def test_prior_multipliers_shift_the_measured_allocation():
    model = _model(4)
    shapes = _shapes(model)
    legacy = U.calculate_ranks(model, NAMES, _cfg())
    curves = _flat_curves(shapes)
    plain = U.allocate_ranks_measured(shapes, curves, 1.0, 2, "parity", legacy)
    mult = {k: math.exp(0.6 * (int(k.split(".", 1)[0]) / 3 - 0.5)) for k in shapes}
    ramped = U.allocate_ranks_measured(shapes, curves, 1.0, 2, "parity", legacy, mult=mult)

    def blk_bits(ranks, i):
        return sum(U.layer_bits(a, b, ranks[k], 2) for k, (a, b) in shapes.items() if k.startswith(f"{i}."))

    assert blk_bits(ramped, 3) > blk_bits(plain, 3)
    assert blk_bits(ramped, 0) < blk_bits(plain, 0)


def test_calculate_ranks_measured_mode_and_fallback(capsys):
    model = _model(2)
    shapes = _shapes(model)
    cfg = _cfg(rank_budget="parity", rank_sensitivity="admm")
    legacy = U.calculate_ranks(model, NAMES, _cfg())
    # before the probe has run (static accounting ahead of calibration) the legacy rule is returned
    assert U.calculate_ranks(model, NAMES, cfg) == legacy
    assert "pending" in capsys.readouterr().out
    curves = _flat_curves(shapes)
    curves["1.mlp.down_proj"] = (3.0 * math.log(736), 3.0)  # J(736) = 1, 736x the flat curves' level, steeper
    sens = {"curves": curves, "probes": {k: {} for k in shapes}}
    ranks = U.calculate_ranks(model, NAMES, cfg, sensitivity=sens)
    assert ranks["1.mlp.down_proj"] > legacy["1.mlp.down_proj"]
    assert ranks == U.allocate_ranks_measured(shapes, curves, 1.0, 2, "parity", legacy)
    # the accounting threads the same argument
    acc = B.static_accounting(model, NAMES, cfg, sensitivity=sens)
    assert acc["layers"]["1.mlp.down_proj"]["rank"] == ranks["1.mlp.down_proj"]
    # measured mode needs a non-uniform budget
    with pytest.raises(ValueError):
        U.calculate_ranks(model, NAMES, _cfg(rank_sensitivity="admm"), sensitivity=sens)
    with pytest.raises(ValueError):
        U.calculate_ranks(model, NAMES, _cfg(rank_budget="parity", rank_sensitivity="banana"), sensitivity=sens)


# --------------------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------------------
def test_parse_probe_ranks_and_candidate_grid():
    assert U.parse_probe_ranks("0.5,1.0,1.5") == [0.5, 1.0, 1.5]
    assert U.parse_probe_ranks(" 1 ") == [1.0]
    for bad in ("", "0,1", "a", "-1"):
        with pytest.raises(ValueError):
            U.parse_probe_ranks(bad)
    # candidates: multiples of the uniform rank on the 32-grid, clamped to [32, ceiling], de-duplicated, sorted
    assert rank_probe.candidate_ranks(640, 1024, 2048, [0.5, 1.0, 1.5], 1.0) == [320, 640, 960]
    assert rank_probe.candidate_ranks(736, 3072, 1024, [0.5, 1.0, 1.5], 1.0) == [352, 736, 1024]
    assert rank_probe.candidate_ranks(736, 3072, 1024, [0.5, 1.0, 1.5], 2.0) == [352, 736, 1088]
    assert rank_probe.candidate_ranks(32, 64, 64, [0.5, 1.0, 1.5], 1.0) == [32]  # 48 floors to 32: one probe
    assert rank_probe.candidate_ranks(128, 256, 384, [0.5, 1.0, 1.5], 1.0) == [64, 128, 192]


def test_deployed_matrix_matches_the_module_forward():
    torch.manual_seed(0)
    out_f, in_f, rank = 12, 10, 4
    res = {"A": torch.randn(rank, out_f), "B": torch.randn(rank, in_f), "scale_pre": torch.rand(1, in_f) + 0.5,
           "scale_post": torch.rand(1, out_f) + 0.5}
    W_hat = rank_probe.deployed_matrix(res)
    x = torch.randn(3, in_f)
    y = torch.nn.functional.linear(x * res["scale_pre"], res["B"].sign())
    y = torch.nn.functional.linear(y, res["A"].mT.sign()) * res["scale_post"]
    assert torch.allclose(x @ W_hat.mT, y, atol=1e-5)
    res["scale_mid"] = torch.rand(1, rank) + 0.5
    W_hat3 = rank_probe.deployed_matrix(res)
    y3 = torch.nn.functional.linear(x * res["scale_pre"], res["B"].sign()) * res["scale_mid"]
    y3 = torch.nn.functional.linear(y3, res["A"].mT.sign()) * res["scale_post"]
    assert torch.allclose(x @ W_hat3.mT, y3, atol=1e-5)


def _attach_stats(model, dense: bool):
    for blk in model.model.layers:
        for lx in U.find_layers(blk).values():
            a, b = lx.in_features, lx.out_features
            lx.i_norm = torch.full((a,), 2.0)
            lx.o_norm = torch.full((b,), 0.5)
            if dense:
                lx.i_cov = torch.diag(lx.i_norm)
                lx.o_cov = torch.diag(lx.o_norm)


@pytest.mark.parametrize("dense", [False, True])
def test_measure_sensitivity_on_a_tiny_model(dense):
    torch.manual_seed(0)
    model = _model(2, TINY)
    _attach_stats(model, dense)
    cfg = _cfg(rank_budget="parity", rank_sensitivity="admm", rank_probe_ranks="0.5,1.0,1.5", rank_probe_iters=20,
               admm_outer_iters=400, admm_inner_iters=5, admm_reg=3e-2, admm_penalty_scheduler="linear",
               admm_print_steps=False, curvature="kron" if dense else "diag", kron_eigh_dtype="float64",
               admm_curvature_power=1.0, admm_curvature_cond_max=0.0, admm_curvature_spike_rank=0,
               rank_max_ratio=1.0)
    before = {f"{i}.{k}": lx.weight.detach().clone()
              for i, blk in enumerate(model.model.layers) for k, lx in U.find_layers(blk).items()}
    sens = rank_probe.measure_sensitivity(model, NAMES, cfg, dev="cpu")
    shapes = _shapes(model)
    assert set(sens["probes"]) == set(shapes) == set(sens["curves"])
    for k, (a, b) in shapes.items():
        probes = sens["probes"][k]
        ranks = sorted(probes)
        assert ranks == rank_probe.candidate_ranks(U.calculate_ranks(model, NAMES, _cfg())[k], a, b, [0.5, 1.0, 1.5], 1.0)
        assert all(v > 0 for v in probes.values())
        # more rank, less curvature-weighted error (monotone up to ADMM noise at these tiny sizes)
        assert probes[ranks[0]] > probes[ranks[-1]]
        _, beta = sens["curves"][k]
        assert U.BETA_RANGE[0] <= beta <= U.BETA_RANGE[1]
    # the modules are untouched: still nn.Linear, weights and statistics intact
    for i, blk in enumerate(model.model.layers):
        for k, lx in U.find_layers(blk).items():
            assert type(lx) is nn.Linear
            assert torch.equal(lx.weight, before[f"{i}.{k}"])
            assert hasattr(lx, "i_norm") and hasattr(lx, "o_norm")
            assert hasattr(lx, "i_cov") == dense
    assert sens["meta"]["n_layers"] == len(shapes) and sens["meta"]["seconds"] >= 0
    # the allocator consumes the artifact end to end
    ranks = U.calculate_ranks(model, NAMES, cfg, sensitivity=sens)
    assert set(ranks) == set(shapes) and all(r % 32 == 0 for r in ranks.values())


def test_svd_sensitivity_proxy_is_a_decreasing_curve():
    torch.manual_seed(0)
    model = _model(1, TINY)
    _attach_stats(model, dense=True)
    cfg = _cfg(rank_budget="parity", rank_sensitivity="svd", rank_probe_ranks="0.5,1.0,1.5", curvature="kron")
    sens = rank_probe.measure_sensitivity(model, NAMES, cfg, dev="cpu")
    for probes in sens["probes"].values():
        ranks = sorted(probes)
        assert all(probes[r1] > probes[r2] for r1, r2 in pairwise(ranks))


# --------------------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------------------
def test_config_plumbing_and_cache_keys():
    cfg = NanoQuantConfig(model_id="t")
    assert cfg["rank_sensitivity"] == "none" and cfg["rank_probe_ranks"] == "0.5,1.0,1.5"
    assert cfg["rank_probe_iters"] == 50
    base = NanoQuantConfig(model_id="tiny/model", num_calib_samples=4, seqlen=16)

    def over(**kw):
        c = dict(base)
        c.update(kw)
        return c

    for field, value in (("rank_sensitivity", "admm"), ("rank_probe_ranks", "0.75,1.25"), ("rank_probe_iters", 10)):
        assert C.chain_keys(base, 2)[0] != C.chain_keys(over(**{field: value}), 2)[0], field
        assert C.probe_key(base) != C.probe_key(over(**{field: value})), field
    # the probe key ignores the production iteration count and everything downstream of the ADMM inputs
    assert C.probe_key(base) == C.probe_key(over(admm_outer_iters=7, tail_logit_blocks=4, model_kd_lr=1.0))
    assert C.probe_key(base) != C.probe_key(over(calib_shrinkage=0.1))
    assert C.probe_key(base) != C.probe_key(over(bits=0.9))
    pipeline.validate_config(over(rank_budget="parity", rank_sensitivity="admm"))
    pipeline.validate_config(over(rank_budget="full", rank_sensitivity="svd", rank_depth_ramp=0.6))
    for bad in ({"rank_sensitivity": "admm"}, {"rank_sensitivity": "banana", "rank_budget": "parity"},
                {"rank_sensitivity": "admm", "rank_budget": "parity", "rank_probe_ranks": "0,1"},
                {"rank_sensitivity": "admm", "rank_budget": "parity", "rank_probe_iters": 0}):
        with pytest.raises(ValueError):
            pipeline.validate_config(over(**bad))
