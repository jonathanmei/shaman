"""Tests for the rank budget rule: the paper's uniform rule and the plumbing of the measured allocation knobs."""

from types import SimpleNamespace

import pytest
from torch import nn

from nanoquant.core import pipeline
from nanoquant.modules.quant_config import NanoQuantConfig
from nanoquant.utils import bits as B
from nanoquant.utils import cache as C
from nanoquant.utils import utils as U

QWEN3_0P6B = {"hidden": 1024, "q": 2048, "kv": 1024, "inter": 3072}
NAMES = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj',
         'mlp.up_proj', 'mlp.down_proj']
LEGACY = {"self_attn.q_proj": 640, "self_attn.k_proj": 480, "self_attn.v_proj": 480, "self_attn.o_proj": 640,
          "mlp.gate_proj": 736, "mlp.up_proj": 736, "mlp.down_proj": 736}


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


def _model(n_blocks=4):
    return SimpleNamespace(config=SimpleNamespace(model_type="qwen3"),
                           model=SimpleNamespace(layers=[_block(QWEN3_0P6B) for _ in range(n_blocks)]))


def _cfg(**over):
    cfg = {"bits": 1.0, "admm_type": "nanoquant", "admm_mid_scale": False}
    cfg.update(over)
    return cfg


def _by_block(ranks):
    out = {}
    for k, r in ranks.items():
        blk, name = k.split(".", 1)
        out.setdefault(int(blk), {})[name] = r
    return out


def test_defaults_reproduce_the_uniform_rule_exactly():
    ranks = U.calculate_ranks(_model(), NAMES, _cfg())
    for blk in _by_block(ranks).values():
        assert blk == LEGACY
    assert all(U.uniform_rank(1024, 2048, 1.0, 2) == 640 for _ in range(1))
    assert U.rank_ceiling(3072, 1024) == 1024
    acc = B.static_accounting(_model(2), NAMES, _cfg())
    assert acc["factorized_bpw"] == pytest.approx(0.9729, abs=5e-4)


def test_non_uniform_budget_requires_a_measured_sensitivity():
    with pytest.raises(ValueError):
        U.calculate_ranks(_model(), NAMES, _cfg(rank_budget="parity"))
    with pytest.raises(ValueError):
        U.calculate_ranks(_model(), NAMES, _cfg(rank_budget="full"))
    with pytest.raises(ValueError):  # the depth prior alone does not define an allocation
        U.calculate_ranks(_model(), NAMES, _cfg(rank_depth_ramp=0.6))
    with pytest.raises(ValueError):
        U.calculate_ranks(_model(), NAMES, _cfg(rank_budget="banana"))
    with pytest.raises(ValueError):
        U.calculate_ranks(_model(), NAMES, _cfg(rank_budget="parity", rank_sensitivity="banana"))


def test_depth_multipliers_are_a_geometric_ramp():
    shapes = {f"{b}.x": (8, 8) for b in range(4)}
    flat = U.depth_multipliers(shapes, 4, 0.0)
    assert all(v == pytest.approx(1.0) for v in flat.values())
    ramp = U.depth_multipliers(shapes, 4, 0.6)
    assert ramp["3.x"] / ramp["0.x"] == pytest.approx(pytest.approx(2.718281828 ** 0.6))
    assert ramp["1.x"] < ramp["2.x"]
    assert U.depth_multipliers({"0.x": (8, 8)}, 1, 0.6) == {"0.x": 1.0}


def test_config_plumbing_and_cache_keys():
    cfg = NanoQuantConfig(model_id="t")
    assert cfg["rank_budget"] == "uniform" and cfg["rank_sensitivity"] == "none" and cfg["rank_depth_ramp"] == 0.0
    assert cfg["rank_probe_ranks"] == "0.5,1.0,1.5" and cfg["rank_probe_iters"] == 50
    assert cfg["rank_type_weights"] == ""
    for gone in ("rank_max_ratio", "block_loss", "tail_logit_blocks", "retain_latent",
                 "model_kd_mode", "model_kd_feature_weight", "admm_curvature_cond_max"):
        assert gone not in cfg
    base = NanoQuantConfig(model_id="tiny/model", num_calib_samples=4, seqlen=16)

    def over(**kw):
        c = dict(base)
        c.update(kw)
        return c

    for field, value in (("rank_budget", "full"), ("rank_depth_ramp", 0.5), ("rank_sensitivity", "admm"),
                         ("rank_probe_ranks", "0.75,1.25"), ("rank_probe_iters", 10),
                         ("rank_type_weights", "down_proj:1.15")):
        assert C.chain_keys(base, 2)[0] != C.chain_keys(over(**{field: value}), 2)[0], field
    # the type table is applied to the fitted curves after the probe: the probe artifact stays reusable
    assert C.probe_key(base) == C.probe_key(over(rank_type_weights="down_proj:1.15"))
    pipeline.validate_config(over(rank_budget="parity", rank_sensitivity="admm", rank_depth_ramp=0.6))
    pipeline.validate_config(over(rank_budget="full", rank_sensitivity="svd"))
    pipeline.validate_config(over(rank_budget="parity", rank_sensitivity="admm", rank_type_weights="v_proj:1.1"))
    for bad in ({"rank_budget": "banana"}, {"rank_depth_ramp": 0.5}, {"rank_budget": "parity"},
                {"rank_sensitivity": "admm"}, {"rank_budget": "parity", "rank_sensitivity": "banana"},
                {"rank_type_weights": "v_proj:1.1"},  # a prior alone does not define an allocation
                {"rank_budget": "parity", "rank_sensitivity": "admm", "rank_type_weights": "v_proj"}):
        with pytest.raises(ValueError):
            pipeline.validate_config(over(**bad))


def test_parse_type_weights():
    assert U.parse_type_weights("") == {}
    assert U.parse_type_weights("v_proj:1.2, down_proj:1.15,q_proj:0.9") == {"v_proj": 1.2, "down_proj": 1.15,
                                                                            "q_proj": 0.9}
    for bad in ("v_proj", "v_proj:x", "v_proj:0", "v_proj:-1"):
        with pytest.raises(ValueError):
            U.parse_type_weights(bad)


def _flat_curves(model, beta=1.0):
    """Identical sensitivity curves for every layer, so only the priors and shapes decide the allocation."""
    curves = {}
    for i, _ in enumerate(model.model.layers):
        for name in NAMES:
            curves[f"{i}.{name}"] = (0.0, beta)
    return {"curves": curves}


def test_type_weights_shift_measured_allocation_between_layer_types():
    model = _model(2)
    sens = _flat_curves(model)
    base = _cfg(rank_budget="parity", rank_sensitivity="admm")
    plain = U.calculate_ranks(model, NAMES, base, sensitivity=sens)
    same = U.calculate_ranks(model, NAMES, dict(base, rank_type_weights=""), sensitivity=sens)
    assert same == plain
    tilted = U.calculate_ranks(model, NAMES, dict(base, rank_type_weights="down_proj:1.3,q_proj:0.7"),
                               sensitivity=sens)
    p, t = _by_block(plain), _by_block(tilted)
    assert all(t[b]["mlp.down_proj"] >= p[b]["mlp.down_proj"] for b in p)
    assert all(t[b]["self_attn.q_proj"] <= p[b]["self_attn.q_proj"] for b in p)
    assert sum(t[b]["mlp.down_proj"] for b in t) > sum(p[b]["mlp.down_proj"] for b in p)
    assert sum(t[b]["self_attn.q_proj"] for b in t) < sum(p[b]["self_attn.q_proj"] for b in p)
    # same bit target (parity): the totals agree up to one 32-step of the largest layer
    bits_p = sum(U.layer_bits(*QWEN3_SHAPES[k.split('.', 1)[1]], r, 2) for k, r in plain.items())
    bits_t = sum(U.layer_bits(*QWEN3_SHAPES[k.split('.', 1)[1]], r, 2) for k, r in tilted.items())
    assert abs(bits_p - bits_t) <= 32 * (3072 + 1024)
    assert all(r % 32 == 0 for r in tilted.values())
    # the depth ramp and the type table compose multiplicatively
    both = U.calculate_ranks(model, NAMES, dict(base, rank_depth_ramp=0.6, rank_type_weights="down_proj:1.3"),
                             sensitivity=sens)
    ramp_only = U.calculate_ranks(model, NAMES, dict(base, rank_depth_ramp=0.6), sensitivity=sens)
    assert sum(_by_block(both)[b]["mlp.down_proj"] for b in (0, 1)) > \
        sum(_by_block(ramp_only)[b]["mlp.down_proj"] for b in (0, 1))
    with pytest.raises(ValueError):  # unknown layer type is an error, not a silent no-op
        U.calculate_ranks(model, NAMES, dict(base, rank_type_weights="banana_proj:1.3"), sensitivity=sens)


QWEN3_SHAPES = {"self_attn.q_proj": (1024, 2048), "self_attn.k_proj": (1024, 1024), "self_attn.v_proj": (1024, 1024),
                "self_attn.o_proj": (2048, 1024), "mlp.gate_proj": (1024, 3072), "mlp.up_proj": (1024, 3072),
                "mlp.down_proj": (3072, 1024)}
