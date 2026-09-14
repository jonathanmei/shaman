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
    for gone in ("rank_type_weights", "rank_max_ratio", "block_loss", "tail_logit_blocks", "retain_latent",
                 "model_kd_mode", "model_kd_feature_weight", "admm_curvature_spike_rank"):
        assert gone not in cfg
    base = NanoQuantConfig(model_id="tiny/model", num_calib_samples=4, seqlen=16)

    def over(**kw):
        c = dict(base)
        c.update(kw)
        return c

    for field, value in (("rank_budget", "full"), ("rank_depth_ramp", 0.5), ("rank_sensitivity", "admm"),
                         ("rank_probe_ranks", "0.75,1.25"), ("rank_probe_iters", 10)):
        assert C.chain_keys(base, 2)[0] != C.chain_keys(over(**{field: value}), 2)[0], field
    pipeline.validate_config(over(rank_budget="parity", rank_sensitivity="admm", rank_depth_ramp=0.6))
    pipeline.validate_config(over(rank_budget="full", rank_sensitivity="svd"))
    for bad in ({"rank_budget": "banana"}, {"rank_depth_ramp": 0.5}, {"rank_budget": "parity"},
                {"rank_sensitivity": "admm"}, {"rank_budget": "parity", "rank_sensitivity": "banana"}):
        with pytest.raises(ValueError):
            pipeline.validate_config(over(**bad))
