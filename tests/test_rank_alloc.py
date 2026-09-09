"""Tests for the non-uniform rank allocation (depth ramp, per-layer-type weights, bit-budget matching)."""

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


def _total_bits(model, ranks, num_scales=2):
    tot = 0
    for i, blk in enumerate(model.model.layers):
        for name, lx in U.find_layers(blk).items():
            tot += U.layer_bits(lx.in_features, lx.out_features, ranks[f"{i}.{name}"], num_scales)
    return tot


def test_defaults_reproduce_the_legacy_rule_exactly():
    ranks = U.calculate_ranks(_model(), NAMES, _cfg())
    for blk in _by_block(ranks).values():
        assert blk == LEGACY
    # parity mode without multipliers is also the legacy rule (the budget already matches)
    assert U.calculate_ranks(_model(), NAMES, _cfg(rank_budget="parity")) == ranks


def test_parse_type_weights():
    assert U.parse_type_weights("") == {}
    assert U.parse_type_weights("v_proj:1.2, down_proj:1.15,q_proj:0.9") == {"v_proj": 1.2, "down_proj": 1.15,
                                                                            "q_proj": 0.9}
    for bad in ("v_proj", "v_proj:x", "v_proj:0", "v_proj:-1"):
        with pytest.raises(ValueError):
            U.parse_type_weights(bad)


def test_multipliers_require_a_non_uniform_budget():
    with pytest.raises(ValueError):
        U.calculate_ranks(_model(), NAMES, _cfg(rank_depth_ramp=0.5))
    with pytest.raises(ValueError):
        U.calculate_ranks(_model(), NAMES, _cfg(rank_type_weights="v_proj:1.2"))
    with pytest.raises(ValueError):  # a type that does not exist in the model
        U.calculate_ranks(_model(), NAMES, _cfg(rank_budget="parity", rank_type_weights="banana:1.2"))


def test_depth_ramp_moves_bits_to_late_blocks_at_parity():
    model = _model(4)
    legacy = U.calculate_ranks(model, NAMES, _cfg())
    ranks = U.calculate_ranks(model, NAMES, _cfg(rank_budget="parity", rank_depth_ramp=0.6))
    by_blk = _by_block(ranks)
    first = sum(U.layer_bits(lx.in_features, lx.out_features, by_blk[0][n], 2)
                for n, lx in U.find_layers(model.model.layers[0]).items())
    last = sum(U.layer_bits(lx.in_features, lx.out_features, by_blk[3][n], 2)
               for n, lx in U.find_layers(model.model.layers[3]).items())
    assert last > first
    assert all(r % 32 == 0 and r >= 32 for r in ranks.values())
    # parity: the same total bits as the legacy rule, to within one 32-rank step of the widest layer
    tot, tot_legacy = _total_bits(model, ranks), _total_bits(model, legacy)
    assert tot <= tot_legacy
    assert tot_legacy - tot <= 32 * (3072 + 1024)
    assert ranks != legacy


def test_type_weights_shift_rank_between_layer_types():
    model = _model(2)
    ranks = U.calculate_ranks(model, NAMES, _cfg(rank_budget="parity", rank_type_weights="down_proj:1.3,q_proj:0.7"))
    blk = _by_block(ranks)[0]
    assert blk["mlp.down_proj"] > LEGACY["mlp.down_proj"]
    assert blk["self_attn.q_proj"] < LEGACY["self_attn.q_proj"]
    assert blk["self_attn.k_proj"] == LEGACY["self_attn.k_proj"]  # untouched type keeps ~its share
    assert all(r <= min(lx.in_features, lx.out_features) for (n, lx), r in
               zip(U.find_layers(model.model.layers[0]).items(), [blk[n] for n in U.find_layers(model.model.layers[0])]))


def test_full_budget_spends_the_rounding_remainder():
    model = _model(3)
    cfg = _cfg(rank_budget="full")
    legacy = U.calculate_ranks(model, NAMES, _cfg())
    ranks = U.calculate_ranks(model, NAMES, cfg)
    weights = sum(lx.in_features * lx.out_features for blk in model.model.layers for lx in U.find_layers(blk).values())
    tot, tot_legacy = _total_bits(model, ranks), _total_bits(model, legacy)
    assert tot_legacy < tot <= weights  # 1.0 bpw target
    assert weights - tot <= 32 * (3072 + 1024)
    assert all(ranks[k] >= legacy[k] for k in ranks)
    acc = B.static_accounting(model, NAMES, cfg)
    assert acc["factorized_bpw"] == pytest.approx(tot / weights)
    assert acc["factorized_bpw"] > 0.99


def test_rank_max_ratio_lets_late_blocks_exceed_min_dim():
    model = _model(8)
    strong = _cfg(rank_budget="parity", rank_depth_ramp=1.2, rank_type_weights="down_proj:1.3,up_proj:1.2")
    capped = U.calculate_ranks(model, NAMES, strong)
    lifted = U.calculate_ranks(model, NAMES, dict(strong, rank_max_ratio=2.0))
    last_capped, last_lifted = _by_block(capped)[7], _by_block(lifted)[7]
    assert last_capped["mlp.down_proj"] == 1024 == last_capped["mlp.up_proj"]  # min(3072, 1024), the legacy cap
    assert last_lifted["mlp.down_proj"] > 1024 and last_lifted["mlp.up_proj"] > 1024
    assert max(lifted.values()) <= 2 * 1024
    assert all(r % 32 == 0 for r in lifted.values())
    assert abs(_total_bits(model, lifted) - _total_bits(model, capped)) <= 32 * (3072 + 1024)  # same parity target
    # ratio 1.0 is the legacy cap exactly; ratios below 1 are rejected
    assert U.calculate_ranks(model, NAMES, dict(strong, rank_max_ratio=1.0)) == capped
    with pytest.raises(ValueError):
        U.calculate_ranks(model, NAMES, dict(strong, rank_max_ratio=0.5))
    # the uniform rule never reaches min(a, n), so the ratio does not change it
    assert U.calculate_ranks(model, NAMES, _cfg(rank_max_ratio=2.0)) == U.calculate_ranks(model, NAMES, _cfg())


def test_config_plumbing_and_cache_keys():
    cfg = NanoQuantConfig(model_id="t")
    assert cfg["rank_budget"] == "uniform" and cfg["rank_depth_ramp"] == 0.0 and cfg["rank_type_weights"] == ""
    assert cfg["rank_max_ratio"] == 1.0
    base = NanoQuantConfig(model_id="tiny/model", num_calib_samples=4, seqlen=16)

    def over(**kw):
        c = dict(base)
        c.update(kw)
        return c

    for field, value in (("rank_budget", "full"), ("rank_depth_ramp", 0.5), ("rank_type_weights", "v_proj:1.2"),
                         ("rank_max_ratio", 2.0)):
        assert C.chain_keys(base, 2)[0] != C.chain_keys(over(**{field: value}), 2)[0], field
    pipeline.validate_config(over(rank_budget="parity", rank_depth_ramp=0.5, rank_type_weights="v_proj:1.2",
                                  rank_max_ratio=2.0))
    for bad in ({"rank_budget": "banana"}, {"rank_depth_ramp": 0.5}, {"rank_type_weights": "v_proj:x",
                                                                       "rank_budget": "parity"},
                {"rank_max_ratio": 0.9}):
        with pytest.raises(ValueError):
            pipeline.validate_config(over(**bad))
