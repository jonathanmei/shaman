"""Tests for the late-block, high-sensitivity layer picker (dict inputs, no model)."""

import math

import pytest

from nanoquant.quantum import pick_layer as P
from nanoquant.utils import utils as U

NAMES = ["self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj"]


def _curves(n_blocks: int) -> dict[str, tuple[float, float]]:
    """Flat level ``a = 0`` everywhere except a rising trend in the last blocks and one huge early outlier."""
    curves = {}
    for b in range(n_blocks):
        for nm in NAMES:
            a = 0.0
            if nm == "mlp.down_proj":
                a = 0.3
            if b >= n_blocks - 3 and nm == "self_attn.v_proj":
                a = 0.5 + 0.1 * (b - (n_blocks - 3))
            curves[f"{b}.{nm}"] = (a, 0.9)
    curves["2.mlp.down_proj"] = (math.log(200.0), 0.9)  # pathological Fisher (100x level)
    return curves


def test_ranking_is_restricted_to_late_blocks_and_sorted():
    n_blocks = 12
    curves = _curves(n_blocks)
    ranks = {k: 256 for k in curves}
    scores = P.rank_late_layers(curves, ranks, n_blocks, depth_ramp=0.6)
    assert scores
    assert all(s.block >= n_blocks - n_blocks // 3 for s in scores)
    assert [s.score for s in scores] == sorted((s.score for s in scores), reverse=True)
    assert scores[0].key == f"{n_blocks - 1}.self_attn.v_proj"
    assert scores[0].name == "self_attn.v_proj" and scores[0].rank == 256


def test_score_is_depth_adjusted_predicted_loss_at_allocated_rank():
    n_blocks = 12
    curves = _curves(n_blocks)
    ranks = {k: 512 if k.endswith("v_proj") else 256 for k in curves}
    scores = {s.key: s for s in P.rank_late_layers(curves, ranks, n_blocks, depth_ramp=0.6)}
    key = "11.self_attn.v_proj"
    mult = U.depth_multipliers({key: (0, 0)}, n_blocks, 0.6)[key]
    a, beta = curves[key]
    assert scores[key].score == pytest.approx(U.predicted_loss((a + math.log(mult), beta), 512))
    assert scores[key].level == pytest.approx(math.exp(a) * mult)


def test_outlier_is_dropped_when_it_falls_in_the_window():
    n_blocks = 4  # last third is block 3 only; put the outlier there
    curves = _curves(n_blocks)
    curves["3.mlp.down_proj"] = (math.log(200.0), 0.9)
    ranks = {k: 256 for k in curves}
    scores = P.rank_late_layers(curves, ranks, n_blocks, depth_ramp=0.0)
    keys = [s.key for s in scores]
    assert "3.mlp.down_proj" not in keys
    assert keys[0] == "3.self_attn.v_proj"
    kept = P.rank_late_layers(curves, ranks, n_blocks, depth_ramp=0.0, outlier_factor=1e9)
    assert kept[0].key == "3.mlp.down_proj"


def test_layers_without_a_curve_or_rank_are_skipped():
    curves = {"5.mlp.down_proj": (0.0, 0.9), "5.self_attn.q_proj": (0.0, 0.9)}
    ranks = {"5.mlp.down_proj": 128}
    scores = P.rank_late_layers(curves, ranks, n_blocks=6, depth_ramp=0.0)
    assert [s.key for s in scores] == ["5.mlp.down_proj"]
