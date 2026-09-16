"""``num_stats_samples``: a larger calibration set for the curvature statistics (and refreshes) only.

The block-reconstruction and KD stages keep ``num_calib_samples`` (their cost is linear in samples × epochs and
their activations are held on the GPU), so the loader they receive must be byte-identical to the legacy one.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import torch

from nanoquant.core import pipeline
from nanoquant.modules.quant_config import NanoQuantConfig
from nanoquant.utils import cache as C
from nanoquant.utils import data_utils as D

_TOK = SimpleNamespace(pad_token_id=0, eos_token_id=0)


def _pool(n: int, seqlen: int = 4) -> list[dict]:
    return [{"input_ids": [i] * seqlen} for i in range(1, n + 1)]


def _cfg(**over) -> dict:
    cfg = NanoQuantConfig(model_id="tiny/model", num_calib_samples=4, seqlen=4)
    cfg.update(over)
    return cfg


# ---------------------------------------------------------------- loader
def test_calib_loader_without_replacement_uses_every_sequence_once():
    """``replace=False`` draws a permutation of the pool; the default keeps the legacy with-replacement draw."""
    pool = _pool(8)
    legacy = D.get_calib_loader(pool, _TOK, 8, 0, 4)
    assert legacy.shape == (8, 4)
    assert torch.equal(legacy, D.get_calib_loader(pool, _TOK, 8, 0, 4))  # seeded, deterministic
    distinct = D.get_calib_loader(pool, _TOK, 8, 0, 4, replace=False)
    assert distinct.shape == (8, 4)
    assert sorted(distinct[:, 0].tolist()) == list(range(1, 9))
    assert torch.equal(distinct, D.get_calib_loader(pool, _TOK, 8, 0, 4, replace=False))


# ---------------------------------------------------------------- resolution
def test_stats_sample_count_defaults_to_the_calibration_count():
    assert pipeline.stats_sample_count(_cfg()) == 4
    assert pipeline.stats_sample_count(_cfg(num_stats_samples=0)) == 4
    assert pipeline.stats_sample_count(_cfg(num_stats_samples=4)) == 4
    assert pipeline.stats_sample_count(_cfg(num_stats_samples=16)) == 16
    assert pipeline.stats_sample_count(_cfg(num_stats_samples=2)) == 4  # never fewer than the block/KD stages


def test_build_stats_loader_leaves_the_calibration_loader_untouched():
    loader = D.get_calib_loader(_pool(4), _TOK, 4, 0, 4)
    before = loader.clone()

    # default: the very same object, no dataset preparation
    with mock.patch.object(pipeline, "prepare_dataset") as prep:
        assert pipeline.build_stats_loader("tiny/model", _TOK, _cfg(), loader) is loader
    prep.assert_not_called()

    # 16 stats samples: a separate pool of 16 sequences, each used once
    with mock.patch.object(pipeline, "prepare_dataset", return_value=_pool(16)) as prep:
        stats = pipeline.build_stats_loader("tiny/model", _TOK, _cfg(num_stats_samples=16), loader)
    prep.assert_called_once()
    assert prep.call_args.args[0] == "tiny/model"
    assert prep.call_args.args[1]["num_calib_samples"] == 16
    assert stats.shape == (16, 4)
    assert sorted(stats[:, 0].tolist()) == list(range(1, 17))
    assert torch.equal(loader, before)


# ---------------------------------------------------------------- cache keys
def test_stats_key_tracks_num_stats_samples_only_when_set():
    base = _cfg()
    assert C.stats_key(base) == C.stats_key(_cfg(num_stats_samples=0))
    legacy = dict(base)
    legacy.pop("num_stats_samples", None)
    assert C.stats_key(base) == C.stats_key(legacy)  # configs written before the knob hash identically
    assert C.stats_key(base) != C.stats_key(_cfg(num_stats_samples=16))
    assert C.chain_keys(base, 2)[0] != C.chain_keys(_cfg(num_stats_samples=16), 2)[0]
    assert C.probe_key(base) != C.probe_key(_cfg(num_stats_samples=16))
