"""Tests for the content-addressed artifact cache (keys, storage, ledger)."""

import pytest
import torch

from nanoquant.modules.quant_config import NanoQuantConfig
from nanoquant.utils import cache as C


def _cfg(**over):
    cfg = NanoQuantConfig(model_id="tiny/model", num_calib_samples=4, seqlen=16)
    cfg.update(over)
    return cfg


# ---------------------------------------------------------------- hashing primitives
def test_hash_tensors_distinguishes_values_shapes_dtypes_and_none():
    a = torch.arange(6, dtype=torch.float32)
    assert C.hash_tensors(a) == C.hash_tensors(a.clone())
    assert C.hash_tensors(a) != C.hash_tensors(a.reshape(2, 3))
    assert C.hash_tensors(a) != C.hash_tensors(a.to(torch.float64))
    b = a.clone()
    b[3] += 1e-3
    assert C.hash_tensors(a) != C.hash_tensors(b)
    assert C.hash_tensors(a, None) != C.hash_tensors(a)
    assert C.hash_tensors(a.to(torch.bfloat16)) == C.hash_tensors(a.to(torch.bfloat16).clone())
    # non-contiguous views hash like their contiguous copies
    m = torch.randn(4, 6)
    assert C.hash_tensors(m.mT) == C.hash_tensors(m.mT.contiguous())


def test_fingerprint_sources_is_stable_and_group_specific():
    assert C.fingerprint_sources("stats") == C.fingerprint_sources("stats")
    assert C.fingerprint_sources("stats") != C.fingerprint_sources("admm")


# ---------------------------------------------------------------- stage keys
def test_stats_key_ignores_shrinkage_and_downstream_but_tracks_calibration():
    base = _cfg()
    assert C.stats_key(base) == C.stats_key(_cfg(calib_shrinkage=0.9, admm_outer_iters=7, model_kd_lr=1.0))
    for field, value in (("num_calib_samples", 8), ("curvature", "kron"), ("kron_nkp_iters", 5), ("seed", 1),
                         ("calib_dataset", "c4"), ("seqlen", 32)):
        assert C.stats_key(base) != C.stats_key(_cfg(**{field: value})), field


def test_chain_keys_track_block_level_fields_and_are_chained():
    base = _cfg()
    keys = C.chain_keys(base, 3)
    assert len(keys) == 3 and len(set(keys)) == 3
    assert keys == C.chain_keys(base, 3)
    assert keys[:2] == C.chain_keys(base, 2)  # prefix stability
    assert keys == C.chain_keys(_cfg(model_kd_lr=1.0, tune_model=False, ppl_task="x"), 3)  # KD/eval irrelevant
    for field, value in (("calib_shrinkage", 0.9), ("bits", 0.8), ("admm_outer_iters", 7), ("admm_mid_scale", True),
                         ("admm_mid_scale_export", "balanced"), ("fact_mid_scale_lr", 1e-7), ("fact_epochs", 1),
                         ("tune_nonfact", False), ("curvature", "kron")):
        assert keys[0] != C.chain_keys(_cfg(**{field: value}), 3)[0], field
    assert keys == C.chain_keys(_cfg(model_kd_mid_scale_lr=1e-7), 3)  # KD-only knob


def test_kd_and_teacher_keys():
    base = _cfg()
    assert C.kd_key(base, 3) != C.kd_key(_cfg(model_kd_lr=1.0), 3)
    assert C.kd_key(base, 3) != C.kd_key(_cfg(model_kd_mid_scale_lr=1e-7), 3)
    assert C.kd_key(base, 3) != C.kd_key(_cfg(admm_outer_iters=7), 3)  # depends on the chain
    assert C.teacher_key(base) == C.teacher_key(_cfg(admm_outer_iters=7, calib_shrinkage=0.9, curvature="kron"))
    assert C.teacher_key(base) != C.teacher_key(_cfg(num_calib_samples=8))


def test_admm_key_is_content_addressed():
    torch.manual_seed(0)
    W = torch.randn(8, 6)
    i_n, o_n = torch.rand(6), torch.rand(8)
    base = _cfg()
    k = C.admm_key(W, i_n, o_n, None, None, 4, base)
    assert k == C.admm_key(W.clone(), i_n.clone(), o_n.clone(), None, None, 4, _cfg(fact_epochs=1, model_kd_lr=1.0))
    W2 = W.clone()
    W2[0, 0] += 1e-3
    assert k != C.admm_key(W2, i_n, o_n, None, None, 4, base)
    assert k != C.admm_key(W, i_n, o_n, None, None, 8, base)
    assert k != C.admm_key(W, i_n, o_n, torch.eye(6), torch.eye(8), 4, base)
    assert k != C.admm_key(W, i_n, o_n, None, None, 4, _cfg(admm_mid_scale=True))
    assert k != C.admm_key(W, i_n, o_n, None, None, 4, _cfg(admm_mid_scale=True, admm_mid_scale_export="balanced"))
    assert k != C.admm_key(W, i_n, o_n, None, None, 4, _cfg(admm_outer_iters=7))
    # the export is stored in the memo, the tuning learning rates are not
    assert k == C.admm_key(W, i_n, o_n, None, None, 4, _cfg(fact_mid_scale_lr=1e-7, model_kd_mid_scale_lr=1e-7))


# ---------------------------------------------------------------- storage
def test_cache_round_trip_hit_miss_and_atomic_write(tmp_path):
    cache = C.ArtifactCache(tmp_path / "cache")
    assert cache.enabled
    obj = {"a": torch.arange(3), "n": 2, "s": "x", "d": {"t": torch.ones(2, 2, dtype=torch.bfloat16)}}
    assert cache.load("stats", "k1") is None
    p = cache.save("stats", "k1", obj)
    assert p.exists() and not list(p.parent.glob("*.tmp"))
    got = cache.load("stats", "k1")
    assert torch.equal(got["a"], obj["a"]) and got["n"] == 2 and got["s"] == "x"
    assert torch.equal(got["d"]["t"], obj["d"]["t"])

    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"v": torch.tensor([1.0])}

    assert cache.load_or_compute("admm", "k2", compute)["v"].item() == 1.0
    assert cache.load_or_compute("admm", "k2", compute)["v"].item() == 1.0
    assert calls["n"] == 1

    # a stale partial temp file does not break a fresh save
    (p.parent / "k3.pt.tmp").write_bytes(b"garbage")
    cache.save("stats", "k3", {"x": 1})
    assert cache.load("stats", "k3") == {"x": 1}


def test_cache_key_mismatch_raises_and_disabled_cache_is_noop(tmp_path):
    cache = C.ArtifactCache(tmp_path)
    cache.save("stats", "right", {"x": 1})
    (tmp_path / "stats" / "wrong.pt").write_bytes((tmp_path / "stats" / "right.pt").read_bytes())
    with pytest.raises(ValueError):
        cache.load("stats", "wrong")

    off = C.ArtifactCache("")
    assert not off.enabled
    assert off.save("stats", "k", {"x": 1}) is None
    assert off.load("stats", "k") is None
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return 7

    assert off.load_or_compute("stats", "k", compute) == 7
    assert off.load_or_compute("stats", "k", compute) == 7
    assert calls["n"] == 2


def test_ledger_append_and_read(tmp_path):
    C.append_ledger(tmp_path, {"config": {"a": 1}, "results": {"ppl": 12.5}})
    C.append_ledger(tmp_path, {"config": {"a": 2}, "results": {"ppl": 11.0}})
    rows = C.read_ledger(tmp_path)
    assert [r["results"]["ppl"] for r in rows] == [12.5, 11.0]
    assert all("timestamp" in r for r in rows)
    assert C.read_ledger(tmp_path / "nothing") == []


def test_config_defaults_for_cache_fields():
    cfg = NanoQuantConfig()
    assert cfg["cache_dir"] == "cache"
    assert cfg["checkpoint_every_blocks"] == 1
    assert cfg["model_kd_teacher"] == "ram"
