"""Tests for block checkpoint/restore, the ADMM memo, and the KD teacher-logit modes."""

import argparse
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nanoquant.core import compress_block, resume, teacher
from nanoquant.modules.linear import NanoQuantLinear
from nanoquant.modules.quant_config import NanoQuantConfig
from nanoquant.utils.cache import ArtifactCache, chain_keys, chain_root

RANK = 4


class _Block(nn.Module):
    """Tiny decoder-like block: two linears (one will be factorised) and a norm."""
    def __init__(self, d=8, hidden=6):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(d, hidden, bias=False)
        self.mlp.down_proj = nn.Linear(hidden, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return x + self.mlp.down_proj(torch.tanh(self.mlp.up_proj(self.norm(x))))


def _synthetic_factors(out_f, in_f, rank, mid_scale):
    torch.manual_seed(0)
    A = torch.randint(0, 2, (rank, out_f)).float() * 2 - 1
    B = torch.randint(0, 2, (rank, in_f)).float() * 2 - 1
    f = {
        "A": A, "B": B, "A_latent": A.clone(), "B_latent": B.clone(),
        "scale_pre": torch.rand(1, in_f) + 0.5, "scale_post": torch.rand(1, out_f) + 0.5,
        "W_final": torch.zeros(out_f, in_f),
    }
    if mid_scale:
        f["scale_mid"] = torch.rand(1, rank) + 0.5
    return argparse.Namespace(**f)


def _factorise(block, mid_scale):
    """Convert ``block.mlp.up_proj`` to a binarised NanoQuantLinear (bf16, like the real pipeline)."""
    block = block.to(torch.bfloat16)
    lin = block.mlp.up_proj
    lin.__class__ = NanoQuantLinear
    lin.__quant_convert__(do_train=False, rank=RANK,
                          factor_results=_synthetic_factors(lin.out_features, lin.in_features, RANK, mid_scale))
    return block


@pytest.mark.parametrize("mid_scale", [False, True])
def test_block_checkpoint_round_trip(tmp_path, mid_scale):
    torch.manual_seed(1)
    block = _factorise(_Block(), mid_scale)
    x = torch.randn(2, 5, 8).to(torch.bfloat16)
    with torch.no_grad():
        ref = block(x)

    state = resume.block_state(block)
    assert any(k.endswith("V_packed") for k in state) and all(v.device.type == "cpu" for v in state.values())
    assert ("mlp.up_proj.scale_mid" in state) == mid_scale

    fresh = _Block().to(torch.bfloat16)
    resume.restore_block(fresh, state)
    assert isinstance(fresh.mlp.up_proj, NanoQuantLinear)
    assert hasattr(fresh.mlp.up_proj, "scale_mid") == mid_scale
    assert type(fresh.mlp.down_proj) is nn.Linear
    with torch.no_grad():
        got = fresh(x)
    assert torch.equal(got, ref)
    assert torch.equal(fresh.mlp.up_proj.V, block.mlp.up_proj.V)
    assert torch.equal(fresh.mlp.up_proj.U, block.mlp.up_proj.U)


def _latent_factors(out_f, in_f, rank):
    """Synthetic factors whose latents are continuous (not already ±1), so hardening is observable."""
    f = _synthetic_factors(out_f, in_f, rank, False)
    torch.manual_seed(5)
    f.A_latent = f.A * (torch.rand_like(f.A) + 0.5)
    f.B_latent = f.B * (torch.rand_like(f.B) + 0.5)
    return f


def test_finalize_default_drops_latents_and_keep_latent_retains_them():
    lin = nn.Linear(8, 6, bias=False).to(torch.bfloat16)
    lin.__class__ = NanoQuantLinear
    lin.__quant_convert__(do_train=True, rank=RANK, factor_results=_latent_factors(6, 8, RANK))
    assert lin.has_latent and not hasattr(lin, "U")
    lin.finalize()
    assert not lin.has_latent and hasattr(lin, "U") and not lin.do_train and lin._binarized

    lin2 = nn.Linear(8, 6, bias=False).to(torch.bfloat16)
    lin2.__class__ = NanoQuantLinear
    lin2.__quant_convert__(do_train=True, rank=RANK, factor_results=_latent_factors(6, 8, RANK))
    lin2.finalize(keep_latent=True)
    assert lin2.has_latent and not lin2.do_train and lin2._binarized
    assert not lin2.U_latent.requires_grad and not lin2.V_latent.requires_grad
    assert torch.equal(lin2.U, lin2.binary_ste(lin2.U_latent)) and torch.equal(lin2.V, lin2.binary_ste(lin2.V_latent))
    x = torch.randn(3, 8).to(torch.bfloat16)
    with torch.no_grad():
        y_hard = lin2(x)
        lin2.drop_latent()
        assert not lin2.has_latent
        assert torch.equal(lin2(x), y_hard)


def test_quant_convert_keep_latent_without_training():
    lin = nn.Linear(8, 6, bias=False).to(torch.bfloat16)
    lin.__class__ = NanoQuantLinear
    f = _latent_factors(6, 8, RANK)
    lin.__quant_convert__(do_train=False, rank=RANK, factor_results=f, keep_latent=True)
    assert lin.has_latent and hasattr(lin, "U") and not lin.do_train
    assert torch.equal(lin.U_latent, f.A_latent.mT.to(torch.bfloat16))


def test_state_dict_keeps_latents_and_packs_only_hardened_factors():
    lin = nn.Linear(8, 6, bias=False).to(torch.bfloat16)
    lin.__class__ = NanoQuantLinear
    lin.__quant_convert__(do_train=True, rank=RANK, factor_results=_latent_factors(6, 8, RANK))
    lin.finalize(keep_latent=True)
    state = lin.state_dict(prefix="l.")
    assert {"l.V_packed", "l.U_packed", "l.V_shape", "l.U_shape", "l.V_latent", "l.U_latent"} <= set(state)
    assert "l.V" not in state and "l.U" not in state
    lin.drop_latent()
    state = lin.state_dict(prefix="l.")
    assert "l.V_latent" not in state and "l.U_latent" not in state and "l.V_packed" in state


def test_block_checkpoint_round_trip_retains_latents_for_kd():
    torch.manual_seed(1)
    block = _Block().to(torch.bfloat16)
    lin = block.mlp.up_proj
    lin.__class__ = NanoQuantLinear
    lin.__quant_convert__(do_train=True, rank=RANK, factor_results=_latent_factors(lin.out_features, lin.in_features, RANK))
    lin.finalize(keep_latent=True)
    state = resume.block_state(block)
    x = torch.randn(2, 5, 8).to(torch.bfloat16)
    with torch.no_grad():
        ref = block(x)

    fresh = _Block().to(torch.bfloat16)
    resume.restore_block(fresh, state)
    got_lin = fresh.mlp.up_proj
    assert got_lin.has_latent
    assert torch.equal(got_lin.U_latent, lin.U_latent) and torch.equal(got_lin.V_latent, lin.V_latent)
    assert torch.equal(got_lin.U, lin.U) and torch.equal(got_lin.V, lin.V)
    with torch.no_grad():
        assert torch.equal(fresh(x), ref)

    # a checkpoint written without latents restores a plain hardened layer
    lin.drop_latent()
    fresh2 = _Block().to(torch.bfloat16)
    resume.restore_block(fresh2, resume.block_state(block))
    assert not fresh2.mlp.up_proj.has_latent


def test_restore_prefix_uses_progress_and_chain(tmp_path):
    cfg = NanoQuantConfig(model_id="tiny", num_calib_samples=2, seqlen=8)
    cache = ArtifactCache(tmp_path)
    keys = chain_keys(cfg, 3)
    root = chain_root(cfg)
    blocks = [_factorise(_Block(), False) for _ in range(3)]

    # nothing saved -> start from scratch
    assert resume.restore_prefix(cache, root, keys, [_Block() for _ in range(3)]) == (0, None, None)

    # blocks 0 and 1 done, progress after block 1
    ci, oi = torch.randn(2, 8, 8), torch.randn(2, 8, 8)
    resume.save_block_checkpoint(cache, keys[0], blocks[0])
    resume.save_block_checkpoint(cache, keys[1], blocks[1])
    resume.save_progress(cache, root, 1, ci, oi)
    fresh = [_Block() for _ in range(3)]
    k, ci2, oi2 = resume.restore_prefix(cache, root, keys, fresh)
    assert k == 2 and torch.equal(ci2, ci) and torch.equal(oi2, oi)
    assert isinstance(fresh[0].mlp.up_proj, NanoQuantLinear) and isinstance(fresh[1].mlp.up_proj, NanoQuantLinear)
    assert type(fresh[2].mlp.up_proj) is nn.Linear

    # a checkpoint for block 2 without matching progress is not resumable beyond the progress record
    resume.save_block_checkpoint(cache, keys[2], blocks[2])
    k, _, _ = resume.restore_prefix(cache, root, keys, [_Block() for _ in range(3)])
    assert k == 2

    # a gap in the chain (block 1 missing) truncates the prefix to block 0 only
    cache.path("block", keys[1]).unlink()
    resume.save_progress(cache, root, 0, ci, oi)
    k, _, _ = resume.restore_prefix(cache, root, keys, [_Block() for _ in range(3)])
    assert k == 1

    # a different chain (other HPs) sees nothing
    other = chain_keys(NanoQuantConfig(model_id="tiny", num_calib_samples=2, seqlen=8, admm_outer_iters=3), 3)
    assert resume.completed_prefix(cache, other) == 0


def test_factorize_and_replace_memoises_admm(tmp_path, monkeypatch):
    torch.manual_seed(2)
    cfg = NanoQuantConfig(model_id="tiny")
    cfg.update({"tune_fact": False, "admm_outer_iters": 3})
    cache = ArtifactCache(tmp_path)
    calls = {"n": 0}
    real = compress_block.factorize_admm_nanoquant

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(compress_block, "factorize_admm_nanoquant", counting)

    def make_block():
        torch.manual_seed(3)
        block = _Block(d=16, hidden=12)
        for lin in (block.mlp.up_proj, block.mlp.down_proj):
            lin.register_buffer("i_norm", torch.rand(lin.in_features) + 0.5, persistent=False)
            lin.register_buffer("o_norm", torch.rand(lin.out_features) + 0.5, persistent=False)
        return block

    b1 = make_block()
    compress_block.factorize_and_replace(b1, "mlp.up_proj", RANK, cfg, cache=cache)
    assert calls["n"] == 1
    b2 = make_block()
    compress_block.factorize_and_replace(b2, "mlp.up_proj", RANK, cfg, cache=cache)
    assert calls["n"] == 1  # identical inputs -> memo hit
    assert torch.equal(b1.mlp.up_proj.V, b2.mlp.up_proj.V)
    assert torch.equal(b1.mlp.up_proj.scale_pre, b2.mlp.up_proj.scale_pre)

    b3 = make_block()
    with torch.no_grad():
        b3.mlp.up_proj.weight[0, 0] += 1.0
    compress_block.factorize_and_replace(b3, "mlp.up_proj", RANK, cfg, cache=cache)
    assert calls["n"] == 2  # changed input -> miss

    compress_block.factorize_and_replace(make_block(), "mlp.up_proj", RANK, cfg, cache=None)
    assert calls["n"] == 3  # cache disabled -> always compute


def test_factorize_and_replace_threads_admm_reg(monkeypatch):
    cfg = NanoQuantConfig(model_id="tiny", admm_reg=0.123, tune_fact=False)
    block = _Block(d=8, hidden=6)
    lin = block.mlp.up_proj
    lin.register_buffer("i_norm", torch.ones(lin.in_features), persistent=False)
    lin.register_buffer("o_norm", torch.ones(lin.out_features), persistent=False)
    seen = {}

    def fake_factorize(W, i_norm, o_norm, mid_rank, **kwargs):
        seen["reg"] = kwargs["reg"]
        return vars(_synthetic_factors(W.shape[0], W.shape[1], mid_rank, mid_scale=False))

    monkeypatch.setattr(compress_block, "factorize_admm_nanoquant", fake_factorize)
    compress_block.factorize_and_replace(block, "mlp.up_proj", RANK, cfg, cache=None)

    assert seen["reg"] == pytest.approx(0.123)


class _TinyLM(nn.Module):
    def __init__(self, vocab=11, d=6):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        return SimpleNamespace(logits=self.head(torch.tanh(self.emb(ids))))


@pytest.mark.parametrize("mode", ["ram", "disk", "online"])
def test_teacher_logits_modes_agree(tmp_path, mode):
    torch.manual_seed(4)
    model = _TinyLM()
    samples = [torch.randint(0, 11, (1, 7)) for _ in range(3)]
    with torch.no_grad():
        ref = [model(s).logits for s in samples]
    cache = ArtifactCache(tmp_path)
    t = teacher.TeacherLogits(mode, model, samples, "cpu", cache=cache, key="tk")
    for idx, s in enumerate(samples):
        got = t.get(idx, s).float()
        tol = 2e-2 if mode == "disk" else 1e-6  # disk stores bf16
        assert torch.allclose(got, ref[idx], atol=tol, rtol=tol), (mode, idx)
    if mode == "disk":
        assert cache.path("teacher", "tk", ".bin").exists() and cache.path("teacher", "tk", ".json").exists()
        # second instance reuses the memmap without recomputation
        t2 = teacher.TeacherLogits("disk", _TinyLM(), samples, "cpu", cache=cache, key="tk")  # different weights!
        assert torch.allclose(t2.get(1, samples[1]).float(), ref[1], atol=2e-2, rtol=2e-2)


def test_teacher_rejects_unknown_mode_and_disk_without_cache():
    with pytest.raises(ValueError):
        teacher.TeacherLogits("banana", _TinyLM(), [], "cpu")
    with pytest.raises(ValueError):
        teacher.TeacherLogits("disk", _TinyLM(), [], "cpu", cache=ArtifactCache(""), key="k")
