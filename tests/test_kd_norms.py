"""Tests for the KD-stage extensions (trainable normalisation weights, best-epoch selection on validation
perplexity) and for latent row-normalisation before block-level factor tuning."""

import argparse

import pytest
import torch
from torch import nn

from nanoquant.core import compress_block, compress_model, latent, pipeline
from nanoquant.modules.linear import NanoQuantLinear
from nanoquant.modules.quant_config import NanoQuantConfig
from nanoquant.utils import cache as C

RANK = 4


class FakeRMSNorm(nn.Module):
    """Stand-in for ``Qwen3RMSNorm``: a module whose class name ends in ``Norm`` and owns one weight vector."""

    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * self.weight


def _factors(out_f, in_f, rank, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randint(0, 2, (rank, out_f), generator=g).float() * 2 - 1
    B = torch.randint(0, 2, (rank, in_f), generator=g).float() * 2 - 1
    return argparse.Namespace(A=A, B=B, A_latent=A * (torch.rand(A.shape, generator=g) + 0.5) * 1e-3,
                              B_latent=B * (torch.rand(B.shape, generator=g) + 0.5) * 1e-3,
                              scale_pre=torch.rand(1, in_f, generator=g) + 0.5,
                              scale_post=torch.rand(1, out_f, generator=g) + 0.5, W_final=torch.zeros(out_f, in_f))


def _training_layer(in_f, out_f, seed=0):
    """A freshly factorised layer in the STE (``do_train``) state, as :func:`tune_fact` receives it."""
    lin = nn.Linear(in_f, out_f, bias=False).to(torch.bfloat16)
    lin.__class__ = NanoQuantLinear
    lin.__quant_convert__(do_train=True, rank=RANK, factor_results=_factors(out_f, in_f, RANK, seed))
    return lin


def _model():
    m = nn.Module()
    m.input_layernorm = FakeRMSNorm(8)
    m.self_attn = nn.Module()
    m.self_attn.q_proj = _training_layer(8, 6, seed=1)
    m.self_attn.q_proj.finalize(keep_latent=True)
    m.self_attn.q_norm = FakeRMSNorm(6)
    m.mlp = nn.Module()
    m.mlp.down_proj = _training_layer(6, 8, seed=2)
    m.mlp.down_proj.finalize(keep_latent=True)
    m.final_norm = nn.LayerNorm(8)
    m.plain = nn.Linear(3, 3)
    return m


# ---------------------------------------------------------------- KD: normalisation weights
def test_norm_parameters_collects_norm_layers_only():
    m = _model()
    for p in m.parameters():
        p.requires_grad = False
    params = compress_model._norm_parameters(m)
    # two RMSNorm weights + LayerNorm weight and bias; no linear or scale parameters
    assert len(params) == 4
    ids = {id(p) for p in params}
    assert id(m.input_layernorm.weight) in ids and id(m.self_attn.q_norm.weight) in ids
    assert id(m.final_norm.weight) in ids and id(m.final_norm.bias) in ids
    assert id(m.plain.weight) not in ids
    assert all(p.requires_grad for p in params)
    assert not m.plain.weight.requires_grad
    assert compress_model._norm_parameters(nn.Linear(2, 2)) == []


# ---------------------------------------------------------------- KD: best-epoch selection
def test_best_epoch_tracker_restores_lowest_validation_ppl():
    params = [torch.zeros(2), torch.zeros(3)]
    tracker = compress_model.BestEpochTracker()
    assert tracker.epoch is None and not tracker.restore(params)
    params[0].fill_(1.0)
    assert tracker.update(20.0, 1, params)
    params[0].fill_(2.0)
    params[1].fill_(2.0)
    assert tracker.update(15.0, 2, params)  # best so far
    params[0].fill_(3.0)
    assert not tracker.update(16.0, 3, params)  # worse: not recorded
    assert tracker.epoch == 2 and tracker.ppl == 15.0
    assert tracker.restore(params)
    assert torch.equal(params[0], torch.full((2,), 2.0)) and torch.equal(params[1], torch.full((3,), 2.0))
    # round trip through a checkpoint dict
    state = tracker.state_dict()
    fresh = compress_model.BestEpochTracker.from_state_dict(state)
    assert fresh.epoch == 2 and fresh.ppl == 15.0
    params[0].zero_()
    assert fresh.restore(params) and params[0][0].item() == 2.0
    assert compress_model.BestEpochTracker.from_state_dict(None).epoch is None
    # restoring the last epoch is a no-op (nothing to change)
    last = compress_model.BestEpochTracker()
    last.update(10.0, 3, params)
    assert not last.restore(params, last_epoch=3)


# ---------------------------------------------------------------- block tuning: latent normalisation
def test_normalize_latents_on_a_single_training_layer_keeps_ste_forward():
    lin = _training_layer(8, 6)
    x = torch.randn(2, 8).to(torch.bfloat16)
    with torch.no_grad():
        ref = lin(x)
        signs = latent.hard_sign(lin.U_latent).clone()
        assert lin.U_latent.abs().float().mean().item() < 1e-2  # ADMM-scale latents are tiny
    latent.normalize_latents(lin)
    with torch.no_grad():
        rows = lin.U_latent.float().abs().mean(dim=1)
        assert torch.allclose(rows, torch.ones_like(rows), atol=2e-2)
        assert torch.equal(latent.hard_sign(lin.U_latent), signs)
        assert torch.equal(lin(x), ref)


def test_tune_fact_normalizes_latents_when_configured(monkeypatch):
    seen = {}

    def fake_loop(block, optimizer, scheduler, *a, **k):
        seen["mean"] = block.U_latent.detach().float().abs().mean().item()

    monkeypatch.setattr(compress_block, "_tune_loop", fake_loop)
    cfg = NanoQuantConfig(model_id="t", num_calib_samples=2, fact_epochs=1, fact_latent_normalize=True)
    lin = _training_layer(8, 6)
    compress_block.tune_fact(lin, lin, torch.zeros(2, 1, 8), torch.zeros(2, 1, 6),
                             compress_block.BlockCurvature(torch.ones(6), None, False, {}), {}, cfg)
    assert seen["mean"] == pytest.approx(1.0, abs=2e-2)
    assert not lin.has_latent  # hardened afterwards (retain_latent false)
    seen.clear()
    cfg["fact_latent_normalize"] = False
    lin = _training_layer(8, 6)
    compress_block.tune_fact(lin, lin, torch.zeros(2, 1, 8), torch.zeros(2, 1, 6),
                             compress_block.BlockCurvature(torch.ones(6), None, False, {}), {}, cfg)
    assert seen["mean"] < 1e-2


# ---------------------------------------------------------------- config plumbing and cache keys
def test_config_defaults_and_cache_keys():
    cfg = NanoQuantConfig(model_id="t")
    assert cfg["model_kd_norm_weights"] is False and cfg["model_kd_norm_lr"] == 1e-5
    assert cfg["model_kd_select_best"] is False and cfg["fact_latent_normalize"] is False
    base = NanoQuantConfig(model_id="tiny/model", num_calib_samples=4, seqlen=16)

    def over(**kw):
        c = dict(base)
        c.update(kw)
        return c

    for field, value in (("model_kd_norm_weights", True), ("model_kd_norm_lr", 1e-4), ("model_kd_select_best", True)):
        assert C.kd_key(base, 2) != C.kd_key(over(**{field: value}), 2), field
        assert C.chain_keys(base, 2) == C.chain_keys(over(**{field: value}), 2), field
    assert C.chain_keys(base, 2)[0] != C.chain_keys(over(fact_latent_normalize=True), 2)[0]
    pipeline.validate_config(over(model_kd_norm_weights=True, model_kd_select_best=True))
    with pytest.raises(ValueError):
        pipeline.validate_config(over(model_kd_norm_weights=True, model_kd_norm_lr=0.0))
