"""Tests for latent retention across the pipeline: flip/margin diagnostics, normalisation, KD parameter selection,
and the up-front configuration validation."""

import argparse

import pytest
import torch
from torch import nn

from nanoquant.core import compress_model, latent, pipeline
from nanoquant.modules.linear import NanoQuantLinear
from nanoquant.modules.quant_config import NanoQuantConfig

RANK = 4


def _factors(out_f, in_f, rank, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randint(0, 2, (rank, out_f), generator=g).float() * 2 - 1
    B = torch.randint(0, 2, (rank, in_f), generator=g).float() * 2 - 1
    return argparse.Namespace(A=A, B=B, A_latent=A * (torch.rand(A.shape, generator=g) + 0.5),
                              B_latent=B * (torch.rand(B.shape, generator=g) + 0.5),
                              scale_pre=torch.rand(1, in_f, generator=g) + 0.5,
                              scale_post=torch.rand(1, out_f, generator=g) + 0.5, W_final=torch.zeros(out_f, in_f))


def _layer(in_f, out_f, keep_latent=True, seed=0):
    lin = nn.Linear(in_f, out_f, bias=False).to(torch.bfloat16)
    lin.__class__ = NanoQuantLinear
    lin.__quant_convert__(do_train=True, rank=RANK, factor_results=_factors(out_f, in_f, RANK, seed))
    lin.finalize(keep_latent=keep_latent)
    return lin


def _model():
    m = nn.Module()
    m.self_attn = nn.Module()
    m.mlp = nn.Module()
    m.self_attn.q_proj = _layer(8, 6, seed=1)
    m.mlp.down_proj = _layer(6, 8, seed=2)
    m.plain = nn.Linear(3, 3)
    return m


def test_flip_stats_zero_then_counts_flips_by_type():
    m = _model()
    s = latent.latent_flip_stats(m)
    assert s["flipped"] == 0 and s["layers"] == 2 and s["zero_flip_layers"] == 2
    assert set(s["by_type"]) == {"q_proj", "down_proj"}
    assert s["total"] == sum(p.numel() for _, mod in latent.iter_latent_layers(m) for p in (mod.U_latent, mod.V_latent))
    with torch.no_grad():
        m.mlp.down_proj.U_latent[0, 0] *= -1
        m.mlp.down_proj.V_latent[1, 2] *= -1
    s = latent.latent_flip_stats(m)
    assert s["flipped"] == 2 and s["by_type"]["down_proj"]["flipped"] == 2 and s["by_type"]["q_proj"]["flipped"] == 0
    assert s["zero_flip_layers"] == 1
    assert "down_proj" in latent.format_flip_stats(s)


def test_margin_stats_thresholds():
    m = _model()
    with torch.no_grad():
        m.self_attn.q_proj.U_latent.zero_()  # every entry below every threshold
    s = latent.latent_margin_stats(m, thresholds=(1e-3, 1.0))
    q = s["by_type"]["q_proj"]
    n_u = m.self_attn.q_proj.U_latent.numel()
    n_v = m.self_attn.q_proj.V_latent.numel()
    assert q["below"][1e-3] == pytest.approx(n_u / (n_u + n_v))
    assert s["by_type"]["down_proj"]["below"][1e-3] == 0.0  # magnitudes are in [0.5, 1.5)
    assert s["by_type"]["down_proj"]["below"][1.0] > 0.0
    assert "median" in latent.format_margin_stats(s)


def test_normalize_latents_preserves_signs_and_forward():
    m = _model()
    x = torch.randn(2, 8).to(torch.bfloat16)
    with torch.no_grad():
        ref = m.self_attn.q_proj(x)
        signs_before = latent.hard_sign(m.self_attn.q_proj.U_latent).clone()
    latent.normalize_latents(m)
    with torch.no_grad():
        assert torch.equal(latent.hard_sign(m.self_attn.q_proj.U_latent), signs_before)
        rows = m.self_attn.q_proj.U_latent.float().abs().mean(dim=1)
        assert torch.allclose(rows, torch.ones_like(rows), atol=2e-2)
        assert torch.equal(m.self_attn.q_proj(x), ref)
    assert latent.latent_flip_stats(m)["flipped"] == 0


def test_drop_latents():
    m = _model()
    latent.drop_latents(m)
    assert not any(True for _ in latent.iter_latent_layers(m))
    assert hasattr(m.mlp.down_proj, "U")


def test_kd_parameters_modes():
    m = _model()
    scales, latents = compress_model._kd_parameters(m, "scales")
    assert len(scales) == 4 and latents == []
    assert all(mod._binarized and not mod.do_train for _, mod in latent.iter_latent_layers(m))
    scales, latents = compress_model._kd_parameters(m, "scales_latent")
    assert len(scales) == 4 and len(latents) == 4
    assert all(not mod._binarized and mod.do_train for _, mod in latent.iter_latent_layers(m))
    # STE forward on the latents equals the hardened forward at the start of KD
    x = torch.randn(2, 8).to(torch.bfloat16)
    with torch.no_grad():
        ste = m.self_attn.q_proj(x)
        m.self_attn.q_proj.do_train = False
        m.self_attn.q_proj._binarized = True
        assert torch.equal(ste, m.self_attn.q_proj(x))
    # finishing KD hardens and drops the latents; a flipped latent changes the deployed sign
    m2 = _model()
    compress_model._kd_parameters(m2, "scales_latent")
    with torch.no_grad():
        m2.mlp.down_proj.U_latent[0, 0] = -m2.mlp.down_proj.U_latent[0, 0]
    old = m2.mlp.down_proj.U[0, 0].item()
    compress_model._finish_kd(m2, "scales_latent")
    assert not m2.mlp.down_proj.has_latent and m2.mlp.down_proj.U[0, 0].item() == -old
    m3 = _model()
    compress_model._kd_parameters(m3, "scales_latent")
    latent.drop_latents(m3)
    with pytest.raises(ValueError):
        compress_model._kd_parameters(m3, "scales_latent")


def test_validate_config():
    ok = NanoQuantConfig(model_id="tiny", retain_latent=True, model_kd_mode="scales_latent", curvature="kron",
                         block_loss="mahalanobis", block_loss_source="plain")
    pipeline.validate_config(ok)
    pipeline.validate_config(NanoQuantConfig(model_id="tiny", model_kd_mode="scales_latent", tune_model=False))
    for bad in ({"model_kd_mode": "scales_latent"},
                {"block_loss": "mahalanobis"},
                {"block_loss_source": "plain"},
                {"block_loss": "banana"}, {"model_kd_mode": "banana"}, {"curvature": "banana"},
                {"block_loss_source": "banana"}, {"max_blocks": -1}, {"admm_input_factor": "banana"}):
        with pytest.raises(ValueError):
            pipeline.validate_config(NanoQuantConfig(model_id="tiny", **bad))


def test_pipeline_validates_before_loading_anything(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("load_model must not be called for an invalid config")

    monkeypatch.setattr(pipeline, "load_model", boom)
    with pytest.raises(ValueError):
        pipeline.run_quantization_pipeline("tiny", NanoQuantConfig(model_id="tiny", model_kd_mode="scales_latent"))
