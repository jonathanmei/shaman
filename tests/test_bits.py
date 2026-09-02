"""Tests for the bits-per-weight accounting."""

import argparse
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nanoquant.modules.linear import NanoQuantLinear
from nanoquant.utils import bits as B

QWEN3_0P6B = dict(hidden=1024, q=2048, kv=1024, inter=3072)
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


def _fake_model(n_blocks=1):
    return SimpleNamespace(config=SimpleNamespace(model_type="qwen3"),
                           model=SimpleNamespace(layers=[_block(QWEN3_0P6B) for _ in range(n_blocks)]))


def test_layer_bits_formula():
    assert B.layer_bits(1024, 1024, 480, 2) == 480 * 2048 + 16 * 2048
    assert B.layer_bits(1024, 1024, 480, 3) == 480 * 2048 + 16 * (2048 + 480)


@pytest.mark.parametrize("mid_scale, expected_bpw", [(False, 0.9729), (True, 0.9774)])
def test_static_accounting_qwen3_0p6b(mid_scale, expected_bpw):
    cfg = {"bits": 1.0, "admm_type": "nanoquant", "admm_mid_scale": mid_scale}
    acc = B.static_accounting(_fake_model(2), NAMES, cfg)
    assert acc["num_scales"] == (3 if mid_scale else 2)
    # rank rounding to multiples of 32 makes the ranks identical for 2 and 3 scales at 1.0 bpw
    ranks = {k.split(".", 1)[1]: v["rank"] for k, v in acc["layers"].items() if k.startswith("0.")}
    assert ranks == {"self_attn.q_proj": 640, "self_attn.k_proj": 480, "self_attn.v_proj": 480,
                     "self_attn.o_proj": 640, "mlp.gate_proj": 736, "mlp.up_proj": 736, "mlp.down_proj": 736}
    assert acc["factorized_bpw"] == pytest.approx(expected_bpw, abs=5e-4)
    assert acc["factorized_weights"] == 2 * 15_728_640
    assert "bpw" in B.format_accounting(acc)


def _quantised_linear(in_f, out_f, rank, mid_scale):
    lin = nn.Linear(in_f, out_f, bias=False)
    lin.__class__ = NanoQuantLinear
    A = torch.ones(rank, out_f)
    Bm = torch.ones(rank, in_f)
    f = {"A": A, "B": Bm, "A_latent": A, "B_latent": Bm, "scale_pre": torch.ones(1, in_f),
         "scale_post": torch.ones(1, out_f), "W_final": torch.zeros(out_f, in_f)}
    if mid_scale:
        f["scale_mid"] = torch.ones(1, rank)
    lin.__quant_convert__(do_train=False, rank=rank, factor_results=argparse.Namespace(**f))
    return lin


@pytest.mark.parametrize("mid_scale", [False, True])
def test_model_accounting_counts_binaries_scales_and_other_params(mid_scale):
    model = nn.Module()
    model.q = _quantised_linear(16, 8, 4, mid_scale)
    model.emb = nn.Embedding(10, 16)  # 160 params in fp32 -> 5120 bits
    model.norm = nn.LayerNorm(16)  # 32 params -> 1024 bits
    acc = B.model_accounting(model)
    binary = 4 * 16 + 4 * 8
    scales = 16 + 8 + (4 if mid_scale else 0)
    assert acc["layers"]["q"]["binary_bits"] == binary
    assert acc["layers"]["q"]["scale_bits"] == 16 * scales
    assert acc["factorized_bits"] == binary + 16 * scales
    assert acc["factorized_weights"] == 16 * 8
    assert acc["factorized_bpw"] == pytest.approx((binary + 16 * scales) / 128)
    assert acc["other_params"] == 160 + 32
    assert acc["other_bits"] == (160 + 32) * 32
    assert acc["model_bpw"] == pytest.approx((binary + 16 * scales + 192 * 32) / (128 + 192))
    text = B.format_accounting(acc, title="t")
    assert "whole model" in text and "rank=4" in text


def test_model_accounting_counts_tied_weights_once():
    model = nn.Module()
    model.emb = nn.Embedding(10, 4)
    model.head = nn.Linear(4, 10, bias=False)
    model.head.weight = model.emb.weight
    acc = B.model_accounting(model)
    assert acc["other_params"] == 40
