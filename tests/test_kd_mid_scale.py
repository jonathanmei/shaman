"""Tests for the KD-only unit middle scale (``model_kd_mid_scale``)."""

import argparse

import torch
from torch import nn

from nanoquant.core import compress_model
from nanoquant.core.kd_mid_scale import insert_unit_mid_scales
from nanoquant.core.resume import compressed_state_dict
from nanoquant.modules.linear import NanoQuantLinear
from nanoquant.modules.quant_config import NanoQuantConfig
from nanoquant.utils import cache as C

RANK = 4


class _Block(nn.Module):
    def __init__(self, d=8, hidden=6):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(d, hidden, bias=False)
        self.mlp.down_proj = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.mlp.down_proj(torch.tanh(self.mlp.up_proj(x)))


def _factors(out_f, in_f, rank):
    torch.manual_seed(0)
    A = torch.randint(0, 2, (rank, out_f)).float() * 2 - 1
    B = torch.randint(0, 2, (rank, in_f)).float() * 2 - 1
    return argparse.Namespace(A=A, B=B, A_latent=A.clone(), B_latent=B.clone(), scale_pre=torch.rand(1, in_f) + 0.5,
                              scale_post=torch.rand(1, out_f) + 0.5, W_final=torch.zeros(out_f, in_f))


def _model():
    block = _Block().to(torch.bfloat16)
    lin = block.mlp.up_proj
    lin.__class__ = NanoQuantLinear
    lin.__quant_convert__(do_train=False, rank=RANK, factor_results=_factors(lin.out_features, lin.in_features, RANK))
    return block


def test_insert_unit_mid_scales_is_identity_and_trainable_by_kd():
    torch.manual_seed(2)
    block = _model()
    x = torch.randn(2, 5, 8).to(torch.bfloat16)
    with torch.no_grad():
        before = block(x)
    assert insert_unit_mid_scales(block) == 1
    assert insert_unit_mid_scales(block) == 0  # idempotent
    lin = block.mlp.up_proj
    assert lin.scale_mid.shape == (RANK,) and torch.all(lin.scale_mid == 1) and lin.scale_mid.dtype == torch.bfloat16
    assert lin.scale_mid.optim_group == "scale"
    with torch.no_grad():
        after = block(x)
    assert torch.equal(before, after)
    params = compress_model._kd_parameters(block)
    names = {n for n, p in lin.named_parameters() if any(p is q for q in params)}
    assert names == {"scale_pre", "scale_mid", "scale_post"} and all(p.requires_grad for p in params)
    # KD moving the middle scale changes the output, and the packed state carries it
    with torch.no_grad():
        lin.scale_mid[0] = 2.0
        moved = block(x)
    assert not torch.equal(before, moved)
    assert any("scale_mid" in k for k in compressed_state_dict(block))


def test_model_kd_mid_scale_is_a_kd_key_field():
    base = NanoQuantConfig(model_id="tiny/model", num_calib_samples=4, seqlen=16)
    assert base["model_kd_mid_scale"] is False
    on = dict(base, model_kd_mid_scale=True)
    assert C.kd_key(base, 3) != C.kd_key(on, 3)
    assert C.chain_keys(base, 3) == C.chain_keys(on, 3)  # block stage unaffected
