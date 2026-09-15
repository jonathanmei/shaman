"""Tests for the MLP-only tuning forward (frozen attention half skipped) and the tuning loop's ``forward_fn`` hook."""

import copy

import torch
from torch import nn

from nanoquant.core import compress_block as cb
from nanoquant.optimi import AdamW

D, INTER, SEQ, N = 8, 16, 5, 6


class _FrozenAttn(nn.Module):
    """Stand-in for a block whose attention linears were all converted (no ``nn.Linear`` left, nothing trainable)."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(D, D) * 0.1, requires_grad=False)

    def forward(self, x):
        return torch.tanh(x @ self.w)


class _Block(nn.Module):
    """HF decoder-layer structure: h = x + attn(ln1(x)); out = h + mlp(ln2(h))."""

    def __init__(self, attn=None):
        super().__init__()
        self.self_attn = attn if attn is not None else _FrozenAttn()
        self.input_layernorm = nn.LayerNorm(D)
        self.post_attention_layernorm = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, INTER), nn.GELU(), nn.Linear(INTER, D))

    def forward(self, x, **kwargs):
        h = x + self.self_attn(self.input_layernorm(x))
        return (h + self.mlp(self.post_attention_layernorm(h)),)


def _data(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(N, SEQ, D, generator=g), torch.randn(N, SEQ, D, generator=g)


def test_mlp_only_forward_matches_full_block():
    torch.manual_seed(1)
    block = _Block()
    x, _ = _data()
    fn = cb.mlp_only_forward(block, x, {}, "mlp.2")
    assert fn is not None
    for j in range(N):
        assert torch.allclose(fn(j), block(x[j:j + 1])[0], atol=1e-6)


def test_mlp_only_forward_falls_back_when_not_applicable():
    torch.manual_seed(2)
    x, _ = _data()
    assert cb.mlp_only_forward(_Block(), x, {}, "self_attn.q_proj") is None  # attention layer
    assert cb.mlp_only_forward(_Block(), x, {}, None) is None
    attn_linear = nn.Sequential(nn.Linear(D, D))  # a full-precision linear still in the attention: not frozen
    assert cb.mlp_only_forward(_Block(attn=attn_linear), x, {}, "mlp.0") is None
    trainable = _FrozenAttn()
    trainable.w.requires_grad_(True)
    assert cb.mlp_only_forward(_Block(attn=trainable), x, {}, "mlp.0") is None
    plain = nn.Sequential(nn.Linear(D, D))  # no HF attribute names (e.g. OPT): full forward
    assert cb.mlp_only_forward(plain, x, {}, "mlp.0") is None


def _tune(block, x, y, forward_fn):
    params = [p for p in block.mlp.parameters()]
    for p in params:
        p.requires_grad_(True)
    opt = AdamW(params, lr=1e-3, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2 * N, eta_min=1e-7)
    torch.manual_seed(0)
    with torch.enable_grad():
        cb._tune_loop(block, opt, sched, x, y, torch.ones(D), {}, batch_size=2, epochs=2, num_samples=N,
                      forward_fn=forward_fn)
    return [p.detach().clone() for p in params]


def test_tune_loop_with_mlp_forward_matches_full_forward():
    torch.manual_seed(3)
    x, y = _data(5)
    block_full = _Block()
    block_fast = copy.deepcopy(block_full)
    ref = _tune(block_full, x, y, None)
    fn = cb.mlp_only_forward(block_fast, x, {}, "mlp.0")
    got = _tune(block_fast, x, y, fn)
    for a, b in zip(ref, got):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)
