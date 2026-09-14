"""Shared fresh input factors: group detection, probe validation and the eigendecomposition cache."""

import torch
from torch import nn

from nanoquant.core import admm_nq
from nanoquant.core import compress_block as cb
from nanoquant.core.importance import shrink_toward_identity
from nanoquant.utils.utils import find_layers


class _MLPBlock(nn.Module):
    def __init__(self, d=8, hidden=6):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(d, hidden, bias=False)
        self.mlp.down_proj = nn.Linear(hidden, d, bias=False)

    def forward(self, x, **kwargs):
        return (x + self.mlp.down_proj(torch.tanh(self.mlp.up_proj(x))),)


class _Scale(nn.Module):
    """RMSNorm-like affine scale: the module whose weight `tune_nonfact` never trains."""

    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * self.weight


class _TwoNormBlock(nn.Module):
    """q, k share the first norm's output; up is fed by the second norm; o and down are unique."""

    def __init__(self, d=8, h=6):
        super().__init__()
        self.norm1 = _Scale(d)
        self.norm2 = _Scale(d)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(d, h, bias=False)
        self.self_attn.k_proj = nn.Linear(d, h, bias=False)
        self.self_attn.o_proj = nn.Linear(h, d, bias=False)
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(d, h, bias=False)
        self.mlp.down_proj = nn.Linear(h, d, bias=False)

    def forward(self, x, **kwargs):
        a = self.norm1(x)
        h = x + self.self_attn.o_proj(torch.tanh(self.self_attn.q_proj(a) + self.self_attn.k_proj(a)))
        b = self.norm2(h)
        return (h + self.mlp.down_proj(torch.tanh(self.mlp.up_proj(b))),)


_NAMES = ["self_attn.q_proj", "self_attn.o_proj", "self_attn.k_proj", "mlp.up_proj", "mlp.down_proj"]


def test_shared_input_groups_detects_shared_and_unique_inputs():
    torch.manual_seed(20)
    block = _TwoNormBlock()
    layers = find_layers(block)
    groups = cb.shared_input_groups(block, layers, _NAMES, torch.randn(1, 5, 8), {})
    assert groups["self_attn.q_proj"] == groups["self_attn.k_proj"] == "self_attn.q_proj"
    assert groups["self_attn.o_proj"] == "self_attn.o_proj"
    assert groups["mlp.up_proj"] == "mlp.up_proj"
    assert groups["mlp.down_proj"] == "mlp.down_proj"
    assert cb.group_size(groups, "self_attn.q_proj") == 2 and cb.group_size(groups, "mlp.up_proj") == 1


def test_input_second_moment_returns_probe():
    torch.manual_seed(21)
    block = _MLPBlock()
    x = torch.randn(3, 5, 8)
    R_only = cb.input_second_moment(block, block.mlp.down_proj, x, {}, num_samples=3)
    R, probe = cb.input_second_moment(block, block.mlp.down_proj, x, {}, num_samples=3, return_probe=True)
    with torch.no_grad():
        expected = torch.tanh(block.mlp.up_proj(x[:1])).flatten(0, -2)
    assert torch.allclose(R, R_only) and torch.allclose(probe, expected, atol=1e-6)
    assert torch.allclose(cb.layer_input_probe(block, block.mlp.down_proj, x[:1], {}), expected, atol=1e-6)


def test_fresh_input_factor_reuses_within_group_and_revalidates(monkeypatch):
    torch.manual_seed(22)
    block = _TwoNormBlock()
    layers = find_layers(block)
    x = torch.randn(3, 5, 8)
    groups = cb.shared_input_groups(block, layers, _NAMES, x[:1], {})
    calls = {"n": 0}
    real = cb.input_second_moment

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(cb, "input_second_moment", counting)
    fresh_cache, eig_cache = {}, admm_nq.EigCache()

    def fetch(name):
        return cb.fresh_input_factor(block, layers[name], name, groups, x, {}, 3, 0.2, fresh_cache, eig_cache)

    R_q, Rs_q, reused = fetch("self_attn.q_proj")
    assert not reused and calls["n"] == 1 and "self_attn.q_proj" in fresh_cache
    assert torch.allclose(Rs_q, shrink_toward_identity(R_q, 0.2))
    # the shared factor is registered, so ADMM's eigendecomposition of it is cached
    admm_nq._normalized_curvature(Rs_q, Rs_q.diagonal(), torch.float64, 1e-12, eig_cache=eig_cache)
    assert len(eig_cache) == 1
    R_k, Rs_k, reused = fetch("self_attn.k_proj")
    assert reused and calls["n"] == 1 and R_k is R_q and Rs_k is Rs_q
    assert len(eig_cache) == 1  # a valid reuse keeps the eigendecomposition cache
    # a unique input is never cached
    _, _, reused = fetch("self_attn.o_proj")
    assert not reused and calls["n"] == 2 and "self_attn.o_proj" not in fresh_cache
    # the shared input changes upstream -> the probe check fails -> recompute and evict the eigendecompositions
    with torch.no_grad():
        block.norm1.weight.mul_(2.0)
    R_k2, _, reused = fetch("self_attn.k_proj")
    assert not reused and calls["n"] == 3 and len(eig_cache) == 0
    assert torch.allclose(R_k2, 4.0 * R_q, atol=1e-5)
