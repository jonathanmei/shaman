"""Tests for the MLP-only tuning forward (frozen attention half skipped) and the tuning loop's ``forward_fn`` hook."""

import copy
import math

import pytest
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


NAMES = ['self_attn.q_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'self_attn.k_proj', 'mlp.gate_proj',
         'mlp.up_proj', 'mlp.down_proj']


def _eval_loss(block, x, y):
    with torch.no_grad():
        return sum(cb.fused_weighted_mse(block(x[j:j + 1])[0], y[j:j + 1], torch.ones(D)).item() for j in range(N))


def _run_keep_best(lr, keep_best, epochs=4):
    torch.manual_seed(3)
    block = _Block()
    x, y = _data(1)
    params = list(block.mlp.parameters())
    for p in params:
        p.requires_grad_(True)
    init = [p.detach().clone() for p in params]
    opt = torch.optim.SGD(params, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    torch.manual_seed(0)
    with torch.enable_grad():
        run = cb._tune_loop(block, opt, sched, x, y, torch.ones(D), {}, batch_size=1, epochs=epochs, num_samples=N,
                            keep_best=keep_best)
    return block, x, y, params, init, run


def test_tune_loop_keep_best_restores_the_best_state():
    """With a diverging learning rate the pre-tuning state wins and is restored; a sane run keeps its last epoch."""
    block, x, y, params, init, run = _run_keep_best(lr=50.0, keep_best=True)
    assert run == 4
    for p, p0 in zip(params, init):
        assert torch.equal(p.detach(), p0)  # the initial (ADMM) state was the best candidate
    block_plain, x, y, params_plain, init_plain, _ = _run_keep_best(lr=50.0, keep_best=False)
    assert not all(torch.equal(p.detach(), p0) for p, p0 in zip(params_plain, init_plain))
    plain_loss = _eval_loss(block_plain, x, y)
    assert math.isnan(plain_loss) or _eval_loss(block, x, y) < plain_loss  # the diverged run may be NaN
    # a well-behaved run: keep_best selects the last epoch and changes nothing
    block_a, x, y, pa, _, _ = _run_keep_best(lr=1e-2, keep_best=True)
    block_b, _, _, pb, _, _ = _run_keep_best(lr=1e-2, keep_best=False)
    for p, q in zip(pa, pb):
        assert torch.allclose(p.detach(), q.detach())


def test_scaled_epochs():
    assert cb.scaled_epochs(8, 1.0) == 8 and cb.scaled_epochs(8, 0.25) == 2 and cb.scaled_epochs(8, 0.05) == 1
    assert cb.scaled_epochs(8, 0.75) == 6


def test_tuning_epoch_weights_modes():
    assert cb.tuning_epoch_weights(None, None, 0, NAMES, {}) == {n: 1.0 for n in NAMES}
    typed = cb.tuning_epoch_weights(None, None, 0, NAMES, {"tune_epoch_weights": "type"})
    assert typed["self_attn.q_proj"] == 0.25 and typed["mlp.down_proj"] == 1.0 and typed["mlp.gate_proj"] == 0.75
    # floor applies to the type table too
    assert cb.tuning_epoch_weights(None, None, 0, NAMES, {"tune_epoch_weights": "type", "tune_epoch_min_frac": 0.5})[
        "self_attn.k_proj"] == 0.5
    assert cb.tuning_epoch_weights(None, None, 0, ["fc1", "fc2"], {"tune_epoch_weights": "type"}) == {"fc1": 1.0,
                                                                                                        "fc2": 1.0}
    # measured: predicted loss exp(a) r^-beta at the allocated rank, normalised to the block's largest, floored
    curves = {f"1.{n}": (a, 1.0) for n, a in zip(NAMES, [0.0, 0.0, 1.0, -2.0, 2.0, 2.5, 3.0])}
    ranks = {f"1.{n}": 100 for n in NAMES}
    w = cb.tuning_epoch_weights({"curves": curves}, ranks, 1, NAMES, {"tune_epoch_weights": "measured",
                                                                       "tune_epoch_min_frac": 0.1})
    assert w["mlp.down_proj"] == 1.0
    assert abs(w["mlp.up_proj"] - math.exp(-0.5)) < 1e-9 and abs(w["mlp.gate_proj"] - math.exp(-1.0)) < 1e-9
    assert w["self_attn.k_proj"] == 0.1  # exp(-5) floored
    # a larger rank lowers the predicted loss and hence the weight
    ranks2 = dict(ranks, **{"1.mlp.down_proj": 400})
    w2 = cb.tuning_epoch_weights({"curves": curves}, ranks2, 1, NAMES, {"tune_epoch_weights": "measured"})
    assert w2["mlp.down_proj"] < 1.0 and w2["mlp.up_proj"] == 1.0
    # missing curves -> type table fallback; unknown mode -> error
    fb = cb.tuning_epoch_weights({"curves": {}}, ranks, 1, NAMES, {"tune_epoch_weights": "measured"})
    assert fb == cb.tuning_epoch_weights(None, None, 1, NAMES, {"tune_epoch_weights": "type"})
    with pytest.raises(ValueError):
        cb.tuning_epoch_weights(None, None, 0, NAMES, {"tune_epoch_weights": "banana"})


def test_nonfact_rounds_per_group():
    groups = {"self_attn.q_proj": "self_attn.q_proj", "self_attn.v_proj": "self_attn.q_proj",
              "self_attn.k_proj": "self_attn.q_proj", "self_attn.o_proj": "self_attn.o_proj",
              "mlp.gate_proj": "mlp.gate_proj", "mlp.up_proj": "mlp.gate_proj", "mlp.down_proj": "mlp.down_proj"}
    assert list(cb.nonfact_rounds(NAMES, groups, True).values()) == [True, False, True, False, True, False, True]
    assert all(cb.nonfact_rounds(NAMES, groups, False).values())
    assert all(cb.nonfact_rounds(NAMES, {}, True).values())


def test_tune_loop_plateau_stop():
    torch.manual_seed(4)
    x, y = _data(6)
    calls = {"n": 0}

    def _run(tol):
        block = _Block()
        params = list(block.mlp.parameters())
        opt = AdamW(params, lr=1e-9, weight_decay=0)  # tiny lr: the loss barely moves -> plateau at epoch 2
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5 * N, eta_min=1e-12)
        calls["n"] = 0

        def fn(idx):
            calls["n"] += 1
            return block(x[idx:idx + 1])[0]

        with torch.enable_grad():
            return cb._tune_loop(block, opt, sched, x, y, torch.ones(D), {}, batch_size=2, epochs=5, num_samples=N,
                                 forward_fn=fn, plateau_tol=tol)

    assert _run(0.0) == 5 and calls["n"] == 5 * N
    assert _run(0.5) == 2 and calls["n"] == 2 * N


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
