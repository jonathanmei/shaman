"""Tests for the logit-level (suffix-aware) reconstruction objective of the last decoder blocks."""

import pytest
import torch
from torch import nn

from nanoquant.core import compress_block, compress_model, pipeline, tail
from nanoquant.modules.quant_config import NanoQuantConfig
from nanoquant.utils import cache as C
from nanoquant.utils import utils as U

HIDDEN, VOCAB, SEQ, N = 8, 11, 5, 3


class Block(nn.Module):
    """Decoder-block stand-in: returns a tuple like HF decoder layers and accepts arbitrary kwargs."""

    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, h, **kwargs):
        return (h + torch.tanh(self.lin(h)),)


def _suffix(seed=0):
    torch.manual_seed(seed)
    blocks = [Block(seed + 1), Block(seed + 2)]
    norm = nn.LayerNorm(HIDDEN)
    head = nn.Linear(HIDDEN, VOCAB, bias=False)
    return blocks, norm, head


def _targets():
    torch.manual_seed(7)
    return torch.randn(N, SEQ, HIDDEN)


def test_kd_kl_loss_is_the_masked_teacher_cross_entropy():
    torch.manual_seed(0)
    s = torch.randn(1, 4, VOCAB)
    t = torch.randn(1, 4, VOCAB)
    mask = torch.tensor([[1, 1, 0, 1]])
    p = torch.softmax(t, -1)
    ref = -(p * torch.log_softmax(s, -1)).sum(-1)
    ref = (ref.view(-1) * mask.view(-1)).sum() / mask.sum()
    assert tail.kd_kl_loss(s, t, mask).item() == pytest.approx(ref.item(), rel=1e-5)
    assert tail.kd_kl_loss(t, t, mask).item() == pytest.approx(-(p * torch.log(p)).sum(-1).view(-1)[mask.view(-1) == 1]
                                                               .mean().item(), rel=1e-5)
    # compress_model uses the shared implementation
    assert compress_model.kd_kl_loss is tail.kd_kl_loss


def test_tail_objective_zero_at_target_and_grad_only_into_student():
    blocks, norm, head = _suffix()
    targets = _targets()
    obj = tail.TailLogitObjective(blocks, norm, head, targets, {"position_ids": None}, device="cpu")
    assert obj.kl(targets[1:2], 1).item() == pytest.approx(obj.entropy(1).item(), rel=1e-5)  # student == teacher
    student = (targets[1:2] + 0.5).requires_grad_(True)
    kl = obj.kl(student, 1)
    assert kl.item() > obj.entropy(1).item()
    kl.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert all(p.grad is None for b in blocks for p in b.parameters())
    assert head.weight.grad is None and norm.weight.grad is None


def test_tail_objective_excess_kl_is_nonnegative_and_decreases_toward_target():
    blocks, norm, head = _suffix()
    targets = _targets()
    obj = tail.TailLogitObjective(blocks, norm, head, targets, {}, device="cpu")
    far = obj.excess_kl(targets[0:1] + 1.0, 0).item()
    near = obj.excess_kl(targets[0:1] + 0.1, 0).item()
    assert far > near >= 0.0


def _tuneable_block():
    torch.manual_seed(3)
    b = nn.Module()
    b.lin = nn.Linear(HIDDEN, HIDDEN, bias=False)
    b.forward = lambda h, **kw: (b.lin(h),)
    return b


def test_tune_loop_with_tail_objective_reduces_the_kl(capsys):
    blocks, norm, head = _suffix()
    targets = _targets()
    obj = tail.TailLogitObjective(blocks, norm, head, targets, {}, device="cpu", mix=1.0)
    block = _tuneable_block()
    inputs = targets.clone()
    opt = torch.optim.Adam(block.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
    curv = compress_block.BlockCurvature(torch.ones(HIDDEN), None, False, {})
    with torch.no_grad():
        before = sum(obj.excess_kl(block(inputs[j:j + 1])[0], j).item() for j in range(N))
    with torch.enable_grad():
        compress_block._tune_loop(block, opt, sched, inputs, targets, curv, {}, batch_size=1, epochs=20,
                                  num_samples=N, tail=obj)
    with torch.no_grad():
        after = sum(obj.excess_kl(block(inputs[j:j + 1])[0], j).item() for j in range(N))
    assert after < 0.5 * before
    out = capsys.readouterr().out
    assert "KL" in out and "diag" in out


def test_tune_loop_mixed_objective_runs(capsys):
    blocks, norm, head = _suffix()
    targets = _targets()
    obj = tail.TailLogitObjective(blocks, norm, head, targets, {}, device="cpu", mix=0.5)
    block = _tuneable_block()
    opt = torch.optim.Adam(block.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
    curv = compress_block.BlockCurvature(torch.ones(HIDDEN), None, False, {})
    with torch.enable_grad():
        compress_block._tune_loop(block, opt, sched, targets.clone(), targets, curv, {}, batch_size=1, epochs=2,
                                  num_samples=N, tail=obj)
    assert "mix 0.5" in capsys.readouterr().out


def test_final_norm_and_head_lookup():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"model_type": "qwen3"})()
            self.model = nn.Module()
            self.model.norm = nn.LayerNorm(HIDDEN)
            self.lm_head = nn.Linear(HIDDEN, VOCAB)

    m = M()
    norm, head = U.get_final_norm_and_head(m)
    assert norm is m.model.norm and head is m.lm_head
    m.config.model_type = "banana"
    with pytest.raises(ValueError):
        U.get_final_norm_and_head(m)


def test_config_plumbing_validation_and_cache_keys():
    cfg = NanoQuantConfig(model_id="t")
    assert cfg["tail_logit_blocks"] == 0 and cfg["tail_logit_mix"] == 1.0
    base = NanoQuantConfig(model_id="tiny/model", num_calib_samples=4, seqlen=16)

    def over(**kw):
        c = dict(base)
        c.update(kw)
        return c

    for field, value in (("tail_logit_blocks", 4), ("tail_logit_mix", 0.5)):
        assert C.chain_keys(base, 2)[0] != C.chain_keys(over(**{field: value}), 2)[0], field
    assert C.kd_key(base, 2) != C.kd_key(over(tail_logit_blocks=4), 2)  # KD sits on top of the chain
    pipeline.validate_config(over(tail_logit_blocks=4, tail_logit_mix=0.5))
    for bad in ({"tail_logit_blocks": -1}, {"tail_logit_mix": 1.5}, {"tail_logit_mix": -0.1}):
        with pytest.raises(ValueError):
            pipeline.validate_config(over(**bad))
