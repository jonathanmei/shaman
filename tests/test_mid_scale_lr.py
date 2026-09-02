"""Tests for the separate optimizer group / learning rate of the per-rank middle scale."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nanoquant.core import admm_nq
from nanoquant.core.compress_block import get_param_group_config
from nanoquant.modules.linear import NanoQuantLinear

OUT, IN, RANK = 12, 16, 8


def _factors(mid_scale: bool):
    torch.manual_seed(3)
    W = torch.randn(OUT, IN) * 0.05
    out = admm_nq.factorize_admm_nanoquant(W, torch.ones(IN), torch.ones(OUT), mid_rank=RANK, outer_iters=20,
                                           rho_scheduler="linear", mid_scale=mid_scale, mid_scale_export="balanced")
    return SimpleNamespace(**out)


def _module(mid_scale: bool, do_train: bool = True) -> NanoQuantLinear:
    lin = nn.Linear(IN, OUT, bias=True)
    lin.__class__ = NanoQuantLinear
    lin.__quant_convert__(do_train=do_train, rank=RANK, factor_results=_factors(mid_scale))
    return lin


def _groups(module, **kw):
    cfg = get_param_group_config(module, binary_lr=1e-5, scale_lr=1e-5, bias_lr=1e-5, **kw)
    by_lr_and_names = []
    for g in cfg:
        names = sorted(n for n, p in module.named_parameters() if any(p is q for q in g["params"]))
        by_lr_and_names.append((tuple(names), g["lr"]))
    return by_lr_and_names


def test_param_groups_put_scale_mid_in_its_own_group():
    module = _module(mid_scale=True)
    groups = _groups(module, mid_scale_lr=1e-7)
    assert (("U_latent", "V_latent"), 1e-5) in groups
    assert (("scale_post", "scale_pre"), 1e-5) in groups
    assert (("scale_mid",), 1e-7) in groups
    assert (("bias",), 1e-5) in groups
    assert len(groups) == 4


def test_param_groups_mid_lr_inherits_scale_lr_by_default():
    module = _module(mid_scale=True)
    groups = _groups(module)
    assert (("scale_mid",), 1e-5) in groups
    assert (("scale_post", "scale_pre"), 1e-5) in groups


def test_param_groups_without_mid_scale_have_no_mid_group():
    module = _module(mid_scale=False)
    groups = _groups(module, mid_scale_lr=1e-7)
    assert not hasattr(module, "scale_mid")
    assert all("scale_mid" not in names for names, _ in groups)
    assert (("scale_post", "scale_pre"), 1e-5) in groups
    assert len(groups) == 3


@pytest.mark.parametrize("mid_scale, n_mid", [(True, 1), (False, 0)])
def test_scale_params_splits_outer_and_mid_for_kd(mid_scale, n_mid):
    module = _module(mid_scale=mid_scale)
    module.finalize()  # KD starts from a hardened module, as in the pipeline
    assert module.do_train is False
    outer, mid = module.scale_params()
    assert [p is q for p, q in zip(outer, [module.scale_pre, module.scale_post])] == [True, True]
    assert len(mid) == n_mid
    if mid_scale:
        assert mid[0] is module.scale_mid
    assert all(p.requires_grad for p in outer + mid)
    assert module.do_train is True
    # binaries stay frozen
    assert not module.U.requires_grad and not module.V.requires_grad


def test_adam_step_moves_mid_by_its_own_lr_only():
    module = _module(mid_scale=True).double()  # fp64 so that a 1e-7 step on a value near 1 is representable
    groups = get_param_group_config(module, binary_lr=1e-5, scale_lr=1e-5, bias_lr=1e-5, mid_scale_lr=1e-7)
    opt = torch.optim.AdamW(groups, weight_decay=0)
    before = {n: p.detach().clone() for n, p in module.named_parameters()}

    x = torch.randn(4, IN, dtype=torch.float64)
    loss = (module(x) - torch.randn(4, OUT, dtype=torch.float64)).square().mean()
    loss.backward()
    opt.step()

    # Adam's first step is +-lr wherever the gradient is non-zero
    d_mid = (module.scale_mid.detach() - before["scale_mid"]).abs().max()
    d_pre = (module.scale_pre.detach() - before["scale_pre"]).abs().max()
    d_post = (module.scale_post.detach() - before["scale_post"]).abs().max()
    assert d_mid <= 1.1e-7
    assert 0.9e-5 <= d_pre <= 1.1e-5 and 0.9e-5 <= d_post <= 1.1e-5
    # mean-one mid scale (up to the module's bf16 storage precision): the relative step equals the learning rate
    assert torch.isclose(before["scale_mid"].mean(), torch.tensor(1.0, dtype=torch.float64), atol=1e-2)
