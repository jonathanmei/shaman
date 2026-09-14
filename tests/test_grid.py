"""Tests for the KL-Shampoo Kronecker fit and its configuration keys."""

import pytest
import torch
from torch import nn

from nanoquant.core import pipeline
from nanoquant.core import importance as imp
from nanoquant.modules.quant_config import NanoQuantConfig
from nanoquant.utils import cache as C


def _spd(n, seed, spread=100.0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    lam = torch.logspace(0, torch.log10(torch.tensor(spread)).item(), n)
    return (Q * lam) @ Q.mT


class _TinyMLP(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(6, 5, bias=False), nn.Tanh(), nn.Linear(5, 3, bias=False))


def _fake_loop(dataloader, model, dev, model_offload, use_truefisher):
    for batch in dataloader:
        inp = batch.clone().requires_grad_(True)
        loss = model(inp).square().mean()
        loss.backward()
        model.zero_grad(set_to_none=True)


# ---------------------------------------------------------------- KL-Shampoo fit
def test_als_weights_kl_uses_damped_inverses():
    R = _spd(5, 11, spread=10.0)
    w = imp._als_weights({"i_cov": {"a": R}, "o_cov": {}}, "kl")
    damped = R + 1e-3 * R.diagonal().mean() * torch.eye(5)  # Tikhonov damping of _damped_inverse
    assert torch.allclose(w["i_cov"]["a"] @ damped, torch.eye(5), atol=1e-4)
    assert torch.allclose(w["i_cov"]["a"] @ R, torch.eye(5), atol=1e-2)  # well conditioned: nearly the exact inverse
    assert imp._als_weights({"i_cov": {"a": R}, "o_cov": {}}, "frobenius")["i_cov"]["a"] is R
    with pytest.raises(ValueError):
        imp._als_weights({"i_cov": {}, "o_cov": {}}, "banana")


def test_nkp_fit_kl_pass1_equals_frobenius_and_pass2_uses_inverse_weights():
    torch.manual_seed(12)
    x, delta = torch.randn(30, 4), torch.randn(30, 3)
    L1, R1 = imp.nkp_fit(x, delta, num_iters=1, fit="kl")
    L1f, R1f = imp.nkp_fit(x, delta, num_iters=1, fit="frobenius")
    assert torch.allclose(L1, L1f) and torch.allclose(R1, R1f)
    L2, R2 = imp.nkp_fit(x, delta, num_iters=2, fit="kl")
    L_ref, R_ref = imp.nkp_update(x, delta, L_prev=imp._damped_inverse(L1), R_prev=imp._damped_inverse(R1))
    assert torch.allclose(L2, L_ref / L_ref.norm(), atol=1e-5)
    assert torch.allclose(R2, R_ref / R_ref.norm(), atol=1e-5)


def test_kl_fit_is_less_distorted_by_massive_tokens():
    """A few tokens with 30x inputs and a fixed gradient direction dominate the Frobenius output factor; the
    leverage-weighted KL fit stays closer to the bulk gradient covariance."""
    torch.manual_seed(13)
    T, n_in, n_out = 4000, 6, 5
    R0, L0 = _spd(n_in, 21, spread=4.0), _spd(n_out, 22, spread=4.0)
    x = torch.randn(T, n_in) @ torch.linalg.cholesky(R0).mT
    delta = torch.randn(T, n_out) @ torch.linalg.cholesky(L0).mT
    u = torch.randn(n_out)
    u = u / u.norm()
    idx = torch.arange(0, T, 200)  # 20 "massive" tokens
    x[idx] *= 30.0
    delta[idx] = u * delta[idx].norm(dim=1, keepdim=True)

    def rel_err(L):
        return ((L / L.norm()) - (L0 / L0.norm())).norm().item()

    L_f, _ = imp.nkp_fit(x, delta, num_iters=3, fit="frobenius")
    L_k, _ = imp.nkp_fit(x, delta, num_iters=3, fit="kl")
    assert rel_err(L_k) < rel_err(L_f)
    assert rel_err(L_f) > 0.5  # the Frobenius factor is dominated by u u^T


def test_collect_stats_kl_matches_offline_fit(monkeypatch):
    torch.manual_seed(14)
    model = _TinyMLP()
    dataloader = [torch.randn(1, 9, 6) for _ in range(2)]
    monkeypatch.setattr(imp, "_run_calibration_loop", _fake_loop)
    xs, ds = [], []
    lin = model[0]
    h1 = lin.register_forward_hook(lambda m, i, o: xs.append(i[0].detach().flatten(0, -2).float()))
    h2 = lin.register_full_backward_hook(lambda m, gi, go: ds.append(go[0].detach().flatten(0, -2).float()))
    _fake_loop(dataloader, model, "cpu", False, False)
    h1.remove()
    h2.remove()
    x, delta = torch.cat(xs), torch.cat(ds) * imp.GRAD_SCALE_FACTOR
    raw = imp.collect_stats(model, dataloader, "cpu", strategy="dbf", curvature="kron", nkp_iters=3, fit="kl")
    L_ref, R_ref = imp.nkp_fit(x, delta, num_iters=3, fit="kl")
    assert torch.allclose(raw["o_cov"]["0"] / raw["o_cov"]["0"].norm(), L_ref, atol=1e-4)
    assert torch.allclose(raw["i_cov"]["0"] / raw["i_cov"]["0"].norm(), R_ref, atol=1e-4)
    # grouped accumulation reproduces the KL fit too
    torch.manual_seed(14)
    got = imp.collect_stats(_TinyMLP(), dataloader, "cpu", strategy="dbf", curvature="kron", nkp_iters=2, fit="kl",
                            gpu_budget_gb=600e-9)
    torch.manual_seed(14)
    ref = imp.collect_stats(_TinyMLP(), dataloader, "cpu", strategy="dbf", curvature="kron", nkp_iters=2, fit="kl")
    for key in ("i_cov", "o_cov"):
        for n in ref[key]:
            assert torch.allclose(got[key][n], ref[key][n], atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError):
        imp.collect_stats(_TinyMLP(), dataloader, "cpu", curvature="kron", fit="banana")


# ---------------------------------------------------------------- keys and validation
def test_grid_config_keys():
    base = NanoQuantConfig(model_id="tiny/model", num_calib_samples=4, seqlen=16)

    def cfg(**over):
        c = dict(base)
        c.update(over)
        return c

    assert C.stats_key(base) != C.stats_key(cfg(kron_fit="kl"))
    assert C.chain_keys(base, 2)[0] != C.chain_keys(cfg(kron_fit="kl"), 2)[0]  # through the calibration key
    pipeline.validate_config(cfg(curvature="kron", kron_fit="kl", admm_curvature_power=0.5))
    with pytest.raises(ValueError):
        pipeline.validate_config(cfg(kron_fit="banana"))
