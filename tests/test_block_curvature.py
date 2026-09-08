"""Tests for the block-output curvature helper (source selection and spectral conditioning)."""

import pytest
import torch
from torch import nn

from nanoquant.core import compress_block as cb
from nanoquant.core import curvature as cv
from nanoquant.modules.quant_config import NanoQuantConfig


def _spd(n, seed, spread=100.0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    lam = torch.logspace(0, torch.log10(torch.tensor(spread)).item(), n)
    return (Q * lam) @ Q.mT


def test_condition_curvature_identity_by_default():
    M = _spd(6, 0)
    out = cb.condition_curvature(M)
    assert torch.allclose(out, M.float(), atol=1e-5)


def test_condition_curvature_caps_condition_number_and_preserves_trace():
    M = _spd(8, 1, spread=1e4)
    out = cb.condition_curvature(M, cond_max=10.0)
    lam = torch.linalg.eigvalsh(out.double())
    assert lam[-1] / lam[0] == pytest.approx(10.0, rel=1e-3)
    assert out.trace().item() == pytest.approx(M.trace().item(), rel=1e-5)
    # eigenvectors are unchanged: the top eigenvector still is an eigenvector
    _, Q_m = torch.linalg.eigh(M.double())
    v = Q_m[:, -1]
    assert torch.allclose(out.double() @ v, (v @ out.double() @ v) * v, atol=1e-6)


def test_condition_curvature_power_and_mix():
    M = _spd(8, 2, spread=1e4)
    half = cb.condition_curvature(M, power=0.5)
    lam = torch.linalg.eigvalsh(half.double())
    assert lam[-1] / lam[0] == pytest.approx(100.0, rel=1e-3)
    assert half.trace().item() == pytest.approx(M.trace().item(), rel=1e-5)
    mixed = cb.condition_curvature(M, mix=0.25)
    assert torch.allclose(mixed.diagonal(), M.diagonal().float(), atol=1e-5)
    off = ~torch.eye(8, dtype=torch.bool)
    assert torch.allclose(mixed[off], 0.25 * M[off].float(), atol=1e-5)
    diag_only = cb.condition_curvature(M, mix=0.0)
    assert torch.allclose(diag_only, torch.diag(M.diagonal()).float(), atol=1e-5)


def test_spectrum_summary_values():
    s = cb.spectrum_summary(torch.eye(4))
    assert s["cond"] == pytest.approx(1.0) and s["eff_rank"] == pytest.approx(4.0)
    assert s["lam_max_over_mean_diag"] == pytest.approx(1.0) and s["top50_share"] == pytest.approx(1.0)
    assert "cond" in cv.format_spectrum(s)


def _block(with_plain: bool):
    down = nn.Linear(6, 4, bias=False)
    down.register_buffer("o_norm", torch.tensor([1.0, 2.0, 3.0, 4.0]), persistent=False)
    down.register_buffer("o_cov", torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0])) + 0.1, persistent=False)
    if with_plain:
        down.register_buffer("o_cov_plain", 2 * torch.eye(4) + 0.5, persistent=False)
    return {"mlp.down_proj": down, "mlp.up_proj": nn.Linear(4, 6, bias=False)}


def test_block_curvature_nkp_source_and_diag_default():
    cfg = NanoQuantConfig(model_id="t", curvature="kron")
    c = cb.block_curvature(_block(True), 4, cfg, "cpu")
    assert not c.optimize_dense
    assert torch.equal(c.importance, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(c.dense, torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0])) + 0.1)
    assert c.summary["cond"] > 1
    c = cb.block_curvature(_block(True), 4, NanoQuantConfig(model_id="t", curvature="kron", block_loss="mahalanobis"),
                           "cpu")
    assert c.optimize_dense


def test_block_curvature_plain_source():
    cfg = NanoQuantConfig(model_id="t", curvature="kron", block_loss_source="plain")
    c = cb.block_curvature(_block(True), 4, cfg, "cpu")
    assert torch.allclose(c.importance, torch.full((4,), 2.5))
    assert torch.allclose(c.dense, 2 * torch.eye(4) + 0.5)
    with pytest.raises(ValueError):
        cb.block_curvature(_block(False), 4, cfg, "cpu")


def test_block_curvature_without_stats_is_uniform():
    layers = {"mlp.down_proj": nn.Linear(6, 4, bias=False)}
    c = cb.block_curvature(layers, 4, NanoQuantConfig(model_id="t"), "cpu")
    assert torch.equal(c.importance, torch.ones(4)) and c.dense is None and c.summary == {}
    with pytest.raises(ValueError):
        cb.block_curvature(layers, 4, NanoQuantConfig(model_id="t", block_loss="mahalanobis"), "cpu")


def test_tune_loop_logs_both_losses_and_optimises_selected(capsys):
    torch.manual_seed(0)
    block = nn.Sequential(nn.Linear(4, 4, bias=False))
    x = torch.randn(6, 3, 4)
    tgt = torch.randn(6, 3, 4)
    dense = _spd(4, 3, spread=50.0)
    curv = cb.BlockCurvature(importance=dense.diagonal().clone(), dense=dense, optimize_dense=True, summary={})
    params = list(block.parameters())
    for p in params:
        p.requires_grad = True
    opt = torch.optim.SGD(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)

    class _Wrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, h, **kwargs):
            return (self.inner(h),)

    w = _Wrap(block)
    with torch.no_grad():
        before = cb.fused_weighted_mahalanobis(w(x)[0], tgt, dense).item()
    with torch.enable_grad():
        cb._tune_loop(w, opt, sched, x, tgt, curv, {}, batch_size=2, epochs=2, num_samples=6)
    with torch.no_grad():
        after = cb.fused_weighted_mahalanobis(w(x)[0], tgt, dense).item()
    assert after < before
    out = capsys.readouterr().out
    assert "Block Loss" in out and "| diag" in out and "| dense" in out


# ---------------------------------------------------------------- fresh ADMM input factor and diagnostics
class _MLPBlock(nn.Module):
    def __init__(self, d=8, hidden=6):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(d, hidden, bias=False)
        self.mlp.down_proj = nn.Linear(hidden, d, bias=False)

    def forward(self, x, **kwargs):
        return (x + self.mlp.down_proj(torch.tanh(self.mlp.up_proj(x))),)


def test_input_second_moment_matches_hooked_inputs():
    torch.manual_seed(4)
    block = _MLPBlock()
    x = torch.randn(3, 5, 8)
    R = cb.input_second_moment(block, block.mlp.down_proj, x, {}, num_samples=3)
    with torch.no_grad():
        a = torch.tanh(block.mlp.up_proj(x)).flatten(0, -2)
    assert torch.allclose(R, a.mT @ a / a.shape[0], atol=1e-6)
    assert R.shape == (6, 6) and torch.allclose(R, R.mT)


def test_factor_drift_zero_for_identical_and_detects_rotation():
    S = _spd(8, 5, spread=100.0)
    d = cb.factor_drift(S, S)
    assert d["rel_spectral"] == pytest.approx(0.0, abs=1e-6)
    assert d["max_angle_deg"] == pytest.approx(0.0, abs=1e-3)
    assert d["trace_ratio"] == pytest.approx(1.0)
    d2 = cb.factor_drift(S, 2 * S)  # pure rescaling: relative perturbation 1, no rotation
    assert d2["rel_spectral"] == pytest.approx(1.0, rel=1e-5) and d2["max_angle_deg"] == pytest.approx(0.0, abs=1e-3)
    assert d2["trace_ratio"] == pytest.approx(2.0)
    lam, Q = torch.linalg.eigh(S)
    rotated = (Q.flip(1) * lam) @ Q.flip(1).mT  # same spectrum, eigenvectors reversed
    d3 = cb.factor_drift(S, rotated, top_k=2)
    assert d3["max_angle_deg"] > 45.0 and d3["rel_spectral"] > 1.0
    assert "principal angle" in cb.format_drift(d3)


def test_mahalanobis_weight_error_dense_and_diagonal():
    torch.manual_seed(6)
    W = torch.randn(4, 3)
    W_hat = W + 0.1 * torch.randn(4, 3)
    L = _spd(4, 7)
    R = _spd(3, 8)
    E = W - W_hat
    assert cb.mahalanobis_weight_error(W, W_hat, L, R) == pytest.approx(torch.trace(L @ E @ R @ E.mT).item(), rel=1e-5)
    dl, dr = L.diagonal(), R.diagonal()
    expect = (dl.unsqueeze(1) * E.square() * dr.unsqueeze(0)).sum().item()
    assert cb.mahalanobis_weight_error(W, W_hat, dl, dr) == pytest.approx(expect, rel=1e-5)
    assert cb.mahalanobis_weight_error(W, W_hat, None, None) == pytest.approx(E.square().sum().item(), rel=1e-5)


def test_evaluate_block_loss_matches_fused_losses():
    torch.manual_seed(9)
    block = _MLPBlock()
    x, tgt = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    dense = _spd(8, 10)
    curv = cb.BlockCurvature(importance=dense.diagonal().clone(), dense=dense, optimize_dense=False, summary={})
    d, m = cb.evaluate_block_loss(block, x, tgt, curv, {}, num_samples=2)
    with torch.no_grad():
        y = block(x)[0]
        assert d == pytest.approx(cb.fused_weighted_mse(y, tgt, curv.importance).item() / tgt.numel(), rel=1e-5)
        assert m == pytest.approx(cb.fused_weighted_mahalanobis(y, tgt, dense).item() / tgt.numel(), rel=1e-5)
    curv_diag = cb.BlockCurvature(importance=curv.importance, dense=None, optimize_dense=False, summary={})
    assert cb.evaluate_block_loss(block, x, tgt, curv_diag, {}, num_samples=2)[1] is None


def test_factorize_and_replace_uses_fresh_input_factor(monkeypatch, capsys):
    torch.manual_seed(11)
    cfg = NanoQuantConfig(model_id="t", tune_fact=False, admm_outer_iters=2, block_diagnostics=True)
    block = _MLPBlock(d=8, hidden=6)
    lin = block.mlp.up_proj
    lin.register_buffer("i_norm", torch.rand(8) + 0.5, persistent=False)
    lin.register_buffer("o_norm", torch.rand(6) + 0.5, persistent=False)
    lin.register_buffer("i_cov", _spd(8, 12), persistent=False)
    lin.register_buffer("o_cov", _spd(6, 13), persistent=False)
    seen = {}

    def fake_factorize(W, i_norm, o_norm, mid_rank, **kwargs):
        seen["i_norm"], seen["i_cov"] = i_norm.clone(), kwargs["i_cov"].clone()
        A = torch.ones(mid_rank, W.shape[0])
        B = torch.ones(mid_rank, W.shape[1])
        return {"A": A, "B": B, "A_latent": A, "B_latent": B, "scale_pre": torch.ones(1, W.shape[1]),
                "scale_post": torch.ones(1, W.shape[0]), "W_final": torch.zeros_like(W)}

    monkeypatch.setattr(cb, "factorize_admm_nanoquant", fake_factorize)
    fresh = _spd(8, 14)
    cb.factorize_and_replace(block, "mlp.up_proj", 4, cfg, cache=None, input_factor=fresh)
    assert torch.allclose(seen["i_cov"], fresh) and torch.allclose(seen["i_norm"], fresh.diagonal())
    out = capsys.readouterr().out
    assert "stale-R" in out and "fresh-R" in out
    assert not hasattr(block.mlp.up_proj, "i_cov")
