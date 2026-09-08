"""Tests for spectral tempering, warm-started curvature collection / refresh, feature distillation and the
pre-KD checkpoint override."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nanoquant.core import admm_nq, compress_model, pipeline, teacher
from nanoquant.core import curvature as cv
from nanoquant.core import importance as imp
from nanoquant.modules.linear import NanoQuantLinear
from nanoquant.modules.quant_config import NanoQuantConfig
from nanoquant.utils import cache as C


def _spd(n, seed, spread=100.0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    lam = torch.logspace(0, torch.log10(torch.tensor(spread)).item(), n)
    return (Q * lam) @ Q.mT


def test_temper_eigenvalues_preserves_trace_and_bounds_condition():
    lam = torch.logspace(-2, 2, 9, dtype=torch.float64)
    same = cv.temper_eigenvalues(lam)
    assert torch.allclose(same, lam)
    half = cv.temper_eigenvalues(lam, power=0.5)
    assert half.sum() == pytest.approx(lam.sum().item())
    assert half.max() / half.min() == pytest.approx((lam.max() / lam.min()).sqrt().item())
    capped = cv.temper_eigenvalues(lam, cond_max=10.0)
    assert capped.sum() == pytest.approx(lam.sum().item())
    assert capped.max() / capped.min() == pytest.approx(10.0)


def test_normalized_curvature_tempering():
    cov = _spd(6, 1, spread=1e4)
    norm = cov.diagonal().sqrt()
    Sigma, lam, _ = admm_nq._normalized_curvature(cov, norm, torch.float64, 1e-12)
    assert torch.allclose(Sigma.diagonal(), torch.ones(6), atol=1e-5)  # unit-diagonal correlation form
    Sigma_t, lam_t, Q_t = admm_nq._normalized_curvature(cov, norm, torch.float64, 1e-12, power=0.5)
    assert lam_t.max() / lam_t.min() == pytest.approx((lam.max() / lam.min()).sqrt().item(), rel=1e-3)
    assert Sigma_t.trace().item() == pytest.approx(Sigma.trace().item(), rel=1e-4)
    assert torch.allclose(Sigma_t, (Q_t * lam_t) @ Q_t.mT, atol=1e-4)
    _, lam_c, _ = admm_nq._normalized_curvature(cov, norm, torch.float64, 1e-12, cond_max=20.0)
    assert lam_c.max() / lam_c.min() == pytest.approx(20.0, rel=1e-3)


def test_factorize_accepts_tempering_and_changes_solution():
    torch.manual_seed(0)
    W = torch.randn(12, 8)
    i_cov, o_cov = _spd(8, 2, 1e3), _spd(12, 3, 1e3)
    i_norm, o_norm = i_cov.diagonal(), o_cov.diagonal()
    torch.manual_seed(1)
    a = admm_nq.factorize_admm_nanoquant(W, i_norm, o_norm, 4, outer_iters=5, i_cov=i_cov, o_cov=o_cov)
    torch.manual_seed(1)
    b = admm_nq.factorize_admm_nanoquant(W, i_norm, o_norm, 4, outer_iters=5, i_cov=i_cov, o_cov=o_cov,
                                         curvature_power=0.5)
    assert a["W_final"].shape == b["W_final"].shape == W.shape
    assert not torch.allclose(a["W_final"], b["W_final"])


class _TinyMLP(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(6, 5, bias=False), nn.Tanh(), nn.Linear(5, 3, bias=False))


def _fake_loop(dataloader, model, dev, model_offload, use_truefisher):
    for batch in dataloader:
        inp = batch.clone().requires_grad_(True)
        loss = model(inp).square().mean()
        loss.backward()
        model.zero_grad(set_to_none=True)


def test_collect_stats_init_factors_warm_start(monkeypatch):
    """One pass from given factors equals the ALS update weighted by those factors (up to normalisation)."""
    torch.manual_seed(5)
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
    R0, L0 = _spd(6, 7), _spd(5, 8)
    raw = imp.collect_stats(model, dataloader, "cpu", strategy="dbf", curvature="kron", nkp_iters=1,
                            init_factors={"i_cov": {"0": R0}, "o_cov": {"0": L0}})
    L_ref, R_ref = imp.nkp_update(x, delta, L_prev=L0 / L0.norm(), R_prev=R0 / R0.norm())
    L, R = raw["o_cov"]["0"], raw["i_cov"]["0"]
    assert torch.allclose(L / L.norm(), L_ref / L_ref.norm(), atol=1e-4)
    assert torch.allclose(R / R.norm(), R_ref / R_ref.norm(), atol=1e-4)


class _TinyModel(nn.Module):
    """Minimal stand-in with the attributes the block loop and collect_stats touch."""
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyMLP(), _TinyMLP()])
        self.config = SimpleNamespace(use_cache=False, model_type="tiny")
        self.gc = False

    def forward(self, x):
        for blk in self.model.layers:
            x = x[..., :6] + nn.functional.pad(blk(x[..., :6]), (0, 3))
        return x

    def gradient_checkpointing_enable(self, **kw):
        self.gc = True

    def gradient_checkpointing_disable(self):
        self.gc = False


def test_refresh_block_curvature_skips_quantised_layers(monkeypatch):
    torch.manual_seed(9)
    model = _TinyModel()
    monkeypatch.setattr(imp, "_run_calibration_loop", _fake_loop)
    dataloader = [torch.randn(1, 4, 6) for _ in range(2)]
    # register stale factors on every linear, then "quantise" the first block's first layer
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.register_buffer("i_cov", torch.eye(m.in_features), persistent=False)
            m.register_buffer("o_cov", torch.eye(m.out_features), persistent=False)
            m.register_buffer("i_norm", torch.ones(m.in_features), persistent=False)
            m.register_buffer("o_norm", torch.ones(m.out_features), persistent=False)
    q = model.model.layers[0][0]
    q.__class__ = NanoQuantLinear
    A = torch.ones(2, 5)
    B = torch.ones(2, 6)
    q.__quant_convert__(do_train=False, rank=2, factor_results=SimpleNamespace(
        A=A, B=B, A_latent=A, B_latent=B, scale_pre=torch.ones(1, 6), scale_post=torch.ones(1, 5),
        W_final=torch.zeros(5, 6)))
    q.float()  # the rest of the tiny model is fp32
    stale = model.model.layers[1][0].i_cov.clone()
    cfg = NanoQuantConfig(model_id="t", curvature="kron", calib_strategy="dbf", calib_shrinkage=0.2,
                          curvature_refresh_iters=1)
    n = compress_model.refresh_block_curvature(model, dataloader, "cpu", cfg)
    assert n == 3  # three remaining nn.Linear layers
    fresh = model.model.layers[1][0].i_cov
    assert fresh.shape == stale.shape and not torch.allclose(fresh, stale)
    assert torch.allclose(model.model.layers[1][0].i_norm, fresh.diagonal())
    assert not model.training and not model.gc


def test_feature_loss_relative_error():
    hs = [torch.randn(1, 4, 3) for _ in range(3)]
    mask = torch.ones(1, 4, dtype=torch.int)
    assert compress_model.feature_loss(hs, hs, mask).item() == pytest.approx(0.0)
    zeros = [torch.zeros_like(h) for h in hs]
    assert compress_model.feature_loss(zeros, hs, mask).item() == pytest.approx(1.0)  # embedding entry skipped
    mask2 = torch.tensor([[1, 1, 0, 0]])
    doubled = [hs[0]] + [2 * h for h in hs[1:]]
    assert compress_model.feature_loss(doubled, hs, mask2).item() == pytest.approx(1.0, rel=1e-5)


class _TinyLM(nn.Module):
    def __init__(self, vocab=11, d=6):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids, output_hidden_states=False):
        h0 = self.emb(ids)
        h1 = torch.tanh(h0)
        out = SimpleNamespace(logits=self.head(h1))
        if output_hidden_states:
            out.hidden_states = (h0, h1)
        return out


def test_teacher_hidden_states_online_only():
    model = _TinyLM()
    samples = [torch.randint(0, 11, (1, 5))]
    t = teacher.TeacherLogits("online", model, samples, "cpu")
    logits, hidden = t.get(0, samples[0], hidden=True)
    assert logits.shape == (1, 5, 11) and len(hidden) == 2 and hidden[1].shape == (1, 5, 6)
    assert torch.allclose(logits, t.get(0, samples[0]))
    t_ram = teacher.TeacherLogits("ram", model, samples, "cpu")
    with pytest.raises(ValueError):
        t_ram.get(0, samples[0], hidden=True)


def test_validate_config_new_fields(tmp_path):
    pipeline.validate_config(NanoQuantConfig(model_id="t", curvature="kron", curvature_refresh_every=7))
    pipeline.validate_config(NanoQuantConfig(model_id="t", model_kd_teacher="online", model_kd_feature_weight=1.0))
    ck = tmp_path / "pre.pt"
    ck.write_bytes(b"x")
    pipeline.validate_config(NanoQuantConfig(model_id="t", pre_kd_checkpoint=str(ck)))
    for bad in ({"curvature_refresh_every": 7}, {"curvature_refresh_every": -1, "curvature": "kron"},
                {"model_kd_feature_weight": 1.0}, {"pre_kd_checkpoint": str(tmp_path / "missing.pt")}):
        with pytest.raises(ValueError):
            pipeline.validate_config(NanoQuantConfig(model_id="t", **bad))


def test_cache_keys_track_new_fields():
    base = NanoQuantConfig(model_id="tiny/model", num_calib_samples=4, seqlen=16)

    def cfg(**over):
        c = dict(base)
        c.update(over)
        return c

    W = torch.randn(8, 6)
    k = C.admm_key(W, torch.rand(6), torch.rand(8), None, None, 4, base)
    assert k != C.admm_key(W, torch.rand(6), torch.rand(8), None, None, 4, cfg(admm_curvature_power=0.5))
    keys = C.chain_keys(base, 2)
    for field, value in (("admm_curvature_power", 0.5), ("admm_curvature_cond_max", 10.0),
                         ("curvature_refresh_every", 7), ("curvature_refresh_iters", 2)):
        assert keys[0] != C.chain_keys(cfg(**{field: value}), 2)[0], field
    assert keys == C.chain_keys(cfg(model_kd_feature_weight=1.0, pre_kd_checkpoint="x.pt"), 2)
    assert C.kd_key(base, 2) != C.kd_key(cfg(model_kd_feature_weight=1.0), 2)
    assert C.kd_key(base, 2) != C.kd_key(cfg(pre_kd_checkpoint="x.pt"), 2)
