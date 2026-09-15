"""Tests for the KL-Shampoo Kronecker fit, the two-sided spectral projection and their configuration keys."""

import math

import pytest
import torch
from torch import nn

from nanoquant.core import admm_nq, compress_model, pipeline
from nanoquant.core import curvature as cv
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


# ---------------------------------------------------------------- two-sided spectral projection
LAM = torch.tensor([100.0, 50.0, 3.0, 2.0, 1.0, 0.5])


def _hm(x):
    return x.numel() / (1.0 / x).sum()


def _gm(x):
    return x.log().mean().exp()


def test_project_spectrum_forward_keeps_spikes_and_flattens_tail():
    out = cv.project_spectrum(LAM, spike_rank=2)
    assert torch.allclose(out[:2], LAM[:2])
    assert torch.allclose(out[2:], torch.full((4,), LAM[2:].mean().item()))
    assert out.sum() == pytest.approx(LAM.sum().item())  # arithmetic mean preserves the trace
    assert torch.equal(cv.project_spectrum(LAM), LAM)
    assert torch.equal(cv.project_spectrum(LAM, spike_rank=6), LAM)
    shuffled = LAM[torch.tensor([3, 0, 5, 1, 4, 2])]  # order-independent
    out_s = cv.project_spectrum(shuffled, spike_rank=2)
    assert out_s[1] == 100.0 and out_s[3] == 50.0
    assert torch.allclose(out_s[[0, 2, 4, 5]], torch.full((4,), 1.625))


def test_project_spectrum_inverse_keeps_dips_with_harmonic_mean():
    out = cv.project_spectrum(LAM, dip_rank=2, flat_mean="hm")
    assert torch.allclose(out[4:], LAM[4:])  # the two smallest eigenvalues (spikes of the inverse) are exact
    assert torch.allclose(out[:4], torch.full((4,), _hm(LAM[:4]).item()))
    out_gm = cv.project_spectrum(LAM, dip_rank=2, flat_mean="gm")
    assert torch.allclose(out_gm[:4], torch.full((4,), _gm(LAM[:4]).item()))
    assert _hm(LAM[:4]) < _gm(LAM[:4]) < LAM[:4].mean()


def test_project_spectrum_two_sided_and_edge_cases():
    out = cv.project_spectrum(LAM, spike_rank=1, dip_rank=1, flat_mean="gm")
    assert out[0] == 100.0 and out[-1] == 0.5
    assert torch.allclose(out[1:5], torch.full((4,), _gm(LAM[1:5]).item()))
    assert torch.equal(cv.project_spectrum(LAM, spike_rank=3, dip_rank=3), LAM)  # nothing left to flatten
    assert torch.equal(cv.project_spectrum(LAM, spike_rank=4, dip_rank=4), LAM)
    with pytest.raises(ValueError):
        cv.project_spectrum(LAM, spike_rank=-1)
    with pytest.raises(ValueError):
        cv.project_spectrum(LAM, spike_rank=1, flat_mean="banana")


def test_temper_projection_then_power():
    lam = LAM.double()
    out = cv.temper_eigenvalues(lam, power=0.5, spike_rank=2)
    ref = cv.project_spectrum(lam, spike_rank=2).sqrt()
    ref = ref * lam.sum() / ref.sum()
    assert torch.allclose(out, ref)
    out2 = cv.temper_eigenvalues(lam, power=0.5, spike_rank=1, dip_rank=1, flat_mean="hm")
    assert out2.sum().item() == pytest.approx(lam.sum().item())
    assert torch.unique(torch.round(out2, decimals=8)).numel() == 3
    cov = _spd(6, 4, 1e4)
    spec = cv.SpectrumSpec(spike_rank=2, dip_rank=1)
    _, lam_t, _ = admm_nq._normalized_curvature(cov, cov.diagonal().sqrt(), torch.float64, 1e-12, spectrum=spec)
    assert torch.unique(torch.round(lam_t, decimals=5)).numel() == 4  # two spikes + one dip + one flat value


def test_projection_gaps_measure_the_dropped_middle():
    gaps = cv.projection_gaps(LAM, spike_rank=2, dip_rank=0)
    mid = LAM[2:]
    assert gaps["middle"] == 4
    assert gaps["log_am_gm"] == pytest.approx(math.log(mid.mean().item() / _gm(mid).item()), rel=1e-5)
    assert gaps["log_gm_hm"] == pytest.approx(math.log(_gm(mid).item() / _hm(mid).item()), rel=1e-5)
    flat = cv.projection_gaps(torch.tensor([5.0, 1.0, 1.0, 1.0]), spike_rank=1, dip_rank=0)
    assert flat["log_am_gm"] == pytest.approx(0.0, abs=1e-6) and flat["log_gm_hm"] == pytest.approx(0.0, abs=1e-6)
    whole = cv.projection_gaps(LAM, 0, 0)  # no projection: dispersion of the whole spectrum
    assert whole["middle"] == 6 and whole["log_am_gm"] > 0 and whole["log_gm_hm"] > 0
    none = cv.projection_gaps(LAM, 3, 3)
    assert none["middle"] == 0 and none["log_am_gm"] == 0.0 and none["log_gm_hm"] == 0.0


def test_spectrum_spec_from_config_and_validation():
    assert cv.SpectrumSpec.from_config({}) == cv.SpectrumSpec()
    assert cv.SpectrumSpec().is_identity
    spec = cv.SpectrumSpec.from_config({"admm_curvature_power": 0.5, "admm_curvature_spike_rank": 64,
                                        "admm_curvature_dip_rank": 8, "admm_curvature_flat_mean": "gm"})
    assert spec == cv.SpectrumSpec(power=0.5, spike_rank=64, dip_rank=8, flat_mean="gm")
    assert not spec.is_identity and not cv.SpectrumSpec(spike_rank=1).is_identity
    assert hash(spec) == hash(cv.SpectrumSpec(0.5, 64, 8, "gm"))
    assert torch.allclose(spec.apply(LAM), cv.temper_eigenvalues(LAM, 0.5, 64, 8, "gm"))
    assert torch.equal(cv.SpectrumSpec().apply(LAM), LAM)
    for bad in ({"spike_rank": -1}, {"dip_rank": -2}, {"flat_mean": "banana"}, {"power": 0.0}):
        with pytest.raises(ValueError):
            cv.SpectrumSpec(**bad)


def test_eig_cache_distinguishes_spectrum_specs():
    cov = _spd(6, 5, 1e3)
    norm = cov.diagonal().sqrt()
    cache = admm_nq.EigCache()
    cache.register(cov)
    a = admm_nq._normalized_curvature(cov, norm, torch.float64, 1e-12, spectrum=cv.SpectrumSpec(spike_rank=2),
                                      eig_cache=cache)
    b = admm_nq._normalized_curvature(cov, norm, torch.float64, 1e-12, spectrum=cv.SpectrumSpec(dip_rank=2),
                                      eig_cache=cache)
    c = admm_nq._normalized_curvature(cov, norm, torch.float64, 1e-12, spectrum=cv.SpectrumSpec(spike_rank=2),
                                      eig_cache=cache)
    assert len(cache) == 2
    assert not torch.allclose(a[1], b[1]) and torch.equal(a[1], c[1])


def test_factorize_reports_projection_gaps_and_two_sided_solution_differs():
    torch.manual_seed(0)
    W = torch.randn(12, 8)
    i_cov, o_cov = _spd(8, 2, 1e3), _spd(12, 3, 1e3)
    i_norm, o_norm = i_cov.diagonal(), o_cov.diagonal()
    diag = {}
    torch.manual_seed(1)
    a = admm_nq.factorize_admm_nanoquant(W, i_norm, o_norm, 4, outer_iters=5, i_cov=i_cov, o_cov=o_cov,
                                         diagnostics=diag)
    assert set(diag) == {"L", "R"} and diag["R"]["middle"] == 8 and diag["L"]["middle"] == 12
    diag2 = {}
    torch.manual_seed(1)
    b = admm_nq.factorize_admm_nanoquant(W, i_norm, o_norm, 4, outer_iters=5, i_cov=i_cov, o_cov=o_cov,
                                         spectrum=cv.SpectrumSpec(spike_rank=2, dip_rank=2, flat_mean="gm"),
                                         diagnostics=diag2)
    assert diag2["R"]["middle"] == 4 and diag2["L"]["middle"] == 8
    assert not torch.allclose(a["W_final"], b["W_final"])
    # transposed path threads the spec too
    torch.manual_seed(1)
    c = admm_nq.factorize_admm_nanoquant(W.mT, o_norm, i_norm, 4, outer_iters=5, i_cov=o_cov, o_cov=i_cov,
                                         is_transpose=True, spectrum=cv.SpectrumSpec(spike_rank=2, dip_rank=2,
                                                                                     flat_mean="gm"))
    assert c["W_final"].shape == W.mT.shape


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
                            gpu_budget_gb=500e-9)  # two layer groups
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
    # the projection knobs reach the block chain, the ADMM memo and the rank probe, and are validated
    W = torch.randn(8, 6)
    for knob, value in (("admm_curvature_spike_rank", 64), ("admm_curvature_dip_rank", 64),
                        ("admm_curvature_flat_mean", "hm")):
        assert C.chain_keys(base, 2)[0] != C.chain_keys(cfg(**{knob: value}), 2)[0], knob
        assert C.probe_key(base) != C.probe_key(cfg(**{knob: value})), knob
        assert C.admm_key(W, torch.rand(6), torch.rand(8), None, None, 4, base) != \
            C.admm_key(W, torch.rand(6), torch.rand(8), None, None, 4, cfg(**{knob: value})), knob
    pipeline.validate_config(cfg(curvature="kron", kron_fit="kl", admm_curvature_spike_rank=64,
                                 admm_curvature_dip_rank=64, admm_curvature_flat_mean="gm"))
    for bad in ({"admm_curvature_spike_rank": -1}, {"admm_curvature_dip_rank": -1},
                {"admm_curvature_flat_mean": "banana"}):
        with pytest.raises(ValueError):
            pipeline.validate_config(cfg(**bad))
    # the per-block perplexity evaluation is logging only: no key depends on it
    assert C.chain_keys(base, 2) == C.chain_keys(cfg(block_ppl_every=3), 2)
    assert "block_ppl_every" in base and base["block_ppl_every"] == 0


def test_eval_block_ppl_gating():
    f = compress_model.eval_block_ppl
    assert not any(f({}, i, 6) for i in range(6))  # default: never
    assert all(f({"block_diagnostics": True}, i, 6) for i in range(6))  # screens: every block
    assert [f({"block_ppl_every": 3}, i, 8) for i in range(8)] == [False, False, True, False, False, True, False, True]
    assert [f({"block_ppl_every": 1}, i, 2) for i in range(2)] == [True, True]
