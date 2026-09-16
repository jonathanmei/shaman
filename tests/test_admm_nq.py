"""Tests for the Mahalanobis (Kronecker-curvature) ADMM path and the middle-scale export in admm_nq."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nanoquant.core import admm_nq
from nanoquant.core.curvature import SpectrumSpec
from nanoquant.modules.quant_config import NanoQuantConfig
from nanoquant.utils import utils as nq_utils

RANK = 8


def _spd(n: int, corr: float = 0.0) -> torch.Tensor:
    """Unit-diagonal SPD matrix with constant off-diagonal correlation ``corr``."""
    return (1 - corr) * torch.eye(n) + corr * torch.ones(n, n)


def _scaled_cov(corr_mat: torch.Tensor, norm_vec: torch.Tensor) -> torch.Tensor:
    """Dense factor whose diagonal equals ``norm_vec`` and whose correlation is ``corr_mat``."""
    s = norm_vec.sqrt()
    return corr_mat * s.unsqueeze(1) * s.unsqueeze(0)


def _run(W, i_norm, o_norm, seed, **kw):
    torch.manual_seed(seed)
    kw.setdefault("rho_scheduler", "linear")
    return admm_nq.factorize_admm_nanoquant(W, i_norm, o_norm, mid_rank=RANK, outer_iters=30, print_admm_steps=False,
                                            **kw)


def _deployed(out):
    """diag(scale_post) S_A diag(scale_mid) S_B diag(scale_pre) from a factorisation result."""
    U = out["A"].mT  # (out, rank)
    V = out["B"]  # (rank, in)
    y = V * out["scale_pre"]  # (rank, in)
    y = y * out["scale_mid"].view(-1, 1)
    return (U @ y) * out["scale_post"].view(-1, 1)


# --------------------------------------------------------------------------------------
# Sylvester solver
# --------------------------------------------------------------------------------------
def test_sylvester_with_identity_matches_euclidean_solve():
    torch.manual_seed(0)
    n_in, k, n_out, rho, reg = 12, 4, 7, 0.3, 3e-2
    # unit-norm design columns, as in the ADMM (then the legacy rho * diag_mean equals rho)
    X = torch.randn(n_in, k)
    X = X / X.norm(dim=0, keepdim=True)
    Y = torch.randn(n_in, n_out)
    Z = torch.randn(k, n_out)
    U = torch.randn(k, n_out)

    euclid = admm_nq._admm_solve_step(X, Y, Z, U, rho, reg)  # (k, out)

    M = X.mT @ X
    C = (X.mT @ Y + rho * (Z - U)).mT  # (out, k)
    syl = admm_nq._sylvester_solve_step(torch.eye(n_out), torch.ones(n_out), M, C, rho, reg)  # (out, k)

    assert torch.allclose(syl, euclid.mT, atol=1e-4, rtol=1e-4)


def test_sylvester_solves_generalised_equation():
    torch.manual_seed(1)
    n, k, rho, reg = 9, 3, 0.5, 1e-2
    A = torch.randn(n, n)
    Sigma = A @ A.mT / n + 0.1 * torch.eye(n)
    lam, Q = torch.linalg.eigh(Sigma)
    B = torch.randn(k, k)
    M = B @ B.mT
    C = torch.randn(n, k)

    F = admm_nq._sylvester_solve_step(Q, lam, M, C, rho, reg)
    sigma = admm_nq._sylvester_stabilizer(lam, M, rho, reg)
    lhs = Sigma @ F @ M + sigma * F
    assert torch.allclose(lhs, C, atol=1e-4, rtol=1e-4)


def test_sylvester_stabilizer_keeps_rho_and_scales_reg():
    M = torch.diag(torch.tensor([1.0, 3.0]))  # mean diag = 2
    lam = torch.tensor([0.5, 1.5, 1.0])  # mean = 1
    sigma = admm_nq._sylvester_stabilizer(lam, M, rho=0.5, reg=0.03)
    assert abs(float(sigma) - (0.5 + 0.03 * 2.0)) < 1e-6
    # unit-diagonal M and Sigma = I -> rho + reg, exactly the legacy stabiliser
    sigma_unit = admm_nq._sylvester_stabilizer(torch.ones(4), torch.eye(3), rho=0.5, reg=0.03)
    assert abs(float(sigma_unit) - 0.53) < 1e-6


# --------------------------------------------------------------------------------------
# Mahalanobis ADMM
# --------------------------------------------------------------------------------------
def test_identity_covariance_reproduces_diagonal_path():
    torch.manual_seed(2)
    W = torch.randn(24, 16)
    i_norm = torch.rand(16) + 0.5
    o_norm = torch.rand(24) + 0.5

    ref = _run(W, i_norm, o_norm, seed=7)
    got = _run(W, i_norm, o_norm, seed=7, i_cov=torch.diag(i_norm), o_cov=torch.diag(o_norm))

    assert set(got) == set(ref)
    for key in ("W_final", "A", "B", "scale_pre", "scale_post"):
        rel = (got[key] - ref[key]).norm() / ref[key].norm().clamp(1e-12)
        assert rel < 1e-2, f"{key}: relative difference {rel:.3e}"


def _mahalanobis_error(W, W_hat, i_norm, o_norm, i_cov, o_cov):
    E = (W - W_hat) * o_norm.sqrt().unsqueeze(1) * i_norm.sqrt().unsqueeze(0)
    Lt = o_cov / o_norm.sqrt().unsqueeze(1) / o_norm.sqrt().unsqueeze(0)
    Rt = i_cov / i_norm.sqrt().unsqueeze(1) / i_norm.sqrt().unsqueeze(0)
    return torch.trace(Lt @ E @ Rt @ E.mT).item()


@pytest.mark.parametrize("corr", [0.3, 0.6, 0.9])
def test_mahalanobis_admm_beats_euclidean_in_its_own_metric(corr):
    torch.manual_seed(3)
    n_out, n_in = 32, 24
    W = torch.randn(n_out, n_in)
    i_norm = torch.rand(n_in) + 0.5
    o_norm = torch.rand(n_out) + 0.5
    o_cov = _scaled_cov(_spd(n_out, corr=corr), o_norm)
    i_cov = _scaled_cov(_spd(n_in, corr=corr), i_norm)

    diag = _run(W, i_norm, o_norm, seed=11)
    kron = _run(W, i_norm, o_norm, seed=11, i_cov=i_cov, o_cov=o_cov)

    err_diag = _mahalanobis_error(W, diag["W_final"], i_norm, o_norm, i_cov, o_cov)
    err_kron = _mahalanobis_error(W, kron["W_final"], i_norm, o_norm, i_cov, o_cov)
    assert err_kron < err_diag


@pytest.mark.parametrize("eigh_dtype", [torch.float64, torch.float32])
def test_transpose_path_with_covariances(eigh_dtype):
    torch.manual_seed(4)
    n_out, n_in = 16, 32  # in > out -> is_transpose
    W = torch.randn(n_out, n_in)
    i_norm = torch.rand(n_in) + 0.5
    o_norm = torch.rand(n_out) + 0.5
    i_cov = _scaled_cov(_spd(n_in, 0.3), i_norm)
    o_cov = _scaled_cov(_spd(n_out, 0.3), o_norm)

    out = _run(W, i_norm, o_norm, seed=5, is_transpose=True, i_cov=i_cov, o_cov=o_cov, eigh_dtype=eigh_dtype)

    assert out["W_final"].shape == (n_out, n_in)
    assert out["A"].shape == (RANK, n_out)
    assert out["B"].shape == (RANK, n_in)
    assert out["scale_pre"].shape == (1, n_in)
    assert out["scale_post"].shape == (1, n_out)
    assert torch.isfinite(out["W_final"]).all()


# --------------------------------------------------------------------------------------
# Middle scale export
# --------------------------------------------------------------------------------------
def test_mid_scale_export_is_exact_deployed_form():
    torch.manual_seed(6)
    n_out, n_in = 24, 16
    W = torch.randn(n_out, n_in)
    i_norm = torch.rand(n_in) + 0.5
    o_norm = torch.rand(n_out) + 0.5

    out = _run(W, i_norm, o_norm, seed=8, mid_scale=True)

    assert out["scale_mid"].shape == (1, RANK)
    assert out["A"].shape == (RANK, n_out) and out["B"].shape == (RANK, n_in)
    assert torch.all(out["A"].abs() == 1) and torch.all(out["B"].abs() == 1)
    assert torch.all(out["scale_mid"] > 0)
    assert torch.allclose(_deployed(out), out["W_final"], atol=1e-4, rtol=1e-4)
    # a real approximation, not garbage
    rel_err = (out["W_final"] - W).norm() / W.norm()
    assert rel_err < 1.0


def test_mid_scale_transpose_swaps_pre_post_and_keeps_mid():
    torch.manual_seed(7)
    n_out, n_in = 16, 32
    W = torch.randn(n_out, n_in)
    i_norm = torch.rand(n_in) + 0.5
    o_norm = torch.rand(n_out) + 0.5

    out = _run(W, i_norm, o_norm, seed=9, is_transpose=True, mid_scale=True)

    assert out["scale_mid"].shape == (1, RANK)
    assert out["scale_pre"].shape == (1, n_in)
    assert out["scale_post"].shape == (1, n_out)
    assert out["W_final"].shape == (n_out, n_in)
    assert torch.allclose(_deployed(out), out["W_final"], atol=1e-4, rtol=1e-4)


def test_mid_scale_with_covariances_runs():
    torch.manual_seed(8)
    n_out, n_in = 24, 16
    W = torch.randn(n_out, n_in)
    i_norm = torch.rand(n_in) + 0.5
    o_norm = torch.rand(n_out) + 0.5
    out = _run(W, i_norm, o_norm, seed=10, mid_scale=True, i_cov=_scaled_cov(_spd(n_in, 0.5), i_norm),
               o_cov=_scaled_cov(_spd(n_out, 0.5), o_norm))
    assert torch.allclose(_deployed(out), out["W_final"], atol=1e-4, rtol=1e-4)


def test_without_mid_scale_export_is_the_legacy_mean_magnitude_one():
    torch.manual_seed(9)
    W = torch.randn(24, 16)
    i_norm = torch.rand(16) + 0.5
    o_norm = torch.rand(24) + 0.5
    out = _run(W, i_norm, o_norm, seed=12)
    assert "scale_mid" not in out
    # legacy: B is the continuous factor, scales are mean magnitudes, W_final = A_final @ B_final
    assert torch.allclose(out["scale_pre"], out["B"].abs().mean(dim=0).view(1, -1))
    assert torch.allclose(out["scale_post"], out["A"].mT.abs().mean(dim=1).view(1, -1))
    assert torch.allclose(out["W_final"], out["A"].mT @ out["B"], atol=1e-5)


# --------------------------------------------------------------------------------------
# Config and rank budget
# --------------------------------------------------------------------------------------
def test_config_defaults_keep_legacy_behaviour():
    cfg = NanoQuantConfig()
    assert cfg["curvature"] == "diag"
    assert cfg["kron_nkp_iters"] == 3
    assert cfg["kron_stats_device"] == "cpu"
    assert cfg["kron_eigh_dtype"] == "float64"
    assert cfg["admm_mid_scale"] is False


def test_has_mid_scale_helper():
    assert nq_utils.has_mid_scale({"admm_type": "dbf", "admm_mid_scale": False})
    assert nq_utils.has_mid_scale({"admm_type": "nanoquant", "admm_mid_scale": True})
    assert not nq_utils.has_mid_scale({"admm_type": "nanoquant", "admm_mid_scale": False})
    assert not nq_utils.has_mid_scale({"admm_type": "nanoquant"})  # old configs without the key


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(1024, 1024, bias=False)


def _fake_model():
    return SimpleNamespace(config=SimpleNamespace(model_type="llama"), model=SimpleNamespace(layers=[_Block()]))


def test_calculate_ranks_pays_for_mid_scale():
    # 1024x1024 at 1.032 bpw: 2 scales -> 512.4 -> 512, 3 scales -> 508.4 -> 480 (ranks are multiples of 32)
    base = {"bits": 1.032, "admm_type": "nanoquant"}
    two = nq_utils.calculate_ranks(_fake_model(), ["self_attn.q_proj"], {**base, "admm_mid_scale": False})
    three = nq_utils.calculate_ranks(_fake_model(), ["self_attn.q_proj"], {**base, "admm_mid_scale": True})
    dbf = nq_utils.calculate_ranks(_fake_model(), ["self_attn.q_proj"], {"bits": 1.032, "admm_type": "dbf"})
    assert two["0.self_attn.q_proj"] == 512
    assert three["0.self_attn.q_proj"] == 480
    assert three == dbf


def test_config_dataclass_accepts_every_config_key():
    """main.py builds NanoQuantConfigDataclass(**NanoQuantConfig(...)): every key must be a dataclass field."""
    from nanoquant.modules.hub import NanoQuantConfigDataclass

    cfg = NanoQuantConfig(model_id="t")
    NanoQuantConfigDataclass(**cfg)
    assert set(cfg) <= set(NanoQuantConfigDataclass.__dataclass_fields__)


# --------------------------------------------------------------------------------------
# Eigendecomposition cache for shared curvature factors
# --------------------------------------------------------------------------------------
def _count_eigh(monkeypatch):
    calls = {"n": 0}
    real = torch.linalg.eigh

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigh", counting)
    return calls


def test_normalized_curvature_eig_cache_hits_same_matrix(monkeypatch):
    calls = _count_eigh(monkeypatch)
    norm = torch.rand(6) + 0.5
    cov = _scaled_cov(_spd(6, 0.4), norm)
    cache = admm_nq.EigCache()
    admm_nq._normalized_curvature(cov, norm, torch.float64, 1e-12, eig_cache=cache)  # unregistered: not stored
    assert calls["n"] == 1 and len(cache) == 0
    cache.register(cov)
    first = admm_nq._normalized_curvature(cov, norm, torch.float64, 1e-12, eig_cache=cache)
    second = admm_nq._normalized_curvature(cov, norm, torch.float64, 1e-12, eig_cache=cache)
    assert calls["n"] == 2 and len(cache) == 1
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    admm_nq._normalized_curvature(cov, norm, torch.float64, 1e-12, spectrum=SpectrumSpec(power=0.5),
                                  eig_cache=cache)  # other conditioning
    assert calls["n"] == 3 and len(cache) == 2
    admm_nq._normalized_curvature(cov.clone(), norm, torch.float64, 1e-12, eig_cache=cache)  # other tensor
    assert calls["n"] == 4 and len(cache) == 2
    admm_nq._normalized_curvature(cov, norm, torch.float64, 1e-12)  # no cache: always computes
    assert calls["n"] == 5
    cache.clear()
    assert len(cache) == 0


def test_factorize_with_eig_cache_shares_the_input_factor(monkeypatch):
    calls = _count_eigh(monkeypatch)
    torch.manual_seed(31)
    n_in = 12
    i_norm = torch.rand(n_in) + 0.5
    i_cov = _scaled_cov(_spd(n_in, 0.3), i_norm)
    cache = admm_nq.EigCache()
    cache.register(i_cov)
    # q-like layer (out > in) and k-like layer (out < in -> transposed path) sharing the input factor
    for n_out, transpose in ((16, False), (8, True)):
        W = torch.randn(n_out, n_in)
        o_norm = torch.rand(n_out) + 0.5
        o_cov = _scaled_cov(_spd(n_out, 0.3), o_norm)
        _run(W, i_norm, o_norm, seed=3, is_transpose=transpose, i_cov=i_cov, o_cov=o_cov, eig_cache=cache)
    # 2 output factors + 1 shared input factor + the per-iteration k x k eigh of both runs (30 iters x 2 updates)
    assert calls["n"] == 3 + 2 * 30 * 2
    assert len(cache) == 1


# --------------------------------------------------------------------------------------
# Low-rank-plus-identity fast path for projected spectra
# --------------------------------------------------------------------------------------
def _projected_factor(n, seed, spec, spread=1e3):
    """Unit-diagonal-ish SPD factor whose spectrum is exactly in the two-sided spike-plus-flat family."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    lam = spec.apply(torch.logspace(0, torch.log10(torch.tensor(spread)).item(), n))
    return (Q * lam) @ Q.mT, lam, Q


def test_structured_sylvester_matches_dense_and_solves_the_equation():
    torch.manual_seed(5)
    n, k = 40, 6
    spec = SpectrumSpec(spike_rank=3, dip_rank=2, flat_mean="gm")
    Sigma, lam, Q = _projected_factor(n, 7, spec)
    fac = admm_nq.CurvatureFactor(Sigma, lam, Q, spec)
    assert fac.structured and fac.U.shape == (n, 5)
    B = torch.randn(k, n)
    M = B @ B.mT
    C = torch.randn(n, k)
    rho, reg = 0.7, 3e-2
    F_dense = admm_nq._sylvester_solve_step(Q, lam, M, C, rho, reg, 1e-12, torch.float64)
    F_fast = fac.sylvester(M, C, rho, reg, 1e-12, torch.float64)
    assert (F_fast - F_dense).norm() / F_dense.norm() < 1e-4
    # the default k x k eigh precision is fp32 and is lossless for the X-update
    assert admm_nq.SYLVESTER_EIGH_DTYPE == torch.float32
    F32 = fac.sylvester(M, C, rho, reg, 1e-12)
    F32_dense = admm_nq._sylvester_solve_step(Q, lam, M, C, rho, reg, 1e-12)
    assert (F32 - F_dense).norm() / F_dense.norm() < 1e-4
    assert (F32_dense - F_dense).norm() / F_dense.norm() < 1e-4
    sigma = admm_nq._sylvester_stabilizer(lam, M, rho, reg)
    residual = Sigma @ F_fast @ M + sigma * F_fast - C
    assert residual.norm() / C.norm() < 1e-4
    # products
    X = torch.randn(n, k)
    assert (fac.left(X) - Sigma @ X).norm() / (Sigma @ X).norm() < 1e-5
    assert (fac.right(X.mT) - X.mT @ Sigma).norm() / (X.mT @ Sigma).norm() < 1e-5


def test_curvature_factor_stays_dense_without_a_flat_block():
    n = 12
    Sigma, lam, Q = _projected_factor(n, 8, SpectrumSpec(power=0.5))
    fac = admm_nq.CurvatureFactor(Sigma, lam, Q, SpectrumSpec(power=0.5))
    assert not fac.structured
    X = torch.randn(n, 3)
    assert torch.equal(fac.left(X), Sigma @ X)
    # ranks covering the whole spectrum: nothing to flatten, dense path
    full = SpectrumSpec(spike_rank=6, dip_rank=6)
    assert not admm_nq.CurvatureFactor(Sigma, lam, Q, full).structured
    # opt-out
    spec = SpectrumSpec(spike_rank=2, dip_rank=1)
    Sigma2, lam2, Q2 = _projected_factor(n, 9, spec)
    assert not admm_nq.CurvatureFactor(Sigma2, lam2, Q2, spec, structured=False).structured


def test_factorize_structured_path_matches_dense_path():
    torch.manual_seed(21)
    n_out, n_in = 24, 16
    spec = SpectrumSpec(spike_rank=2, dip_rank=2, flat_mean="gm")
    o_norm, i_norm = torch.rand(n_out) + 0.5, torch.rand(n_in) + 0.5
    o_cov = _scaled_cov(_spd(n_out, 0.3), o_norm)
    i_cov = _scaled_cov(_spd(n_in, 0.3), i_norm)
    W = torch.randn(n_out, n_in)
    outs = []
    for structured in (True, False):
        torch.manual_seed(3)
        outs.append(admm_nq.factorize_admm_nanoquant(W, i_norm, o_norm, mid_rank=RANK, outer_iters=3,
                                                     rho_scheduler="linear", i_cov=i_cov, o_cov=o_cov,
                                                     spectrum=spec, structured=structured))
    for key in ("W_final", "A_latent", "B_latent"):
        assert torch.allclose(outs[0][key], outs[1][key], atol=1e-3, rtol=1e-3), key
    # transposed layer threads the flag too
    torch.manual_seed(3)
    t = admm_nq.factorize_admm_nanoquant(W.mT, o_norm, i_norm, mid_rank=RANK, outer_iters=3, rho_scheduler="linear",
                                         is_transpose=True, i_cov=o_cov, o_cov=i_cov, spectrum=spec)
    assert t["W_final"].shape == W.mT.shape


# --------------------------------------------------------------------------------------
# Two-device ADMM: the A- and B-updates of one iteration are independent (Jacobi) and may run on separate devices
# --------------------------------------------------------------------------------------
def test_rank1_approx_accepts_start_vector():
    W = torch.randn(12, 7)
    torch.manual_seed(0)
    ref = admm_nq.rank1_approx(W)
    torch.manual_seed(0)
    v = torch.randn(W.shape[1])
    got = admm_nq.rank1_approx(W, v0=v)
    assert torch.equal(ref, got)
    # with a start vector the result does not depend on the global RNG
    torch.manual_seed(123)
    assert torch.equal(admm_nq.rank1_approx(W, v0=v), got)


def _split_case(case):
    torch.manual_seed(11)
    W = torch.randn(24, 16)
    i_norm = torch.rand(16) + 0.5
    o_norm = torch.rand(24) + 0.5
    kw = {}
    if case in ("maha", "transpose", "mid_scale", "tempered"):
        kw["i_cov"] = _scaled_cov(_spd(16, 0.3), i_norm)
        kw["o_cov"] = _scaled_cov(_spd(24, 0.2), o_norm)
    if case == "transpose":
        W, i_norm, o_norm = W.mT.contiguous(), o_norm, i_norm
        kw["i_cov"], kw["o_cov"] = kw["o_cov"], kw["i_cov"]
        kw["is_transpose"] = True
    if case == "mid_scale":
        kw["mid_scale"] = True
    if case == "tempered":
        kw["spectrum"] = SpectrumSpec(power=0.5)
    return W, i_norm, o_norm, kw


@pytest.mark.parametrize("case", ["euclid", "maha", "transpose", "mid_scale", "tempered"])
def test_split_sides_match_serial(case):
    """Running the B side on ``side_device`` reproduces the serial result bit for bit (same op order per side, same
    RNG consumption)."""
    W, i_norm, o_norm, kw = _split_case(case)
    ref = _run(W, i_norm, o_norm, seed=5, **kw)
    got = _run(W, i_norm, o_norm, seed=5, side_device="cpu", **kw)
    assert set(ref) == set(got)
    for k in ref:
        assert torch.equal(ref[k], got[k]), k
        assert got[k].device == ref[k].device


def test_split_sides_with_print_steps_runs(capsys):
    W, i_norm, o_norm, kw = _split_case("maha")
    torch.manual_seed(5)
    out = admm_nq.factorize_admm_nanoquant(W, i_norm, o_norm, mid_rank=RANK, outer_iters=30, print_admm_steps=True,
                                           rho_scheduler="linear", side_device="cpu", **kw)
    assert out["W_final"].shape == W.shape
    assert "[ADMM Step" in capsys.readouterr().out


def test_generator_reproduces_set_seed_stream():
    """A fresh ``torch.Generator`` seeded like the global RNG yields the same factorisation as ``manual_seed``."""
    W, i_norm, o_norm, kw = _split_case("maha")
    ref = _run(W, i_norm, o_norm, seed=3, **kw)
    torch.manual_seed(999)  # the generator path must ignore the global state
    got = admm_nq.factorize_admm_nanoquant(W, i_norm, o_norm, mid_rank=RANK, outer_iters=30, print_admm_steps=False,
                                           rho_scheduler="linear", generator=torch.Generator().manual_seed(3), **kw)
    for k in ref:
        assert torch.equal(ref[k], got[k]), k


def test_curvature_factor_to_device_copies_everything():
    lam = torch.linspace(1.0, 4.0, 6)
    Q, _ = torch.linalg.qr(torch.randn(6, 6))
    Sigma = (Q * lam) @ Q.mT
    f = admm_nq.CurvatureFactor(Sigma, lam, Q, SpectrumSpec(spike_rank=1, dip_rank=1, flat_mean="gm"))
    g = f.to("cpu")
    assert g is not f and g.structured == f.structured
    X = torch.randn(6, 3)
    assert torch.equal(f.left(X), g.left(X)) and torch.equal(f.right(X.mT), g.right(X.mT))
    assert torch.equal(f.lam, g.lam) and f.lam.data_ptr() != g.lam.data_ptr()
