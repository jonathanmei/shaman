"""Tests for the Mahalanobis (Kronecker-curvature) ADMM path and the middle-scale export in admm_nq."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nanoquant.core import admm_nq
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


# --------------------------------------------------------------------------------------
# Inexact Sylvester solve: stale eigenbasis preconditioner with Rayleigh / QR / PCG rungs
# --------------------------------------------------------------------------------------
def _sylvester_problem(seed, n=10, k=5, spectrum=None):
    """Random SPD ``Sigma`` (with its eigendecomposition), SPD Gram matrix ``M`` and right-hand side ``C``."""
    torch.manual_seed(seed)
    A = torch.randn(n, n)
    Sigma = A @ A.mT / n + 0.1 * torch.eye(n)
    lam, Q = torch.linalg.eigh(Sigma)
    if spectrum is None:
        B = torch.randn(k, k)
        M = B @ B.mT + 0.1 * torch.eye(k)
    else:
        Qm, _ = torch.linalg.qr(torch.randn(k, k))
        M = (Qm * spectrum) @ Qm.mT
    C = torch.randn(n, k)
    return Sigma, lam, Q, M, C


def _residual(Sigma, lam, M, C, F, rho, reg):
    sigma = admm_nq._sylvester_stabilizer(lam, M, rho, reg)
    return (Sigma @ F @ M + sigma * F - C).norm().item()


def _rotate(M, angle, i, j):
    """Rotate the eigenvectors ``i`` and ``j`` of ``M`` by ``angle`` (keeps the spectrum)."""
    mu, Qm = torch.linalg.eigh(M)
    G = torch.eye(M.shape[0])
    c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    G[i, i], G[j, j], G[i, j], G[j, i] = c, c, -s, s
    Qr = Qm @ G
    return (Qr * mu) @ Qr.mT


def test_sylvester_tol_zero_is_legacy():
    Sigma, lam, Q, M, C = _sylvester_problem(20)
    ref = admm_nq._sylvester_solve_step(Q, lam, M, C, 0.5, 1e-2)
    state = admm_nq.SylvesterState()
    got = admm_nq._sylvester_solve_step(Q, lam, M, C, 0.5, 1e-2, Sigma=Sigma, state=state, tol=0.0)
    assert torch.equal(got, ref)
    assert state.n_eigh == 1 and state.Q_M is not None and state.mu is not None


def test_sylvester_pcg_matches_exact_with_stale_basis():
    Sigma, lam, Q, M, C = _sylvester_problem(21)
    rho, reg = 0.5, 1e-2
    state = admm_nq.SylvesterState()
    admm_nq._sylvester_solve_step(Q, lam, M, C, rho, reg, Sigma=Sigma, state=state, tol=1e-6)  # fills the state
    torch.manual_seed(22)
    P = torch.randn(M.shape[0], M.shape[0])
    M2 = M + 0.05 * (P @ P.mT) / M.shape[0]
    exact = admm_nq._sylvester_solve_step(Q, lam, M2, C, rho, reg)
    got = admm_nq._sylvester_solve_step(Q, lam, M2, C, rho, reg, Sigma=Sigma, state=state, tol=1e-5, qr_steps=1,
                                        max_pcg=50)
    assert state.n_eigh == 1, "the stale basis plus corrections should have sufficed"
    assert torch.allclose(got, exact, atol=1e-3, rtol=1e-3)
    assert _residual(Sigma, lam, M2, C, got, rho, reg) <= 1e-4 * C.norm().item()


def test_sylvester_rayleigh_and_qr_rungs_reduce_residual():
    spectrum = torch.tensor([100.0, 30.0, 1.0, 1.05, 1.1, 0.95, 1.0])  # two spikes and a cluster
    Sigma, lam, Q, M, C = _sylvester_problem(23, n=12, k=7, spectrum=spectrum)
    rho, reg = 0.3, 1e-2
    mu0, Q0 = torch.linalg.eigh(M)
    mu0 = mu0.clamp(min=0)

    # (a) pure eigenvalue drift: fresh Rayleigh quotients in the stale basis recover the exact solve
    M_drift = (Q0 * (mu0 * torch.tensor([1.2, 0.8, 1.1, 0.9, 1.0, 1.05, 0.95]))) @ Q0.mT
    sigma = admm_nq._sylvester_stabilizer(lam, M_drift, rho, reg)
    r_stale = _residual(Sigma, lam, M_drift, C, admm_nq._precond_solve(Q, lam, Q0, mu0, sigma, C), rho, reg)
    _, mu_ray = admm_nq._rayleigh_refresh(M_drift, Q0)
    r_ray = _residual(Sigma, lam, M_drift, C, admm_nq._precond_solve(Q, lam, Q0, mu_ray, sigma, C), rho, reg)
    assert r_ray < 1e-3 * r_stale

    # (b) the two spikes rotate into each other: Rayleigh alone cannot fix it, one QR step nearly does
    M_rot = _rotate(M, 0.3, 6, 5)  # eigh sorts ascending: indices 6, 5 are the 100 and 30 spikes
    sigma = admm_nq._sylvester_stabilizer(lam, M_rot, rho, reg)
    Z, mu_ray = admm_nq._rayleigh_refresh(M_rot, Q0)
    r_ray = _residual(Sigma, lam, M_rot, C, admm_nq._precond_solve(Q, lam, Q0, mu_ray, sigma, C), rho, reg)
    Q1, _, mu_qr = admm_nq._orthogonal_iteration_step(M_rot, Z, mu_ray)
    r_qr = _residual(Sigma, lam, M_rot, C, admm_nq._precond_solve(Q, lam, Q1, mu_qr, sigma, C), rho, reg)
    assert r_qr < 0.5 * r_ray
    assert torch.allclose(Q1.mT @ Q1, torch.eye(7), atol=1e-5)


def test_sylvester_qr_rung_avoids_eigh_when_it_suffices():
    spectrum = torch.tensor([100.0, 30.0, 1.0, 1.05, 1.1, 0.95, 1.0])
    Sigma, lam, Q, M, C = _sylvester_problem(24, n=12, k=7, spectrum=spectrum)
    rho, reg = 0.3, 1e-2
    state = admm_nq.SylvesterState()
    admm_nq._sylvester_solve_step(Q, lam, M, C, rho, reg, Sigma=Sigma, state=state, tol=1e-6)
    M_rot = _rotate(M, 0.3, 6, 5)
    sigma = admm_nq._sylvester_stabilizer(lam, M_rot, rho, reg)
    Z, mu_ray = admm_nq._rayleigh_refresh(M_rot, state.Q_M)
    r_ray = _residual(Sigma, lam, M_rot, C, admm_nq._precond_solve(Q, lam, state.Q_M, mu_ray, sigma, C), rho, reg)
    Q1, _, mu_qr = admm_nq._orthogonal_iteration_step(M_rot, Z, mu_ray)
    r_qr = _residual(Sigma, lam, M_rot, C, admm_nq._precond_solve(Q, lam, Q1, mu_qr, sigma, C), rho, reg)
    assert r_qr < r_ray
    tol = 0.5 * (r_ray + r_qr) / C.norm().item()

    with_qr = admm_nq.SylvesterState(mu=state.mu, Q_M=state.Q_M, n_eigh=state.n_eigh)
    admm_nq._sylvester_solve_step(Q, lam, M_rot, C, rho, reg, Sigma=Sigma, state=with_qr, tol=tol, qr_steps=1,
                                  max_pcg=0)
    assert with_qr.n_eigh == 1 and with_qr.n_qr == 1

    without = admm_nq.SylvesterState(mu=state.mu, Q_M=state.Q_M, n_eigh=state.n_eigh)
    admm_nq._sylvester_solve_step(Q, lam, M_rot, C, rho, reg, Sigma=Sigma, state=without, tol=tol, qr_steps=0,
                                  max_pcg=0)
    assert without.n_eigh == 2 and without.n_qr == 0


def test_sylvester_pcg_refreshes_when_basis_is_far():
    Sigma, lam, Q, M, C = _sylvester_problem(25)
    rho, reg = 0.5, 1e-2
    state = admm_nq.SylvesterState()
    admm_nq._sylvester_solve_step(Q, lam, M, C, rho, reg, Sigma=Sigma, state=state, tol=1e-6)
    torch.manual_seed(26)
    B = torch.randn(M.shape[0], M.shape[0])
    M_far = 3.0 * B @ B.mT + torch.eye(M.shape[0])
    exact = admm_nq._sylvester_solve_step(Q, lam, M_far, C, rho, reg)
    got = admm_nq._sylvester_solve_step(Q, lam, M_far, C, rho, reg, Sigma=Sigma, state=state, tol=1e-6, qr_steps=1,
                                        max_pcg=2)
    assert state.n_eigh == 2 and state.n_pcg == 2
    assert torch.allclose(got, exact, atol=1e-5, rtol=1e-5)


def test_mahalanobis_admm_with_inexact_sylvester_runs_and_reports(capsys):
    torch.manual_seed(27)
    n_out, n_in = 32, 24
    W = torch.randn(n_out, n_in)
    i_norm = torch.rand(n_in) + 0.5
    o_norm = torch.rand(n_out) + 0.5
    o_cov = _scaled_cov(_spd(n_out, 0.5), o_norm)
    i_cov = _scaled_cov(_spd(n_in, 0.5), i_norm)
    exact = _run(W, i_norm, o_norm, seed=13, i_cov=i_cov, o_cov=o_cov)
    inexact = _run(W, i_norm, o_norm, seed=13, i_cov=i_cov, o_cov=o_cov, sylvester_tol=1e-2, sylvester_qr_steps=1,
                   sylvester_max_pcg=3)
    out = capsys.readouterr().out
    assert "[ADMM sylvester]" in out and "eigh" in out
    err_exact = _mahalanobis_error(W, exact["W_final"], i_norm, o_norm, i_cov, o_cov)
    err_inexact = _mahalanobis_error(W, inexact["W_final"], i_norm, o_norm, i_cov, o_cov)
    assert err_inexact < 1.1 * err_exact


# --------------------------------------------------------------------------------------
# Early stopping on a frozen Z
# --------------------------------------------------------------------------------------
def test_early_stop_off_by_default_and_reproduces_legacy(capsys):
    torch.manual_seed(28)
    W = torch.randn(24, 16)
    i_norm = torch.rand(16) + 0.5
    o_norm = torch.rand(24) + 0.5
    ref = _run(W, i_norm, o_norm, seed=7)
    got = _run(W, i_norm, o_norm, seed=7, early_stop_patience=0)
    assert all(torch.equal(ref[k], got[k]) for k in ("W_final", "A", "B"))
    assert "early stop" not in capsys.readouterr().out


def test_early_stop_triggers_and_keeps_the_reconstruction(capsys):
    torch.manual_seed(29)
    W = torch.randn(24, 16)
    i_norm = torch.rand(16) + 0.5
    o_norm = torch.rand(24) + 0.5
    torch.manual_seed(30)
    full = admm_nq.factorize_admm_nanoquant(W, i_norm, o_norm, mid_rank=RANK, outer_iters=200, rho_scheduler="linear")
    torch.manual_seed(30)
    early = admm_nq.factorize_admm_nanoquant(W, i_norm, o_norm, mid_rank=RANK, outer_iters=200, rho_scheduler="linear",
                                             early_stop_patience=5, early_stop_tol=1e-2, early_stop_min_frac=0.25)
    out = capsys.readouterr().out
    assert "[ADMM] early stop at" in out
    stopped_at = int(out.split("[ADMM] early stop at ")[1].split("/")[0])
    assert 50 <= stopped_at < 200  # not before min_frac, not the full schedule
    # a frozen Z is a heuristic (a late sign flip is still possible), so compare the deployed reconstructions
    err_full = (full["W_final"] - W).norm() / W.norm()
    err_early = (early["W_final"] - W).norm() / W.norm()
    assert err_early <= 1.02 * err_full
    # the binary patterns agree almost everywhere
    agree = (full["A"].sign() == early["A"].sign()).float().mean() * (full["B"].sign() == early["B"].sign()).float().mean()
    assert agree > 0.98


def test_new_admm_config_defaults_are_legacy():
    cfg = NanoQuantConfig()
    assert cfg["admm_early_stop_patience"] == 0
    assert cfg["admm_early_stop_tol"] == 1e-4
    assert cfg["admm_early_stop_min_frac"] == 0.5
    assert cfg["admm_sylvester_tol"] == 0.0
    assert cfg["admm_sylvester_qr_steps"] == 1
    assert cfg["admm_sylvester_max_pcg"] == 3


def test_config_dataclass_accepts_every_config_key():
    """main.py builds NanoQuantConfigDataclass(**NanoQuantConfig(...)): every key must be a dataclass field."""
    from nanoquant.modules.hub import NanoQuantConfigDataclass

    cfg = NanoQuantConfig(model_id="t")
    dc = NanoQuantConfigDataclass(**cfg)
    assert dc.to_dict()["admm_sylvester_tol"] == 0.0
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
    admm_nq._normalized_curvature(cov, norm, torch.float64, 1e-12, power=0.5, eig_cache=cache)  # other conditioning
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
