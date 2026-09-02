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
