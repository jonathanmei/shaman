"""Tests for the sign-tile Ising construction against the curvature-weighted layer error (torch)."""

import pytest
import torch

from nanoquant.core import admm_nq
from nanoquant.core.compress_block import mahalanobis_weight_error
from nanoquant.core.rank_probe import _sign, deployed_matrix
from nanoquant.quantum import sign_tile as T
from nanoquant.quantum.sign_tile import deployed_factors

RANK = 8


def _spd(n: int, corr: float) -> torch.Tensor:
    return (1 - corr) * torch.eye(n) + corr * torch.ones(n, n)


def _scaled_cov(corr_mat: torch.Tensor, norm_vec: torch.Tensor) -> torch.Tensor:
    s = norm_vec.sqrt()
    return corr_mat * s.unsqueeze(1) * s.unsqueeze(0)


def _problem(seed: int = 0, n_out: int = 32, n_in: int = 32, corr: float = 0.5, dense: bool = True):
    """``(W, i_norm, o_norm, i_cov, o_cov, L, R)`` with ``L``/``R`` the factors ``mahalanobis_weight_error`` sees."""
    torch.manual_seed(seed)
    W = torch.randn(n_out, n_in)
    i_norm = torch.rand(n_in) + 0.5
    o_norm = torch.rand(n_out) + 0.5
    if dense:
        o_cov = _scaled_cov(_spd(n_out, corr), o_norm)
        i_cov = _scaled_cov(_spd(n_in, corr), i_norm)
        return W, i_norm, o_norm, i_cov, o_cov, o_cov, i_cov
    return W, i_norm, o_norm, None, None, o_norm, i_norm


def _admm(W, i_norm, o_norm, i_cov, o_cov, mid_scale: bool, seed: int = 1):
    torch.manual_seed(seed)
    return admm_nq.factorize_admm_nanoquant(W, i_norm, o_norm, mid_rank=RANK, outer_iters=30, print_admm_steps=False,
                                            rho_scheduler="linear", i_cov=i_cov, o_cov=o_cov, mid_scale=mid_scale)


def _error64(W, result, L, R) -> float:
    """``tr(L (W - W_hat) R (W - W_hat)^T)`` in float64 (``mahalanobis_weight_error`` accumulates in fp32)."""
    U, Vs, post = (t.double() for t in deployed_factors(result))
    E = W.double() - (U @ Vs) * post.unsqueeze(1)
    LE = L.double() @ E if L.dim() == 2 else L.double().unsqueeze(1) * E
    ER = E @ R.double() if R.dim() == 2 else E * R.double().unsqueeze(0)
    return float((LE * ER).sum().item())


@pytest.mark.parametrize("mid_scale", [False, True])
@pytest.mark.parametrize("dense", [True, False])
def test_tile_energy_equals_curvature_weighted_error(mid_scale, dense):
    W, i_norm, o_norm, i_cov, o_cov, L, R = _problem(dense=dense)
    res = _admm(W, i_norm, o_norm, i_cov, o_cov, mid_scale)
    tile = T.select_tile(res, n=9, layer="t.layer")
    inst = T.tile_ising(W, L, R, res, tile)
    assert inst.n == 9 and inst.meta["layer"] == "t.layer"
    g = torch.Generator().manual_seed(3)
    for _ in range(12):
        s = [int(v) for v in (torch.randint(0, 2, (9,), generator=g) * 2 - 1).tolist()]
        polished = T.apply_tile(res, tile, s)
        assert inst.energy(s) == pytest.approx(_error64(W, polished, L, R), rel=1e-9)
        # and agrees with the pipeline's fp32 metric up to its accumulation error
        assert inst.energy(s) == pytest.approx(mahalanobis_weight_error(W, deployed_matrix(polished), L, R), rel=1e-3)
    # the ADMM bits reproduce the unmodified deployed matrix
    assert inst.energy(tile.signs) == pytest.approx(_error64(W, res, L, R), rel=1e-9)


def test_tile_optimum_is_never_worse_than_the_admm_bits():
    W, i_norm, o_norm, i_cov, o_cov, L, R = _problem(seed=4, corr=0.7)
    res = _admm(W, i_norm, o_norm, i_cov, o_cov, mid_scale=False)
    tile = T.select_tile(res, n=10)
    inst = T.tile_ising(W, L, R, res, tile)
    best_s, best_e = inst.brute_force()
    assert best_e <= inst.energy(tile.signs) + 1e-9
    assert _error64(W, T.apply_tile(res, tile, best_s), L, R) == pytest.approx(best_e, rel=1e-9)


def test_diagonal_curvature_single_column_tile_has_no_couplings():
    W, i_norm, o_norm, _, _, L, R = _problem(dense=False)
    res = _admm(W, i_norm, o_norm, None, None, mid_scale=False)
    rows = [0, 3, 7, 11, 19]
    signs = _sign(res["A"].float())
    tile = T.Tile(entries=[(i, 2) for i in rows], signs=[int(signs[2, i]) for i in rows], confidence=[1.0] * 5)
    inst = T.tile_ising(W, L, R, res, tile)
    assert torch.allclose(torch.tensor(inst.J, dtype=torch.float64), torch.zeros(5, 5, dtype=torch.float64))
    # separable: the optimum is the sign of the field term
    best_s, _ = inst.brute_force()
    assert best_s == [1 if h < 0 else -1 for h in inst.h]


def test_dense_curvature_couples_the_tile():
    W, i_norm, o_norm, i_cov, o_cov, L, R = _problem(corr=0.6)
    res = _admm(W, i_norm, o_norm, i_cov, o_cov, mid_scale=False)
    tile = T.select_tile(res, n=6)
    inst = T.tile_ising(W, L, R, res, tile)
    J = torch.tensor(inst.J, dtype=torch.float64)
    assert (J.abs() > 1e-8).sum() > 0


def test_select_tile_picks_the_smallest_latent_magnitudes():
    W, i_norm, o_norm, i_cov, o_cov, _, _ = _problem()
    res = _admm(W, i_norm, o_norm, i_cov, o_cov, mid_scale=False)
    tile = T.select_tile(res, n=7)
    lat = res["A_latent"].abs()  # (rank, out)
    picked = torch.tensor([lat[k, i] for i, k in tile.entries])
    threshold = torch.topk(lat.reshape(-1), 7, largest=False).values.max()
    assert torch.all(picked <= threshold + 1e-12)
    assert len(set(tile.entries)) == 7
    assert all(0.0 <= c <= 1.0 for c in tile.confidence)
    signs = _sign(res["A"].float())
    assert tile.signs == [int(signs[k, i]) for i, k in tile.entries]


def test_apply_tile_changes_only_the_tile_rows():
    W, i_norm, o_norm, i_cov, o_cov, _, _ = _problem()
    res = _admm(W, i_norm, o_norm, i_cov, o_cov, mid_scale=False)
    tile = T.select_tile(res, n=4)
    flipped = T.apply_tile(res, tile, [-s for s in tile.signs])
    before, after = deployed_matrix(res), deployed_matrix(flipped)
    changed_rows = (before != after).any(dim=1).nonzero().flatten().tolist()
    assert set(changed_rows) <= {i for i, _ in tile.entries}
    assert res["A"].data_ptr() != flipped["A"].data_ptr()
    with pytest.raises(ValueError):
        T.apply_tile(res, tile, [1])
