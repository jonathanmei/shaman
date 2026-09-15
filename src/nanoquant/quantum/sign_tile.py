"""Turn a tile of stored sign bits into an Ising model (the torch side of the proof of concept).

For an ADMM result in deployed form ``W_hat = diag(p) U Vs`` (``U`` the ``+-1`` bits of the first factor,
``Vs`` the scaled second factor), freeze every bit except the tile ``T = {(i_t, k_t)}`` and write the
curvature-weighted layer error ``tr(L (W - W_hat) R (W - W_hat)^T)`` as a function of the tile spins
``s_t = U[i_t, k_t]``. With ``E0 = W - diag(p) U_fixed Vs`` (tile entries zeroed), ``b_k = Vs[k]`` and the
inner product ``<X, Y> = tr(L X R Y^T)``:

``J(s) = <E0, E0> - 2 sum_t s_t <E0, D_t> + sum_{t, t'} s_t s_t' <D_t, D_t'>``, ``D_t = p_{i_t} e_{i_t} b_{k_t}^T``,

``<E0, D_t> = p_{i_t} (L E0 R Vs^T)[i_t, k_t]`` and ``<D_t, D_t'> = p_{i_t} p_{i_t'} L[i_t, i_t'] (Vs R Vs^T)[k_t, k_t']``.

In the Ising convention of :mod:`nanoquant.quantum.ising` (``E = const + h . s + sum_{i<j} J_ij s_i s_j``) this is
``h_t = -2 <E0, D_t>``, ``J_tt' = 2 <D_t, D_t'>`` and ``const = <E0, E0> + sum_t <D_t, D_t>``.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel

from ..core.rank_probe import _sign
from .ising import IsingInstance


@torch.no_grad()
def deployed_factors(result: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three factors of the deployed form ``W_hat = diag(post) U Vs`` of an ADMM ``result``.

    Mirrors :func:`nanoquant.core.rank_probe.deployed_matrix` (kept there untouched because that file is part of the
    cache fingerprints, see :mod:`nanoquant.quantum.layer_admm`).

    Parameters
    ----------
    result : dict
        Output of :func:`nanoquant.core.admm_nq.factorize_admm_nanoquant` (``A`` is ``(rank, out)``, ``B`` is
        ``(rank, in)``; ``scale_mid`` optional).

    Returns
    -------
    tuple of torch.Tensor
        ``U = sign(A)^T`` ``(out, rank)`` of ``+-1``; ``Vs = [diag(scale_mid)] sign(B) diag(scale_pre)``
        ``(rank, in)``; ``post = scale_post`` ``(out,)``. All fp32.
    """
    U = _sign(result["A"].float()).mT
    V = _sign(result["B"].float())
    Vs = V * result["scale_pre"].float().reshape(1, -1)
    mid = result.get("scale_mid")
    if mid is not None:
        Vs = Vs * mid.float().reshape(-1, 1)
    return U, Vs, result["scale_post"].float().reshape(-1)


class Tile(BaseModel):
    """A set of bits of the first factor ``U`` ``(out, rank)``.

    Parameters
    ----------
    entries : list of (int, int)
        ``(i, k)`` = (output row, mid column) of each bit in ``U``.
    signs : list of int
        The ADMM bits ``U[i, k]``.
    confidence : list of float
        Warm-start confidence in ``[0, 1]`` per bit: ``min(|A_latent| / median |A_latent|, 1)``.
    layer : str
        Provenance (``"<block>.<name>"``).
    """

    entries: list[tuple[int, int]]
    signs: list[int]
    confidence: list[float]
    layer: str = ""

    @property
    def n(self) -> int:
        """Number of bits."""
        return len(self.entries)


def _left(L: torch.Tensor | None, X: torch.Tensor) -> torch.Tensor:
    if L is None:
        return X
    return L @ X if L.dim() == 2 else L.unsqueeze(1) * X


def _right(X: torch.Tensor, R: torch.Tensor | None) -> torch.Tensor:
    if R is None:
        return X
    return X @ R if R.dim() == 2 else X * R.unsqueeze(0)


def _symmetrized(F: torch.Tensor) -> torch.Tensor:
    """``(F + F^T) / 2`` for a dense factor; a diagonal (vector) factor is returned as is."""
    return 0.5 * (F + F.mT) if F.dim() == 2 else F


def _sub(L: torch.Tensor | None, rows: torch.Tensor) -> torch.Tensor:
    """``L[rows][:, rows]`` for a dense, diagonal (vector) or identity (``None``) factor.

    ``rows`` may repeat (two tile bits in the same output row): the entry ``(t, t')`` is then ``L[i, i]``, which a
    diagonal factor must also produce, hence the explicit equality mask.
    """
    same = (rows.unsqueeze(1) == rows.unsqueeze(0)).to(torch.float64)
    if L is None:
        return same
    if L.dim() == 2:
        return L[rows][:, rows]
    return same * L[rows].unsqueeze(1)


@torch.no_grad()
def select_tile(result: dict, n: int, layer: str = "") -> Tile:
    """The ``n`` least-confident bits of the first factor: smallest ``|A_latent|``.

    Parameters
    ----------
    result : dict
        ADMM result (``A`` and ``A_latent`` are ``(rank, out)``).
    n : int
        Tile size.
    layer : str
        Provenance string stored on the tile.
    """
    lat = result["A_latent"].detach().float().cpu()  # (rank, out)
    out = lat.shape[1]
    if not 0 < n <= lat.numel():
        raise ValueError(f"tile size {n} must be in 1..{lat.numel()}")
    mag = lat.abs().reshape(-1)
    idx = torch.topk(mag, n, largest=False).indices
    k, i = (idx // out).tolist(), (idx % out).tolist()
    signs = _sign(result["A"].detach().float().cpu())
    median = mag.median().clamp_min(1e-30)
    conf = (mag[idx] / median).clamp(max=1.0).tolist()
    return Tile(entries=list(zip(i, k)), signs=[int(signs[kk, ii].item()) for ii, kk in zip(i, k)],
                confidence=conf, layer=layer)


@torch.no_grad()
def tile_ising(W: torch.Tensor, L: torch.Tensor | None, R: torch.Tensor | None, result: dict,
               tile: Tile) -> IsingInstance:
    """Ising model of the curvature-weighted layer error over the tile's bits (everything else frozen).

    Parameters
    ----------
    W : torch.Tensor
        Target weight ``(out, in)``.
    L, R : torch.Tensor or None
        Output / input curvature factors, dense ``(out, out)`` / ``(in, in)`` or diagonal vectors (``None`` =
        identity), on the raw statistics scale as in :func:`nanoquant.core.compress_block.mahalanobis_weight_error`.
    result : dict
        ADMM result of ``W``.
    tile : Tile
        Bits to free.

    Returns
    -------
    IsingInstance
        ``energy(s)`` equals ``mahalanobis_weight_error(W, W_hat(s), L, R)`` for every tile assignment ``s``.
    """
    U, Vs, post = (t.double().cpu() for t in deployed_factors(result))
    W = W.detach().double().cpu()
    # the metric is defined for symmetric factors; fp32 statistics are symmetric only to rounding, and the quadratic
    # expansion below is exact only for the symmetrised matrices
    L = None if L is None else _symmetrized(L.detach().double().cpu())
    R = None if R is None else _symmetrized(R.detach().double().cpu())
    rows = torch.tensor([i for i, _ in tile.entries])
    cols = torch.tensor([k for _, k in tile.entries])
    U0 = U.clone()
    U0[rows, cols] = 0.0
    W0 = (U0 @ Vs) * post.unsqueeze(1)
    E0 = W - W0
    LE0 = _left(L, E0)
    M = _right(LE0, R) @ Vs.mT  # (out, rank) = L E0 R Vs^T
    G = _right(Vs, R) @ Vs.mT  # (rank, rank) = Vs R Vs^T
    p = post[rows]
    lin = p * M[rows, cols]  # <E0, D_t>
    quad = torch.outer(p, p) * _sub(L, rows) * G[cols][:, cols]  # <D_t, D_t'>
    e0 = float((LE0 * _right(E0, R)).sum().item())  # <E0, E0> in float64 (mahalanobis_weight_error is fp32)
    const = e0 + float(quad.diagonal().sum().item())
    J = 2.0 * quad
    J.fill_diagonal_(0.0)
    J = 0.5 * (J + J.mT)
    meta = {"layer": tile.layer, "entries": [list(e) for e in tile.entries], "admm_signs": list(tile.signs),
            "confidence": list(tile.confidence), "shape": list(W.shape), "rank": int(U.shape[1])}
    return IsingInstance(h=(-2.0 * lin).tolist(), J=J.tolist(), const=const, meta=meta)


@torch.no_grad()
def apply_tile(result: dict, tile: Tile, signs: list[int]) -> dict:
    """Copy of ``result`` with the tile's bits of ``A`` set to ``signs`` (magnitudes kept; ``deployed_matrix`` only
    reads the sign)."""
    if len(signs) != tile.n:
        raise ValueError(f"expected {tile.n} signs, got {len(signs)}")
    out = dict(result)
    A = result["A"].detach().clone()
    for (i, k), s in zip(tile.entries, signs):
        mag = A[k, i].abs().clamp_min(torch.finfo(A.dtype).tiny)
        A[k, i] = mag if s > 0 else -mag
    out["A"] = A
    return out
