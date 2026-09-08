# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Spectral utilities shared by the ADMM data term and the block reconstruction loss.

Dense curvature factors estimated from a few hundred thousand tokens are strongly concentrated (a handful of
directions carry most of the trace). Both consumers can temper that spectrum: raise the eigenvalues to a power
below one and/or floor them so the condition number is bounded, always preserving the trace so that the loss
scale, learning rates and ADMM's ridge keep their meaning.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def spike_flat_eigenvalues(lam: torch.Tensor, rank: int) -> torch.Tensor:
    """Project a spectrum onto the spike-plus-flat family: top-``rank`` eigenvalues kept, tail set to its mean.

    For a fixed eigenbasis this is the M-projection of the KL divergence between zero-mean Gaussians onto
    ``U S U^T + mu P_perp`` (Pro-KLShampoo, arXiv 2605.06316): the optimal flat value is the arithmetic mean of the
    tail eigenvalues, so the trace is preserved, and the approximation gap is proportional to
    ``log(AM / GM)`` of the tail, zero when the tail is already flat.

    Parameters
    ----------
    lam : torch.Tensor
        Eigenvalues in any order.
    rank : int
        Number of spikes to keep; ``<= 0`` or ``>= len(lam)`` returns ``lam`` unchanged.
    """
    n = lam.numel()
    if rank <= 0 or rank >= n:
        return lam
    order = torch.argsort(lam, descending=True)
    out = lam.clone()
    tail = order[rank:]
    out[tail] = lam[tail].mean()
    return out


@torch.no_grad()
def temper_eigenvalues(lam: torch.Tensor, power: float = 1.0, cond_max: float = 0.0,
                       spike_rank: int = 0) -> torch.Tensor:
    """Temper a spectrum in place of the matrix: spike-flat projection, power, condition floor; trace preserved.

    Parameters
    ----------
    lam : torch.Tensor
        Eigenvalues (any order), clamped at zero internally.
    power : float
        Exponent applied to the eigenvalues (``0.5`` halves the log-condition number; ``1`` = off).
    cond_max : float
        If ``> 0``, floor the eigenvalues at ``max(lam) / cond_max`` (lifts the ignored directions instead of
        truncating the dominant ones).
    spike_rank : int
        If ``> 0``, first project onto the spike-plus-flat family with this many spikes
        (:func:`spike_flat_eigenvalues`); the projection is an estimate-level operation and therefore precedes the
        power and the floor, which act on how the estimate is used.

    Returns
    -------
    torch.Tensor
        Tempered eigenvalues with the same sum as the input (same dtype).
    """
    lam0 = lam.clamp_min(0)
    total = lam0.sum()
    out = spike_flat_eigenvalues(lam0, spike_rank)
    if power != 1.0:
        out = out.pow(power)
    if cond_max > 0:
        out = out.clamp_min(out.max() / cond_max)
    return out * (total / out.sum().clamp_min(torch.finfo(out.dtype).tiny))


@torch.no_grad()
def spectrum_summary(M: torch.Tensor) -> dict:
    """Conditioning summary of a symmetric PSD matrix.

    Returns
    -------
    dict
        ``lam_max_over_mean_diag``, ``max_diag_over_mean_diag``, ``cond`` (``lam_max / lam_min``),
        ``eff_rank`` (``tr(M)^2 / tr(M^2)``) and ``top50_share`` (trace share of the 50 largest eigenvalues).
    """
    M = 0.5 * (M.double() + M.double().mT)
    lam = torch.linalg.eigvalsh(M).clamp_min(0)
    mean_diag = M.diagonal().mean().clamp_min(1e-30)
    tr = lam.sum().clamp_min(1e-30)
    k = min(50, lam.numel())
    return {
        "lam_max_over_mean_diag": (lam[-1] / mean_diag).item(),
        "max_diag_over_mean_diag": (M.diagonal().max() / mean_diag).item(),
        "cond": (lam[-1] / lam[0].clamp_min(1e-30)).item(),
        "eff_rank": (tr.square() / lam.square().sum().clamp_min(1e-30)).item(),
        "top50_share": (lam[-k:].sum() / tr).item(),
    }


def format_spectrum(summary: dict, title: str = "block curvature") -> str:
    """One-line rendering of :func:`spectrum_summary`."""
    return (f"[{title}] lam_max/mean_diag {summary['lam_max_over_mean_diag']:.1f} | "
            f"max_diag/mean_diag {summary['max_diag_over_mean_diag']:.1f} | cond {summary['cond']:.3g} | "
            f"eff_rank {summary['eff_rank']:.1f} | top-50 share {summary['top50_share']:.3f}")


@torch.no_grad()
def condition_curvature(cov: torch.Tensor, cond_max: float = 0.0, power: float = 1.0,
                        mix: float = 1.0, spike_rank: int = 0) -> torch.Tensor:
    """Regularise the spectrum of a dense curvature matrix, preserving its trace.

    Parameters
    ----------
    cov : torch.Tensor
        Symmetric PSD matrix ``(d, d)``.
    cond_max, power, spike_rank : float, float, int
        See :func:`temper_eigenvalues`.
    mix : float
        Return ``(1 - mix) * diag(cov) + mix * conditioned`` (``1`` = fully dense, ``0`` = diagonal).

    Returns
    -------
    torch.Tensor
        Conditioned matrix in fp32 with the same trace as ``cov``.
    """
    cov64 = cov.detach().double()
    cov64 = 0.5 * (cov64 + cov64.mT)
    diag = cov64.diagonal().clone()
    out = cov64
    if power != 1.0 or cond_max > 0 or spike_rank > 0:
        lam, Q = torch.linalg.eigh(cov64)
        out = (Q * temper_eigenvalues(lam, power=power, cond_max=cond_max, spike_rank=spike_rank)) @ Q.mT
    if mix != 1.0:
        out = (1.0 - mix) * torch.diag(diag) + mix * out
    return (0.5 * (out + out.mT)).float()
