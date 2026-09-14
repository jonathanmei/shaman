# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Spectral conditioning of the dense Kronecker curvature factors used by the ADMM data term.

Dense curvature factors estimated from a few hundred thousand tokens are strongly concentrated (a handful of
directions carry most of the trace). Two trace-preserving operations act on their spectrum:

* **tempering**: raise the eigenvalues to a power below one, so the Mahalanobis data term trusts the dominant
  directions less; the square root (``power = 0.5``) is the regret-optimal metric under curvature uncertainty and the
  Fisher of a robustified loss at one-bit perturbation scale (docs/curvature_tempering_theory.md);
* **two-sided spike-plus-flat projection**: keep the ``spike_rank`` largest and the ``dip_rank`` smallest
  eigenvalues exactly and replace the middle by one shared value. The family "r free eigenvalues + one shared
  eigenvalue" is closed under inversion, so a low-rank-plus-identity model of the factor (Pro-KLShampoo, top
  eigenvalues exact, arithmetic-mean tail = M-projection of the KL divergence) and of its *inverse* (bottom
  eigenvalues exact, harmonic-mean tail = the precision / reverse-KL fit) are the two one-sided corners of the same
  parametrisation; ``flat_mean`` selects the shared value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

FLAT_MEANS = ("am", "gm", "hm")


def _flat_value(x: torch.Tensor, flat_mean: str) -> torch.Tensor:
    """Arithmetic, geometric or harmonic mean of a positive spectrum slice."""
    if flat_mean == "am":
        return x.mean()
    tiny = torch.finfo(x.dtype).tiny
    if flat_mean == "gm":
        return x.clamp_min(tiny).log().mean().exp()
    if flat_mean == "hm":
        return x.numel() / (1.0 / x.clamp_min(tiny)).sum()
    raise ValueError(f"Unknown flat_mean: {flat_mean!r} (choices: {FLAT_MEANS})")


def _check_ranks(spike_rank: int, dip_rank: int, flat_mean: str) -> None:
    if spike_rank < 0 or dip_rank < 0:
        raise ValueError(f"spike_rank / dip_rank must be >= 0, got {spike_rank} / {dip_rank}")
    if flat_mean not in FLAT_MEANS:
        raise ValueError(f"Unknown flat_mean: {flat_mean!r} (choices: {FLAT_MEANS})")


@torch.no_grad()
def _middle_indices(lam: torch.Tensor, spike_rank: int, dip_rank: int) -> torch.Tensor:
    """Indices of the eigenvalues a projection replaces: everything but the ``spike_rank`` largest and ``dip_rank``
    smallest (empty when the two ranks cover the spectrum)."""
    n = lam.numel()
    if spike_rank + dip_rank >= n:
        return lam.new_empty(0, dtype=torch.long)
    order = torch.argsort(lam, descending=True)
    return order[spike_rank:n - dip_rank]


@torch.no_grad()
def project_spectrum(lam: torch.Tensor, spike_rank: int = 0, dip_rank: int = 0,
                     flat_mean: str = "am") -> torch.Tensor:
    """Project a spectrum onto the two-sided spike-plus-flat family.

    The ``spike_rank`` largest and ``dip_rank`` smallest eigenvalues are kept exactly; the rest is replaced by its
    arithmetic (``"am"``), geometric (``"gm"``) or harmonic (``"hm"``) mean. For a fixed eigenbasis, ``"am"`` is the
    M-projection of ``KL(N(0, F) || N(0, F_hat))`` (Pro-KLShampoo, arXiv 2605.06316; preserves the trace) and
    ``"hm"`` the analogous fit of the inverse, ``KL(N(0, F_hat) || N(0, F))``; the approximation gaps are
    ``log(AM/GM)`` and ``log(GM/HM)`` of the replaced eigenvalues (:func:`projection_gaps`).

    Parameters
    ----------
    lam : torch.Tensor
        Eigenvalues in any order.
    spike_rank : int
        Number of largest eigenvalues kept exactly.
    dip_rank : int
        Number of smallest eigenvalues kept exactly (the spikes of the inverse).
    flat_mean : str
        Shared value of the replaced middle, one of :data:`FLAT_MEANS`.

    Returns
    -------
    torch.Tensor
        Projected eigenvalues in the input order; ``lam`` itself when nothing is replaced.
    """
    _check_ranks(spike_rank, dip_rank, flat_mean)
    if spike_rank + dip_rank == 0:
        return lam
    middle = _middle_indices(lam, spike_rank, dip_rank)
    if middle.numel() == 0:
        return lam
    out = lam.clone()
    out[middle] = _flat_value(lam[middle], flat_mean)
    return out


@torch.no_grad()
def projection_gaps(lam: torch.Tensor, spike_rank: int = 0, dip_rank: int = 0) -> dict:
    """Dispersion of the eigenvalues a projection would replace (its KL approximation gaps per matrix dimension).

    With both ranks zero the whole spectrum is summarised, which describes the unprojected factor.

    Returns
    -------
    dict
        ``middle`` (count of replaced eigenvalues), ``log_am_gm`` (gap of the forward / arithmetic-mean fit) and
        ``log_gm_hm`` (gap of the inverse / harmonic-mean fit); zeros for an empty or flat middle.
    """
    _check_ranks(spike_rank, dip_rank, "am")
    lam0 = lam.detach().double().clamp_min(0)
    middle = _middle_indices(lam0, spike_rank, dip_rank)
    if middle.numel() == 0:
        return {"middle": 0, "log_am_gm": 0.0, "log_gm_hm": 0.0}
    x = lam0[middle]
    am, gm, hm = (_flat_value(x, m).item() for m in FLAT_MEANS)
    tiny = torch.finfo(x.dtype).tiny
    return {
        "middle": int(middle.numel()),
        "log_am_gm": max(math.log(max(am, tiny) / max(gm, tiny)), 0.0),
        "log_gm_hm": max(math.log(max(gm, tiny) / max(hm, tiny)), 0.0),
    }


@torch.no_grad()
def temper_eigenvalues(lam: torch.Tensor, power: float = 1.0, spike_rank: int = 0, dip_rank: int = 0,
                       flat_mean: str = "am") -> torch.Tensor:
    """Condition a spectrum: two-sided projection, then power, rescaled so that the sum (trace) is unchanged.

    The projection acts on the *estimate* and therefore precedes the power, which acts on how the estimate is used.
    With ``flat_mean`` other than ``"am"`` the projection changes the trace and the final rescale multiplies the kept
    eigenvalues by a common factor; the shape of the metric (ratios of eigenvalues) is unaffected.

    Parameters
    ----------
    lam : torch.Tensor
        Eigenvalues (any order), clamped at zero internally.
    power : float
        Exponent applied to the eigenvalues (``0.5`` halves the log-condition number; ``1`` = off).
    spike_rank, dip_rank, flat_mean : int, int, str
        See :func:`project_spectrum`; zero ranks leave the spectrum unprojected.

    Returns
    -------
    torch.Tensor
        Conditioned eigenvalues with the same sum as the input (same dtype).
    """
    lam0 = lam.clamp_min(0)
    total = lam0.sum()
    out = project_spectrum(lam0, spike_rank, dip_rank, flat_mean)
    if power != 1.0:
        out = out.pow(power)
    return out * (total / out.sum().clamp_min(torch.finfo(out.dtype).tiny))


@dataclass(frozen=True)
class SpectrumSpec:
    """How ADMM conditions the spectrum of its unit-diagonal curvature factors (hashable; part of cache keys).

    Parameters
    ----------
    power : float
        Tempering exponent (``admm_curvature_power``; ``1`` = off).
    spike_rank : int
        Largest eigenvalues kept exactly by the projection (``admm_curvature_spike_rank``).
    dip_rank : int
        Smallest eigenvalues kept exactly (``admm_curvature_dip_rank``).
    flat_mean : str
        Shared value of the replaced middle (``admm_curvature_flat_mean``), one of :data:`FLAT_MEANS`.
    """

    power: float = 1.0
    spike_rank: int = 0
    dip_rank: int = 0
    flat_mean: str = "am"

    def __post_init__(self) -> None:
        if not self.power > 0:
            raise ValueError(f"admm_curvature_power must be > 0, got {self.power}")
        _check_ranks(self.spike_rank, self.dip_rank, self.flat_mean)

    @classmethod
    def from_config(cls, quant_config: dict) -> SpectrumSpec:
        """Read the four ``admm_curvature_*`` keys (missing keys = identity)."""
        return cls(
            power=float(quant_config.get("admm_curvature_power", 1.0)),
            spike_rank=int(quant_config.get("admm_curvature_spike_rank", 0) or 0),
            dip_rank=int(quant_config.get("admm_curvature_dip_rank", 0) or 0),
            flat_mean=str(quant_config.get("admm_curvature_flat_mean", "am") or "am"),
        )

    @property
    def is_identity(self) -> bool:
        """True when the spectrum is used as estimated."""
        return self.power == 1.0 and self.spike_rank == 0 and self.dip_rank == 0

    def apply(self, lam: torch.Tensor) -> torch.Tensor:
        """:func:`temper_eigenvalues` with this spec."""
        return temper_eigenvalues(lam, self.power, self.spike_rank, self.dip_rank, self.flat_mean)

    def gaps(self, lam: torch.Tensor) -> dict:
        """:func:`projection_gaps` of the raw spectrum under this spec's ranks."""
        return projection_gaps(lam, self.spike_rank, self.dip_rank)


IDENTITY_SPECTRUM = SpectrumSpec()
"""The spec that uses the factors as estimated (default of the ADMM solver)."""
