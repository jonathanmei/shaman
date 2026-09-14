# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Spectral tempering of the dense Kronecker curvature factors used by the ADMM data term.

Dense curvature factors estimated from a few hundred thousand tokens are strongly concentrated (a handful of
directions carry most of the trace). Raising their eigenvalues to a power below one, trace preserved, makes the
Mahalanobis data term trust the dominant directions less; the square root (``power = 0.5``) is the regret-optimal
metric under curvature uncertainty and the Fisher of a robustified loss at one-bit perturbation scale
(docs/curvature_tempering_theory.md).
"""

from __future__ import annotations

import torch


@torch.no_grad()
def temper_eigenvalues(lam: torch.Tensor, power: float = 1.0) -> torch.Tensor:
    """Raise a spectrum to ``power``, rescaled so that its sum (the trace) is unchanged.

    Parameters
    ----------
    lam : torch.Tensor
        Eigenvalues (any order), clamped at zero internally.
    power : float
        Exponent applied to the eigenvalues (``0.5`` halves the log-condition number; ``1`` = off).

    Returns
    -------
    torch.Tensor
        Tempered eigenvalues with the same sum as the input (same dtype).
    """
    lam0 = lam.clamp_min(0)
    total = lam0.sum()
    out = lam0.pow(power) if power != 1.0 else lam0
    return out * (total / out.sum().clamp_min(torch.finfo(out.dtype).tiny))
