"""Stand-alone ADMM of one calibrated layer, with the run's settings, for the ``dump`` stage.

These mirror the private steps of :func:`nanoquant.core.rank_probe.probe_layer`. They live here rather than in
``rank_probe.py`` on purpose: that file is part of the cache fingerprint of every block checkpoint and of the rank-probe
artifact (``SOURCE_GROUPS`` in ``utils/cache.py``), so touching it would invalidate the cached runs this proof of
concept reads from.
"""

from __future__ import annotations

import torch
from torch import nn

from ..core.admm_nq import factorize_admm_nanoquant
from ..core.rank_probe import PROBE_EIGH_DTYPE
from ..utils.utils import has_mid_scale


@torch.no_grad()
def layer_curvature(lx: nn.Linear, dev: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None,
                                                      torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """The weight and registered curvature of a calibrated layer, as the ADMM and the error metric consume them.

    Parameters
    ----------
    lx : nn.Linear
        Layer carrying ``i_norm``/``o_norm`` (and ``i_cov``/``o_cov`` on the dense path) as buffers.
    dev : str
        Compute device.

    Returns
    -------
    tuple
        ``(W, i_norm, o_norm, i_cov, o_cov, L, R)``: ``i_cov``/``o_cov`` are ``None`` on the diagonal path;
        ``L``/``R`` are the output/input factors for ``mahalanobis_weight_error`` (dense if available, else the
        diagonal vectors). Curvature tensors are fp32 on ``dev``.
    """
    W = lx.weight.detach().to(dev)
    i_norm = lx.i_norm.detach().to(dev, torch.float32)
    o_norm = lx.o_norm.detach().to(dev, torch.float32)
    i_cov = getattr(lx, "i_cov", None)
    o_cov = getattr(lx, "o_cov", None)
    dense = i_cov is not None and o_cov is not None
    i_cov = i_cov.detach().to(dev, torch.float32) if dense else None
    o_cov = o_cov.detach().to(dev, torch.float32) if dense else None
    L = o_cov if dense else o_norm
    R = i_cov if dense else i_norm
    return W, i_norm, o_norm, i_cov, o_cov, L, R


@torch.no_grad()
def admm_for_layer(W: torch.Tensor, i_norm: torch.Tensor, o_norm: torch.Tensor, i_cov: torch.Tensor | None,
                   o_cov: torch.Tensor | None, rank: int, quant_config: dict, outer_iters: int | None = None,
                   eigh_dtype: torch.dtype = PROBE_EIGH_DTYPE) -> dict:
    """``factorize_admm_nanoquant`` of one layer against its registered curvature, with the run's ADMM settings.

    Parameters
    ----------
    W : torch.Tensor
        Weight ``(out, in)``.
    i_norm, o_norm : torch.Tensor
        Diagonal curvature.
    i_cov, o_cov : torch.Tensor or None
        Dense Kronecker factors (``None`` on the diagonal path).
    rank : int
        Factorisation rank.
    quant_config : dict
        Quantisation configuration (``admm_*`` keys).
    outer_iters : int, optional
        ADMM iterations; defaults to the run's ``admm_outer_iters``.
    eigh_dtype : torch.dtype
        Eigendecomposition precision.
    """
    return factorize_admm_nanoquant(
        W, i_norm, o_norm, mid_rank=rank,
        outer_iters=int(outer_iters if outer_iters is not None else quant_config.get("admm_outer_iters", 400)),
        inner_iters=quant_config.get("admm_inner_iters", 5), reg=quant_config.get("admm_reg", 3e-2),
        is_transpose=W.shape[0] < W.shape[1], rho_scheduler=quant_config.get("admm_penalty_scheduler", "linear"),
        print_admm_steps=False, i_cov=i_cov, o_cov=o_cov, eigh_dtype=eigh_dtype,
        mid_scale=has_mid_scale(quant_config),
        curvature_power=float(quant_config.get("admm_curvature_power", 1.0)))
