# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""A per-rank middle scale inserted at the KD stage only (``model_kd_mid_scale``).

The Scale-Binary-Scale-Binary-Scale export of the block stage (``admm_mid_scale``) carried the per-rank SVID
magnitudes into block tuning and KD and lost several PPL at 0.6B (docs/results.md, 2026-09-02); a mean-1 rebalanced
variant with its own learning rate did not recover it either. This module tests the remaining hypothesis in
isolation: the *degree of freedom* is useful to the end-to-end scale-only KD when it starts from the identity, i.e.
from exactly the two-scale model the block stage produced, so nothing upstream changes.
"""

from __future__ import annotations

import torch
from torch import nn

from ..modules.linear import NanoQuantLinear


def insert_unit_mid_scales(model: nn.Module) -> int:
    """Give every :class:`NanoQuantLinear` without a ``scale_mid`` a per-rank middle scale of ones.

    The deployed forward multiplies the rank activations by ``scale_mid`` when the attribute exists, so the model's
    function is unchanged until KD moves the new parameters. The parameter carries the ``"scale"`` optimiser tag and
    is collected by the KD stage like ``scale_pre`` / ``scale_post``; the bit counter (``model_accounting``) sees it.

    Parameters
    ----------
    model : nn.Module
        Reconstructed (pre-KD) model.

    Returns
    -------
    int
        Number of layers that received a middle scale.
    """
    n = 0
    for module in model.modules():
        if not isinstance(module, NanoQuantLinear):
            continue
        if getattr(module, "scale_mid", None) is not None:
            continue
        ref = module.scale_post
        rank = int(module.rank)
        param = nn.Parameter(torch.ones(rank, device=ref.device, dtype=ref.dtype), requires_grad=False)
        param.optim_group = "scale"
        module.scale_mid = param
        n += 1
    return n
