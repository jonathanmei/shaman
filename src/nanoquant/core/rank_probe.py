# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Calibration-time sensitivity probe for the measured rank allocation.

After the calibration statistics are attached to the still-full-precision model, every layer to be factorised is
probed at a few candidate ranks (multiples of its uniform-rule rank): a short ADMM solve at that rank, the deployed
binary matrix rebuilt exactly as ``NanoQuantLinear`` computes it, and the curvature-weighted weight error
``J(r) = tr(L (W - W_hat) R (W - W_hat)^T)`` with the layer's own Kronecker Fisher factors (raw statistics scale,
so the values are comparable across layers). A power law ``J(r) = exp(a) r^-beta`` is fitted through the probes;
:func:`nanoquant.utils.utils.allocate_ranks_measured` then spends the bit budget by marginal predicted loss per
bit. The alternative ``svd`` proxy replaces the ADMM solve by the tail energy of the whitened spectrum of
``L^1/2 W R^1/2`` (no binarisation, one SVD per layer).

The probe never mutates the modules: the weights, statistics buffers and module classes are left as they were.
"""

from __future__ import annotations

import time

import torch
from torch import nn

from ..utils.utils import (
    RANK_STEP,
    _rank_ceiling,
    cleanup_memory,
    find_layers,
    fit_power_law,
    get_decoder_layers,
    has_mid_scale,
    parse_probe_ranks,
    set_seed,
    uniform_rank,
)
from .admm_nq import factorize_admm_nanoquant
from .compress_block import mahalanobis_weight_error

PROBE_KIND = "rank_probe"
# eigendecompositions inside the probe's ADMM: single precision is enough for a sensitivity ranking and is the
# dominant cost at 4B widths (the production solve keeps its configured precision)
PROBE_EIGH_DTYPE = torch.float32


def candidate_ranks(legacy_rank: int, in_features: int, out_features: int, multipliers: list[float],
                    max_ratio: float) -> list[int]:
    """Probe ranks: ``multipliers`` times the uniform rank, floored to the 32-grid, clamped to ``[32, ceiling]``.

    Parameters
    ----------
    legacy_rank : int
        Rank of the uniform rule for this layer.
    in_features, out_features : int
        Layer shape.
    multipliers : list of float
        Multiples of the uniform rank to probe.
    max_ratio : float
        Rank ceiling as a multiple of ``min(in, out)``.

    Returns
    -------
    list of int
        Sorted, de-duplicated candidate ranks.
    """
    hi = _rank_ceiling(in_features, out_features, max_ratio)
    out = set()
    for m in multipliers:
        r = (int(legacy_rank * m) // RANK_STEP) * RANK_STEP
        out.add(min(max(r, RANK_STEP), hi))
    return sorted(out)


def _sign(x: torch.Tensor) -> torch.Tensor:
    """``sign`` with ``sign(0) := +1``, the convention of ``NanoQuantLinear.binary_ste``."""
    y = x.sign()
    y[y == 0] = 1
    return y


@torch.no_grad()
def deployed_matrix(result: dict) -> torch.Tensor:
    """The matrix a ``NanoQuantLinear`` built from an ADMM ``result`` actually applies.

    ``W_hat = diag(scale_post) sign(A)^T [diag(scale_mid)] sign(B) diag(scale_pre)``, i.e. what
    ``NanoQuantLinear._compute_forward`` computes with hardened factors (in fp32).

    Parameters
    ----------
    result : dict
        Output of :func:`nanoquant.core.admm_nq.factorize_admm_nanoquant` (``A`` is ``(rank, out)``, ``B`` is
        ``(rank, in)``; ``scale_mid`` optional).

    Returns
    -------
    torch.Tensor
        ``(out, in)`` fp32.
    """
    U = _sign(result["A"].float()).mT  # (out, rank)
    V = _sign(result["B"].float())  # (rank, in)
    Vs = V * result["scale_pre"].float().reshape(1, -1)
    mid = result.get("scale_mid")
    if mid is not None:
        Vs = Vs * mid.float().reshape(-1, 1)
    return (U @ Vs) * result["scale_post"].float().reshape(-1, 1)


def _sqrt_factor(F: torch.Tensor) -> torch.Tensor:
    """Symmetric square root of a dense SPD factor, or the element-wise root of a diagonal one."""
    if F.dim() == 1:
        return F.float().clamp_min(0).sqrt()
    lam, Q = torch.linalg.eigh(F.float())
    return (Q * lam.clamp_min(0).sqrt()) @ Q.mT


@torch.no_grad()
def probe_layer(lx: nn.Linear, ranks: list[int], quant_config: dict, dev: str) -> dict[int, float]:
    """Curvature-weighted weight error of ``lx`` at each candidate rank.

    Parameters
    ----------
    lx : nn.Linear
        Layer carrying ``i_norm``/``o_norm`` (and ``i_cov``/``o_cov`` on the dense path) as buffers.
    ranks : list of int
        Candidate ranks.
    quant_config : dict
        Quantisation configuration (ADMM settings, ``rank_sensitivity``, ``rank_probe_iters``, ``seed``).
    dev : str
        Compute device.

    Returns
    -------
    dict
        ``rank -> J`` (``> 0``).
    """
    method = quant_config.get("rank_sensitivity", "none") or "none"
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
    out: dict[int, float] = {}
    if method == "svd":
        M = _sqrt_factor(L) @ W.float() @ _sqrt_factor(R) if dense else \
            (_sqrt_factor(L).unsqueeze(1) * W.float()) * _sqrt_factor(R).unsqueeze(0)
        s2 = torch.linalg.svdvals(M).square()
        tail = s2.flip(0).cumsum(0).flip(0)  # tail[r] = sum_{i >= r} s_i^2
        total = float(s2.sum().item())
        for r in ranks:
            j = float(tail[r].item()) if r < tail.numel() else 0.0
            out[r] = max(j, 1e-6 * total)
        return out
    if method != "admm":
        raise ValueError(f"Unknown rank_sensitivity: {method}")
    is_transpose = W.shape[0] < W.shape[1]
    for r in ranks:
        set_seed(quant_config["seed"])
        res = factorize_admm_nanoquant(
            W, i_norm, o_norm, mid_rank=r,
            outer_iters=int(quant_config.get("rank_probe_iters", 50) or 50),
            inner_iters=quant_config.get("admm_inner_iters", 5), reg=quant_config.get("admm_reg", 3e-2),
            is_transpose=is_transpose, rho_scheduler=quant_config.get("admm_penalty_scheduler", "linear"),
            print_admm_steps=False, i_cov=i_cov, o_cov=o_cov, eigh_dtype=PROBE_EIGH_DTYPE,
            mid_scale=has_mid_scale(quant_config),
            curvature_power=float(quant_config.get("admm_curvature_power", 1.0)),
            curvature_cond_max=float(quant_config.get("admm_curvature_cond_max", 0.0) or 0.0),
            curvature_spike_rank=int(quant_config.get("admm_curvature_spike_rank", 0) or 0))
        W_hat = deployed_matrix(res)
        out[r] = max(mahalanobis_weight_error(W, W_hat, L, R), 1e-30)
        del res, W_hat
    del W, i_norm, o_norm, i_cov, o_cov, L, R
    return out


@torch.no_grad()
def measure_sensitivity(model, layers_to_factorize, quant_config: dict, dev: str = "cuda") -> dict:
    """Probe every layer to be factorised and fit its sensitivity curve.

    Parameters
    ----------
    model : nn.Module
        Model with calibration statistics registered on its ``nn.Linear`` layers (CPU or device).
    layers_to_factorize : iterable of str
        Sub-layer names within each decoder block.
    quant_config : dict
        Quantisation configuration (``bits``, ``rank_probe_ranks``, ``rank_probe_iters``, ``rank_max_ratio``,
        ``rank_sensitivity``, ADMM settings).
    dev : str
        Compute device.

    Returns
    -------
    dict
        ``{"probes": {key: {rank: J}}, "curves": {key: (a, beta)}, "meta": {...}}`` with keys
        ``"<block>.<name>"``; plain Python containers so the artifact is ``weights_only``-loadable.
    """
    method = quant_config.get("rank_sensitivity", "none") or "none"
    multipliers = parse_probe_ranks(quant_config.get("rank_probe_ranks", "0.5,1.0,1.5") or "0.5,1.0,1.5")
    max_ratio = float(quant_config.get("rank_max_ratio", 1.0) or 1.0)
    num_scales = 3 if has_mid_scale(quant_config) else 2
    bits = quant_config["bits"]
    probes: dict[str, dict[int, float]] = {}
    curves: dict[str, tuple[float, float]] = {}
    t0 = time.time()
    blocks = get_decoder_layers(model)
    for i, block in enumerate(blocks):
        subset = find_layers(block)
        t_blk = time.time()
        n = 0
        for name in layers_to_factorize:
            if name not in subset:
                continue
            lx = subset[name]
            a, b = lx.in_features, lx.out_features
            ranks = candidate_ranks(uniform_rank(a, b, bits, num_scales), a, b, multipliers, max_ratio)
            key = f"{i}.{name}"
            probes[key] = probe_layer(lx, ranks, quant_config, dev)
            curves[key] = fit_power_law(probes[key])
            n += 1
        betas = " ".join(f"{curves[f'{i}.{nm}'][1]:.2f}" for nm in layers_to_factorize if nm in subset)
        print(f"\t[rank probe] block {i}: {n} layers in {time.time() - t_blk:.0f}s; beta {betas}")
        cleanup_memory()
    meta = {"method": method, "multipliers": multipliers,
            "iters": int(quant_config.get("rank_probe_iters", 50) or 50), "n_layers": len(probes),
            "seconds": time.time() - t0}
    print(f"[rank probe] {meta['n_layers']} layers probed with '{method}' at multiples {multipliers} of the "
          f"uniform rank in {meta['seconds']:.0f}s")
    return {"probes": probes, "curves": curves, "meta": meta}
