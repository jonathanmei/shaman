# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..modules.linear import NanoQuantLinear
from ..optimi import AdamW
from ..utils.cache import ArtifactCache, admm_key
from ..utils.utils import cleanup_memory, find_layers, set_seed
from .admm_dbf import factorize_admm_dbf
from .admm_nq import factorize_admm_nanoquant
from .latent import hard_sign, sign_flips


@torch.jit.script
def fused_weighted_mse(pred, tgt, importance):
    return ((pred.float() - tgt.float()).square() * importance).sum()


@torch.jit.script
def fused_weighted_mahalanobis(pred, tgt, curvature):
    """Dense output-feature quadratic loss for block reconstruction errors."""
    error = (pred.float() - tgt.float()).reshape(-1, pred.size(-1))
    curvature = curvature.float()
    return ((error @ curvature) * error).sum()


# ----------------------------------------------------------------------------------------------------
# Block-output curvature: which weighting the block losses use, and how the dense matrix is conditioned
# ----------------------------------------------------------------------------------------------------
@dataclass
class BlockCurvature:
    """Output-side curvature of one decoder block as used by the block reconstruction losses.

    Attributes
    ----------
    importance : torch.Tensor
        Per-feature weights of the diagonal loss ``sum_t sum_j E_tj^2 importance_j`` (shape ``(hidden,)``).
    dense : torch.Tensor or None
        Conditioned dense matrix of the Mahalanobis loss ``sum_t E_t dense E_t^T`` (``(hidden, hidden)``),
        ``None`` when no dense curvature statistics exist.
    optimize_dense : bool
        Whether the tuning stages minimise the dense loss (``block_loss="mahalanobis"``) or the diagonal one.
    summary : dict
        Spectrum summary of ``dense`` (see :func:`spectrum_summary``); empty when ``dense`` is ``None``.
    """
    importance: torch.Tensor
    dense: torch.Tensor | None
    optimize_dense: bool
    summary: dict


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
                        mix: float = 1.0) -> torch.Tensor:
    """Regularise the spectrum of a dense curvature matrix, preserving its trace.

    Parameters
    ----------
    cov : torch.Tensor
        Symmetric PSD matrix ``(d, d)``.
    cond_max : float
        If ``> 0``, floor the eigenvalues at ``lam_max / cond_max`` so the condition number is at most
        ``cond_max`` (ignored directions are lifted rather than dominant ones truncated).
    power : float
        Raise the eigenvalues to this power (``0.5`` halves the log-condition number; ``1`` = off).
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
    trace = diag.sum()
    out = cov64
    if power != 1.0 or cond_max > 0:
        lam, Q = torch.linalg.eigh(cov64)
        lam = lam.clamp_min(0)
        if power != 1.0:
            lam = lam.pow(power)
        if cond_max > 0:
            lam = lam.clamp_min(lam.max() / cond_max)
        out = (Q * lam) @ Q.mT
        out = out * (trace / out.diagonal().sum().clamp_min(1e-30))
    if mix != 1.0:
        out = (1.0 - mix) * torch.diag(diag) + mix * out
    return (0.5 * (out + out.mT)).float()


@torch.no_grad()
def block_curvature(sublayers: dict, hidden_size: int, quant_config: dict, dev: str) -> BlockCurvature:
    """Select and condition the output-side curvature of a decoder block.

    The block's output is written to the residual stream by ``mlp.down_proj`` (``fc2`` for OPT), so that
    layer's output-side statistics (``o_norm`` and, with Kronecker curvature, the dense ``o_cov``) are the
    natural weights of the block reconstruction error. With ``block_loss_source="nkp"`` the diagonal loss uses
    ``o_norm`` (the legacy behaviour) and the dense loss the Kronecker output factor ``o_cov``; with ``"plain"``
    both come from the unweighted output-gradient covariance ``o_cov_plain`` (its diagonal for the diagonal loss).
    The dense matrix is conditioned by the ``block_loss_*`` knobs and is optimised only when
    ``block_loss="mahalanobis"`` (otherwise it is logged for diagnostics).

    Parameters
    ----------
    sublayers : dict
        ``name -> nn.Linear`` of the block (``find_layers``), before factorisation.
    hidden_size : int
        Residual-stream width (fallback for uniform importance).
    quant_config : dict
        Quantisation configuration (``block_loss``, ``block_loss_cond_max``, ``block_loss_power``,
        ``block_loss_mix``).
    dev : str
        Device of the returned tensors.

    Returns
    -------
    BlockCurvature
    """
    block_loss = quant_config.get("block_loss", "diag")
    if block_loss not in ("diag", "mahalanobis"):
        raise ValueError(f"Unknown block_loss: {block_loss}")
    source = quant_config.get("block_loss_source", "nkp")
    if source not in ("nkp", "plain"):
        raise ValueError(f"Unknown block_loss_source: {source}")
    layer = sublayers.get('mlp.down_proj', sublayers.get('fc2', None))
    if layer is None or not hasattr(layer, 'o_norm'):
        importance = torch.ones(hidden_size, device=dev)
        o_cov = None
    elif source == "plain":
        o_cov = getattr(layer, 'o_cov_plain', None)
        if o_cov is None:
            raise ValueError("block_loss_source='plain' requires the plain output-gradient covariance "
                             "(curvature='kron' with block-output layers in importance.PLAIN_COV_LAYERS)")
        o_cov = o_cov.to(dev)
        importance = o_cov.diagonal().clone()
    else:
        importance = layer.o_norm.to(dev)
        o_cov = getattr(layer, 'o_cov', None)
    if block_loss == "mahalanobis" and o_cov is None:
        raise ValueError("block_loss='mahalanobis' requires dense Kron curvature statistics")
    dense = None
    summary: dict = {}
    if o_cov is not None:
        dense = condition_curvature(o_cov.to(dev), cond_max=float(quant_config.get("block_loss_cond_max", 0.0) or 0.0),
                                    power=float(quant_config.get("block_loss_power", 1.0)),
                                    mix=float(quant_config.get("block_loss_mix", 1.0)))
        summary = spectrum_summary(dense)
    return BlockCurvature(importance=importance, dense=dense, optimize_dense=block_loss == "mahalanobis",
                          summary=summary)


# ----------------------------------------------------------------------------------------------------
# Fresh input-side curvature for ADMM and the diagnostics relating ADMM's objective to the block loss
# (docs/admm_block_tuning_curvature.html, sections 4.3-2 and 5.4)
# ----------------------------------------------------------------------------------------------------
@torch.no_grad()
def input_second_moment(block, layer: nn.Module, block_inputs: torch.Tensor, kwargs: dict,
                        num_samples: int) -> torch.Tensor:
    """Plain second moment ``sum_t x_t x_t^T / T`` of the inputs reaching ``layer`` inside ``block``.

    Measured on the block's current weights and on the activations of the quantised prefix
    (``block_inputs``), i.e. exactly the inputs the layer will see once binarised. Its mean diagonal is
    ``E_t[x^2]``, the scale of the calibration-time ``i_norm``/``i_cov``.

    Parameters
    ----------
    block : nn.Module
        Decoder block containing ``layer``.
    layer : nn.Module
        Linear layer about to be factorised.
    block_inputs : torch.Tensor
        Quantised-prefix activations ``(num_samples, seqlen, hidden)``.
    kwargs : dict
        Extra block forward arguments (attention mask, position embeddings, ...).
    num_samples : int
        Number of calibration samples to run.

    Returns
    -------
    torch.Tensor
        ``(in_features, in_features)`` fp32 on the block's device.
    """
    acc = torch.zeros(layer.in_features, layer.in_features, dtype=torch.float32, device=block_inputs.device)
    count = [0]

    def hook(_m, inp, _out):
        x = inp[0].detach().flatten(0, -2).float()
        acc.addmm_(x.mT, x)
        count[0] += x.shape[0]

    handle = layer.register_forward_hook(hook)
    try:
        for j in range(num_samples):
            block(block_inputs[j:j + 1], **kwargs)
    finally:
        handle.remove()
    return acc / max(1, count[0])


@torch.no_grad()
def factor_drift(stale: torch.Tensor, fresh: torch.Tensor, top_k: int = 16, eps: float = 1e-6) -> dict:
    """How far a fresh curvature factor has moved from its calibration-time (stale) estimate.

    Parameters
    ----------
    stale, fresh : torch.Tensor
        Symmetric PSD ``(n, n)`` matrices on the same scale.
    top_k : int
        Size of the leading eigenspaces compared.
    eps : float
        Relative eigenvalue floor of ``stale`` for its inverse square root.

    Returns
    -------
    dict
        ``rel_spectral`` = ``||stale^-1/2 (fresh - stale) stale^-1/2||_2`` (the bound of eq. 11 in the design
        note), ``max_angle_deg`` = largest principal angle between the top-``k`` eigenspaces,
        ``trace_ratio`` = ``tr(fresh) / tr(stale)``.
    """
    S = stale.double()
    S = 0.5 * (S + S.mT)
    Fm = fresh.double()
    Fm = 0.5 * (Fm + Fm.mT)
    lam_s, Q_s = torch.linalg.eigh(S)
    lam_s = lam_s.clamp_min(eps * lam_s.max().clamp_min(1e-30))
    inv_half = (Q_s / lam_s.sqrt()) @ Q_s.mT
    M = inv_half @ (Fm - S) @ inv_half
    rel = torch.linalg.eigvalsh(0.5 * (M + M.mT)).abs().max().item()
    _, Q_f = torch.linalg.eigh(Fm)
    k = min(top_k, S.shape[0])
    sig = torch.linalg.svdvals(Q_s[:, -k:].mT @ Q_f[:, -k:]).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.arccos(sig.min())).item()
    return {"rel_spectral": rel, "max_angle_deg": angle,
            "trace_ratio": (Fm.diagonal().sum() / S.diagonal().sum().clamp_min(1e-30)).item()}


def format_drift(d: dict, title: str = "input factor drift") -> str:
    """One-line rendering of :func:`factor_drift`."""
    return (f"[{title}] ||R_s^-1/2 (R_f - R_s) R_s^-1/2||_2 = {d['rel_spectral']:.3f} | "
            f"top-k max principal angle {d['max_angle_deg']:.1f} deg | tr(R_f)/tr(R_s) = {d['trace_ratio']:.3f}")


@torch.no_grad()
def mahalanobis_weight_error(W: torch.Tensor, W_hat: torch.Tensor, L: torch.Tensor | None,
                             R: torch.Tensor | None) -> float:
    """Curvature-weighted weight error ``tr(L (W - W_hat) R (W - W_hat)^T)`` (eq. 6 of the design note).

    ``L``/``R`` may be dense matrices or diagonal vectors (``None`` = identity), on the raw statistics scale.
    """
    E = (W.float() - W_hat.float())
    LE = E if L is None else (L.float() @ E if L.dim() == 2 else L.float().unsqueeze(1) * E)
    ER = E if R is None else (E @ R.float() if R.dim() == 2 else E * R.float().unsqueeze(0))
    return (LE * ER).sum().item()


@torch.no_grad()
def evaluate_block_loss(block, block_inputs, block_target_outputs, curvature: BlockCurvature, kwargs,
                        num_samples: int) -> tuple[float, float | None]:
    """Per-element diagonal and dense block losses of the block's current parameters (no optimisation)."""
    numel = block_target_outputs.numel()
    diag = torch.zeros((), device=block_target_outputs.device)
    dense = torch.zeros((), device=block_target_outputs.device)
    for j in range(num_samples):
        y = block(block_inputs[j:j + 1], **kwargs)[0]
        tgt = block_target_outputs[j:j + 1]
        diag += fused_weighted_mse(y, tgt, curvature.importance)
        if curvature.dense is not None:
            dense += fused_weighted_mahalanobis(y, tgt, curvature.dense)
    return (diag / numel).item(), (dense / numel).item() if curvature.dense is not None else None


# ----------------------------------------------------------------------------------------------------
# Tuning loops
# ----------------------------------------------------------------------------------------------------
def get_param_group_config(target_module, binary_lr=1e-5, scale_lr=1e-5, bias_lr=1e-5):
    """
    Get the parameter group config for the optimizer.
    """
    # create param groups
    groups = {'binary': [], 'scale': [], 'bias': []}
    # collect params
    for module in target_module.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            # get tag
            tag = getattr(param, 'optim_group', None)
            # fallback if no tag (for bias)
            if tag is None:
                if param.ndim == 1 and 'bias' in name:
                    tag = 'bias'
                else:
                    continue
            if tag in groups:
                groups[tag].append(param)
    # collect and return param groups with respective lr
    configs = []
    for key, lr in zip(groups.keys(), [binary_lr, scale_lr, bias_lr]):
        if groups[key]:
            configs.append({'params': groups[key], 'lr': lr})
    return configs


def _tune_loop(block, optimizer, scheduler, block_inputs, block_target_outputs, curvature: BlockCurvature, kwargs,
               batch_size: int, epochs: int, num_samples: int) -> None:
    """Shared epoch loop of :func:`tune_nonfact` and :func:`tune_fact`.

    Minimises the selected block loss (diagonal or dense, see :class:`BlockCurvature`) with gradient
    accumulation over ``batch_size`` samples and logs **both** losses every epoch, so that the
    objective actually optimised and the other one can be compared across runs.
    """
    device = block_target_outputs.device
    numel = block_target_outputs.numel()
    importance, dense = curvature.importance, curvature.dense
    t0 = time.time()
    for epoch in range(epochs):
        data_idx = torch.randperm(num_samples, device="cpu", dtype=torch.long)
        epoch_diag = torch.zeros(1, device=device)
        epoch_dense = torch.zeros(1, device=device)
        for i in range(num_samples):
            idx = data_idx[i].item()
            y = block(block_inputs[idx:idx + 1], **kwargs)[0]
            tgt = block_target_outputs[idx:idx + 1]
            if curvature.optimize_dense:
                loss = fused_weighted_mahalanobis(y, tgt, dense)
                with torch.no_grad():
                    other = fused_weighted_mse(y.detach(), tgt, importance)
                epoch_dense += loss.detach()
                epoch_diag += other
            else:
                loss = fused_weighted_mse(y, tgt, importance)
                if dense is not None:
                    with torch.no_grad():
                        epoch_dense += fused_weighted_mahalanobis(y.detach(), tgt, dense)
                epoch_diag += loss.detach()
            (loss / batch_size).backward()
            if (i + 1) % batch_size == 0 or (i + 1) == num_samples:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        cleanup_memory()
        main = epoch_dense if curvature.optimize_dense else epoch_diag
        msg = f"\t\t(Epoch {epoch+1:02d}/{epochs:02d}) Block Loss: {(main / numel).item():.4e}"
        msg += f" | diag {(epoch_diag / numel).item():.4e}"
        if dense is not None:
            msg += f" | dense {(epoch_dense / numel).item():.4e}"
        if epoch == epochs - 1:
            msg += f" | {time.time() - t0:.0f}s"
        print(msg)


@torch.enable_grad()
def tune_nonfact(block, block_inputs, block_target_outputs, curvature: BlockCurvature, kwargs, quant_config):
    """Tune the still full-precision linear layers of ``block`` to absorb the quantisation error so far."""
    set_seed(quant_config['seed'])
    batch_size = quant_config['nonfact_batch_size']
    epochs = quant_config['nonfact_epochs']
    num_samples = quant_config['num_calib_samples']
    total_steps = math.ceil(num_samples / batch_size) * epochs
    lr = quant_config['nonfact_lr']
    params = []
    for module in block.modules():
        if isinstance(module, nn.Linear):
            module.weight.requires_grad = True
            params.append(module.weight)
    assert len(params) > 0, "No linear layers found in the block"
    optimizer = AdamW(params, lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4 * lr)
    _tune_loop(block, optimizer, scheduler, block_inputs, block_target_outputs, curvature, kwargs, batch_size, epochs,
               num_samples)
    for p in params:
        p.requires_grad = False
    block.zero_grad(set_to_none=True)
    del params, optimizer


def _to_device(results: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in results.items()}


@torch.no_grad()
def factorize_and_replace(layer, name, rank, quant_config, cache: ArtifactCache | None = None,
                          input_factor: torch.Tensor | None = None):
    """
    Factorizes and replaces a submodule with a quantized version (NanoQuantLinear).

    When ``cache`` is enabled the ADMM solution is memoised under a content-addressed key
    (weight bytes + curvature tensors + ADMM settings), so bit-identical inputs never re-run ADMM.
    Returns ``(new_module, factor_results)``; ``factor_results.cache_hit`` records whether the memo hit.

    Parameters
    ----------
    input_factor : torch.Tensor, optional
        Fresh, already shrunk input second moment ``(in, in)`` (see :func:`input_second_moment`). When given it
        replaces the calibration-time input curvature: ``i_norm`` becomes its diagonal and, on the dense path,
        ``i_cov`` the matrix itself. With ``block_diagnostics`` the Mahalanobis weight error of the solution is
        printed under both the stale and the fresh factor.
    """
    set_seed(quant_config['seed'])
    lx_orig = find_layers(layer)[name]
    original_weight = lx_orig.weight.data.clone()
    weight_for_factorization = original_weight.clone()
    new_module = lx_orig
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. Iterative Factorization and Module Conversion ---
    W_res = weight_for_factorization.clone()

    admm_time = time.time()
    # Select factorization function based on type
    is_transpose = W_res.shape[0] < W_res.shape[1]
    # Dense Kronecker curvature factors are present only when calibrated with curvature='kron'
    i_cov_stale = getattr(lx_orig, 'i_cov', None)
    i_norm_stale = lx_orig.i_norm
    o_cov = getattr(lx_orig, 'o_cov', None)
    i_cov, i_norm = i_cov_stale, i_norm_stale
    if input_factor is not None:
        input_factor = input_factor.to(i_norm_stale.device, torch.float32)
        i_norm = input_factor.diagonal().clone()
        if o_cov is not None:
            i_cov = input_factor

    memo_key = None
    factor_results = None
    if cache is not None and cache.enabled:
        memo_key = admm_key(W_res, i_norm, lx_orig.o_norm, i_cov, o_cov, rank, quant_config)
        cached = cache.load("admm", memo_key)
        if cached is not None:
            factor_results = _to_device(cached, device)
    cache_hit = factor_results is not None

    if factor_results is None:
        if quant_config['admm_type'] == 'dbf':
            if i_cov is not None or o_cov is not None:
                raise ValueError("curvature='kron' is only supported with admm_type='nanoquant'")
            factor_results = factorize_admm_dbf(W_res.to(device), i_norm.to(device), lx_orig.o_norm.to(device),
                                                mid_rank=rank, iters=quant_config['admm_outer_iters'],
                                                is_transpose=is_transpose)
        elif quant_config['admm_type'] == 'nanoquant':
            eigh_dtype = getattr(torch, quant_config.get('kron_eigh_dtype', 'float64'))
            factor_results = factorize_admm_nanoquant(
                W_res.to(device), i_norm.to(device), lx_orig.o_norm.to(device), mid_rank=rank,
                outer_iters=quant_config['admm_outer_iters'], inner_iters=quant_config['admm_inner_iters'],
                reg=quant_config['admm_reg'],
                is_transpose=is_transpose, rho_scheduler=quant_config['admm_penalty_scheduler'],
                print_admm_steps=quant_config['admm_print_steps'],
                i_cov=None if i_cov is None else i_cov.to(device),
                o_cov=None if o_cov is None else o_cov.to(device),
                eigh_dtype=eigh_dtype, mid_scale=bool(quant_config.get('admm_mid_scale', False)))
        else:
            raise ValueError(f"Unknown admm_type: {quant_config['admm_type']}")
        if memo_key is not None:
            cache.save("admm", memo_key, _to_device(factor_results, "cpu"))
    admm_time = time.time() - admm_time
    if quant_config.get('block_diagnostics', False):
        W_final = factor_results["W_final"].to(W_res.device)
        L = o_cov if o_cov is not None else lx_orig.o_norm
        ref = mahalanobis_weight_error(W_res, torch.zeros_like(W_res), L, i_cov_stale if i_cov_stale is not None
                                       else i_norm_stale)
        stale = mahalanobis_weight_error(W_res, W_final, L, i_cov_stale if i_cov_stale is not None else i_norm_stale)
        msg = f"\t\tADMM Mahalanobis weight error (normalised by the weight's own): stale-R {stale / max(ref, 1e-30):.4e}"
        if input_factor is not None:
            R_f = input_factor if o_cov is not None else input_factor.diagonal()
            ref_f = mahalanobis_weight_error(W_res, torch.zeros_like(W_res), L, R_f)
            fresh = mahalanobis_weight_error(W_res, W_final, L, R_f)
            msg += f" | fresh-R {fresh / max(ref_f, 1e-30):.4e}"
        print(msg)
    # The dense factors are no longer needed for this layer: free the memory.
    for buf_name in ('i_cov', 'o_cov', 'o_cov_plain'):
        if hasattr(lx_orig, buf_name):
            delattr(lx_orig, buf_name)
    del i_cov, o_cov, i_cov_stale

    # Assemble final factorization results
    final_factor_results = argparse.Namespace(**factor_results)
    final_factor_results.cache_hit = cache_hit

    # Replace module class and convert
    do_tuning = quant_config['tune_fact']
    new_module.__class__ = NanoQuantLinear
    new_module.__quant_convert__(do_train=do_tuning, rank=rank, factor_results=final_factor_results,
                                 keep_latent=bool(quant_config.get('retain_latent', False)))

    # --- 2. Finalization ---
    if not do_tuning and new_module.bias is not None and hasattr(lx_orig, 'bias') and lx_orig.bias is not None:
        new_module.bias.data.copy_(lx_orig.bias.data)

    recon_error_raw = (factor_results["W_final"].cpu() - weight_for_factorization.cpu()).square().sum().item()
    original_norm_sq = weight_for_factorization.square().sum().item()
    per_el_error = recon_error_raw / W_res.numel()
    if original_norm_sq > 0:
        normalized_error = recon_error_raw / original_norm_sq
        tag = " (memo hit)" if cache_hit else ""
        print(
            f"\t\tADMM weight recon error: raw={recon_error_raw:.4f}, norm={normalized_error:.4f}, per_el={per_el_error:.4e}, ADMM time={admm_time:.2f}s{tag}"
        )

    del original_weight, W_res, lx_orig
    cleanup_memory()

    return new_module, final_factor_results


@torch.enable_grad()
def tune_fact(block, target_linear, block_inputs, block_target_outputs, curvature: BlockCurvature, kwargs,
              quant_config):
    """Tune the latent binary factors and scales of ``target_linear`` (STE forward), then harden it.

    The number of sign flips relative to the ADMM initialisation is logged; with ``retain_latent`` the
    frozen latents are kept on the layer for a later latent-aware KD stage.
    """
    set_seed(quant_config['seed'])
    batch_size = quant_config['fact_batch_size']
    epochs = quant_config['fact_epochs']
    num_samples = quant_config['num_calib_samples']
    total_steps = math.ceil(num_samples / batch_size) * epochs
    binary_lr = quant_config['fact_binary_lr']
    scale_lr = quant_config['fact_scale_lr']
    bias_lr = quant_config['fact_bias_lr']
    param_config = get_param_group_config(block, binary_lr=binary_lr, scale_lr=scale_lr, bias_lr=bias_lr)
    optimizer = AdamW(param_config, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4 * scale_lr)
    with torch.no_grad():
        init_signs = {n: hard_sign(p.detach()) for n, p in target_linear.named_parameters() if "latent" in n}
    _tune_loop(block, optimizer, scheduler, block_inputs, block_target_outputs, curvature, kwargs, batch_size, epochs,
               num_samples)
    with torch.no_grad():
        flips = sum(sign_flips(getattr(target_linear, n), s) for n, s in init_signs.items())
        total = sum(s.numel() for s in init_signs.values())
    if total:
        print(f"\t\tsign flips during factor tuning: {flips}/{total} ({flips / total:.3e})")
    del init_signs
    # harden latent binary weights
    target_linear.finalize(keep_latent=bool(quant_config.get('retain_latent', False)))
    del param_config, optimizer
