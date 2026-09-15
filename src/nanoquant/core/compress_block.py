# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import time

import torch
import torch.nn as nn

from ..modules.linear import NanoQuantLinear
from ..optimi import AdamW
from ..utils.cache import ArtifactCache, admm_key
from ..utils.utils import cleanup_memory, find_layers, predicted_loss, set_seed
from .admm_dbf import factorize_admm_dbf
from .admm_nq import EigCache, factorize_admm_nanoquant
from .curvature import SpectrumSpec
from .importance import shrink_toward_identity


@torch.jit.script
def fused_weighted_mse(pred, tgt, importance):
    return ((pred.float() - tgt.float()).square() * importance).sum()


# ----------------------------------------------------------------------------------------------------
# Block-output importance: the per-feature weights of the block reconstruction loss
# ----------------------------------------------------------------------------------------------------
@torch.no_grad()
def block_importance(sublayers: dict, hidden_size: int, dev: str) -> torch.Tensor:
    """Per-feature weights of the block reconstruction loss ``sum_t sum_j E_tj^2 importance_j``.

    The block's output is written to the residual stream by ``mlp.down_proj`` (``fc2`` for OPT), so that layer's
    output-side second moments ``o_norm`` are the natural weights of the block reconstruction error (the paper's
    weighted MSE). Falls back to uniform weights when no statistics are attached.

    Parameters
    ----------
    sublayers : dict
        ``name -> nn.Linear`` of the block (``find_layers``), before factorisation.
    hidden_size : int
        Residual-stream width (fallback for uniform importance).
    dev : str
        Device of the returned tensor.

    Returns
    -------
    torch.Tensor
        ``(hidden,)`` fp32.
    """
    layer = sublayers.get('mlp.down_proj', sublayers.get('fc2', None))
    if layer is None or not hasattr(layer, 'o_norm'):
        return torch.ones(hidden_size, device=dev)
    return layer.o_norm.to(dev)


# ----------------------------------------------------------------------------------------------------
# Fresh input-side curvature for ADMM and the diagnostics relating ADMM's objective to the block loss
# (docs/admm_block_tuning_curvature.html, sections 4.3-2 and 5.4)
# ----------------------------------------------------------------------------------------------------
@torch.no_grad()
def input_second_moment(block, layer: nn.Module, block_inputs: torch.Tensor, kwargs: dict,
                        num_samples: int, return_probe: bool = False):
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
    return_probe : bool
        Also return the layer's input on the first sample, ``(seqlen, in_features)`` fp32, which
        :func:`fresh_input_factor` uses to validate the reuse of the factor by another layer.

    Returns
    -------
    torch.Tensor or tuple
        ``(in_features, in_features)`` fp32 on the block's device; with ``return_probe`` the pair ``(R, probe)``.
    """
    acc = torch.zeros(layer.in_features, layer.in_features, dtype=torch.float32, device=block_inputs.device)
    count = [0]
    probe: list[torch.Tensor] = []

    def hook(_m, inp, _out):
        x = inp[0].detach().flatten(0, -2).float()
        if return_probe and not probe:
            probe.append(x.clone())
        acc.addmm_(x.mT, x)
        count[0] += x.shape[0]

    handle = layer.register_forward_hook(hook)
    try:
        for j in range(num_samples):
            block(block_inputs[j:j + 1], **kwargs)
    finally:
        handle.remove()
    R = acc / max(1, count[0])
    return (R, probe[0]) if return_probe else R


@torch.no_grad()
def layer_input_probe(block, layer: nn.Module, sample: torch.Tensor, kwargs: dict) -> torch.Tensor:
    """The input reaching ``layer`` for one block input ``sample`` ``(1, seqlen, hidden)``, as ``(seqlen, in)`` fp32."""
    captured: list[torch.Tensor] = []
    handle = layer.register_forward_hook(
        lambda _m, inp, _out: captured.append(inp[0].detach().flatten(0, -2).float().clone()))
    try:
        block(sample, **kwargs)
    finally:
        handle.remove()
    return captured[0]


@torch.no_grad()
def shared_input_groups(block, layers: dict[str, nn.Module], names, sample: torch.Tensor,
                        kwargs: dict) -> dict[str, str]:
    """Group the layers of a block that read the same activation tensor (q/k/v, gate/up).

    One forward pass with hooks records the storage pointer of every layer's input; layers with the same pointer
    form a group named after its first member in ``names`` order. Model-agnostic.

    Parameters
    ----------
    block : nn.Module
        Decoder block.
    layers : dict
        ``name -> module`` (from ``find_layers``).
    names : iterable of str
        Layer names in factorisation order.
    sample : torch.Tensor
        One block input ``(1, seqlen, hidden)``.
    kwargs : dict
        Extra block forward arguments.

    Returns
    -------
    dict
        ``layer name -> group key`` for every name present in ``layers``.
    """
    ptrs: dict[str, tuple] = {}
    handles = []

    def make_hook(name):
        def hook(_m, inp, _out):
            ptrs[name] = (inp[0].data_ptr(), tuple(inp[0].shape))
        return hook

    for n in names:
        if n in layers:
            handles.append(layers[n].register_forward_hook(make_hook(n)))
    try:
        block(sample, **kwargs)
    finally:
        for h in handles:
            h.remove()
    first: dict[tuple, str] = {}
    return {n: first.setdefault(ptrs[n], n) for n in names if n in ptrs}


def group_size(groups: dict[str, str], key: str) -> int:
    """Number of layers in the group ``key`` of :func:`shared_input_groups`."""
    return sum(1 for g in groups.values() if g == key)


@dataclass
class FreshFactor:
    """A fresh input second moment cached for the layers of one shared-input group.

    Attributes
    ----------
    R, R_shrunk : torch.Tensor
        Plain and shrunk second moments ``(in, in)``.
    probe : torch.Tensor
        The group's input on the first calibration sample, used to validate reuse.
    """
    R: torch.Tensor
    R_shrunk: torch.Tensor
    probe: torch.Tensor


@torch.no_grad()
def fresh_input_factor(block, layer: nn.Module, name: str, groups: dict[str, str], block_inputs: torch.Tensor,
                       kwargs: dict, num_samples: int, shrinkage: float, fresh_cache: dict[str, FreshFactor],
                       eig_cache: EigCache | None) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Fresh input factor of ``layer``, reusing the one measured for an earlier layer of the same group when valid.

    The reuse is validated by one block forward on the first sample: if the layer's input still equals the probe
    stored with the cached factor (relative difference ``<= 1e-3``), the factor and its shrunk form are returned
    as-is (so ADMM's eigendecomposition of the shrunk factor can be shared through ``eig_cache``). Otherwise, or
    for layers whose input nobody else reads, the factor is measured with :func:`input_second_moment`.

    Parameters
    ----------
    block, layer, name : nn.Module, nn.Module, str
        Decoder block, the layer about to be factorised and its name.
    groups : dict
        Output of :func:`shared_input_groups`.
    block_inputs, kwargs, num_samples : torch.Tensor, dict, int
        As for :func:`input_second_moment`.
    shrinkage : float
        ``calib_shrinkage`` applied to the measured factor.
    fresh_cache : dict
        ``group key -> FreshFactor``; updated in place, cleared by the caller at the end of the block.
    eig_cache : EigCache or None
        Shared factors are registered here; a failed validation clears it.

    Returns
    -------
    tuple
        ``(R, R_shrunk, reused)``.
    """
    key = groups.get(name, name)
    shared = group_size(groups, key) > 1
    hit = fresh_cache.get(key) if shared else None
    if hit is not None:
        probe = layer_input_probe(block, layer, block_inputs[:1], kwargs)
        if probe.shape == hit.probe.shape and \
                (probe - hit.probe).norm() <= 1e-3 * hit.probe.norm().clamp_min(torch.finfo(torch.float32).tiny):
            return hit.R, hit.R_shrunk, True
        # the shared input changed since the factor was measured (should not happen: only norm and full-precision
        # linear weights are tuned and neither feeds these inputs); measure again and forget the eigendecompositions
        del fresh_cache[key]
        if eig_cache is not None:
            eig_cache.clear()
    R, probe = input_second_moment(block, layer, block_inputs, kwargs, num_samples, return_probe=True)
    R_shrunk = shrink_toward_identity(R, shrinkage)
    if shared:
        fresh_cache[key] = FreshFactor(R, R_shrunk, probe)
        if eig_cache is not None:
            eig_cache.register(R_shrunk)
    return R, R_shrunk, False


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
def evaluate_block_loss(block, block_inputs, block_target_outputs, importance: torch.Tensor, kwargs,
                        num_samples: int) -> float:
    """Per-element weighted block loss of the block's current parameters (no optimisation)."""
    numel = block_target_outputs.numel()
    total = torch.zeros((), device=block_target_outputs.device)
    for j in range(num_samples):
        y = block(block_inputs[j:j + 1], **kwargs)[0]
        total += fused_weighted_mse(y, block_target_outputs[j:j + 1], importance)
    return (total / numel).item()


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


def mlp_only_forward(block, block_inputs, kwargs, target_name: str | None):
    """Forward of only the MLP half of an HF decoder layer, for tuning an ``mlp.*`` layer whose attention is frozen.

    Blocks of the Llama / Qwen family compute ``h = x + self_attn(input_layernorm(x))`` and
    ``out = h + mlp(post_attention_layernorm(h))``. Layers are binarised in the order q, v, o, k, gate, up, down, so
    when an ``mlp.*`` layer is tuned every attention linear is already a frozen :class:`NanoQuantLinear` and ``h`` is
    fixed: it is captured once for all samples (a forward pre-hook on ``post_attention_layernorm``) and the tuning
    forward runs the MLP path only.

    Parameters
    ----------
    block : nn.Module
        Decoder layer.
    block_inputs : torch.Tensor
        ``(samples, seq, hidden)`` block inputs.
    kwargs : dict
        Block forward keyword arguments (attention mask, position embeddings).
    target_name : str or None
        Name of the layer being tuned (``find_layers`` key).

    Returns
    -------
    callable or None
        ``forward_fn(idx) -> (1, seq, hidden)`` for sample ``idx``; ``None`` when the shortcut does not apply (an
        attention layer, a block without the HF attribute names such as OPT, or a still-trainable attention).
    """
    if not (target_name or "").startswith("mlp."):
        return None
    attn = getattr(block, "self_attn", None)
    mlp = getattr(block, "mlp", None)
    norm = getattr(block, "post_attention_layernorm", None)
    if attn is None or mlp is None or norm is None:
        return None
    if any(isinstance(m, nn.Linear) for m in attn.modules()) or any(p.requires_grad for p in attn.parameters()):
        return None
    captured: list[torch.Tensor] = []
    handle = norm.register_forward_pre_hook(lambda module, inp: captured.append(inp[0].detach()))
    h = torch.empty_like(block_inputs)
    try:
        with torch.no_grad():
            for j in range(block_inputs.shape[0]):
                captured.clear()
                block(block_inputs[j:j + 1], **kwargs)
                h[j:j + 1] = captured[0]
    finally:
        handle.remove()

    def forward_fn(idx: int) -> torch.Tensor:
        hj = h[idx:idx + 1]
        return hj + mlp(norm(hj))

    return forward_fn


# ----------------------------------------------------------------------------------------------------
# Tuning budget: epochs per layer from its sensitivity, plateau stopping, and which layers get a
# non-factorized retuning round
# ----------------------------------------------------------------------------------------------------
TUNE_EPOCH_WEIGHT_MODES = ("none", "type", "measured")

# Relative tuning effort per layer type, from the median relative block-loss jump each layer's binarisation causes
# (k/q ~ +2 %, o/v ~ +10 %, gate +23 %, up +32 %, down +44 %; docs/learnings.md). Anything else (OPT fc1/fc2,
# out_proj) keeps the full budget.
TYPE_EPOCH_WEIGHTS = {"q_proj": 0.25, "k_proj": 0.25, "v_proj": 0.5, "o_proj": 0.5, "gate_proj": 0.75,
                      "up_proj": 1.0, "down_proj": 1.0}


def scaled_epochs(epochs: int, scale: float) -> int:
    """``round(epochs * scale)``, at least 1."""
    return max(1, round(epochs * scale))


def tuning_epoch_weights(sensitivity: dict | None, admm_ranks: dict | None, block: int, names: list[str],
                         quant_config: dict) -> dict[str, float]:
    """Per-layer multipliers of the tuning epochs for one block (``tune_epoch_weights``).

    Parameters
    ----------
    sensitivity : dict or None
        Output of :func:`nanoquant.core.rank_probe.measure_sensitivity` (``"curves"``: ``"{block}.{name}" ->
        (a, beta)``), or ``None``.
    admm_ranks : dict or None
        Allocated ranks keyed ``"{block}.{name}"``.
    block : int
        Block index.
    names : list of str
        Layer names of the block in factorisation order.
    quant_config : dict
        ``tune_epoch_weights`` (``none`` = all 1; ``type`` = :data:`TYPE_EPOCH_WEIGHTS`; ``measured`` = the probe's
        predicted curvature-weighted error at the allocated rank, normalised to the block's largest, falling back to
        ``type`` when no curve is available) and ``tune_epoch_min_frac`` (floor of every weight).

    Returns
    -------
    dict
        ``name -> weight in [min_frac, 1]``.
    """
    mode = quant_config.get("tune_epoch_weights", "none") or "none"
    if mode not in TUNE_EPOCH_WEIGHT_MODES:
        raise ValueError(f"Unknown tune_epoch_weights: {mode!r} (choices: {TUNE_EPOCH_WEIGHT_MODES})")
    if mode == "none":
        return {n: 1.0 for n in names}
    floor = float(quant_config.get("tune_epoch_min_frac", 0.25))
    if mode == "measured":
        curves = (sensitivity or {}).get("curves", {})
        raw = {}
        for n in names:
            key = f"{block}.{n}"
            rank = (admm_ranks or {}).get(key)
            if key in curves and rank:
                raw[n] = predicted_loss(curves[key], int(rank))
        if len(raw) == len(names) and max(raw.values()) > 0:
            top = max(raw.values())
            return {n: max(floor, min(1.0, raw[n] / top)) for n in names}
    return {n: max(floor, TYPE_EPOCH_WEIGHTS.get(n.rsplit(".", 1)[-1], 1.0)) for n in names}


def nonfact_rounds(names: list[str], groups: dict[str, str], per_group: bool) -> dict[str, bool]:
    """Which layers are preceded by a non-factorized retuning round.

    With ``per_group`` (``nonfact_per_group``) only the first layer of each shared-input group
    (:func:`shared_input_groups`; layers reading the same activation cannot change each other's inputs) gets a
    round; the errors of the skipped members are absorbed by the next group's round. For the Qwen/Llama order
    q, v, o, k, gate, up, down that is q, o, gate, down: four rounds instead of seven.
    """
    if not per_group or not groups:
        return {n: True for n in names}
    return {n: groups.get(n, n) == n for n in names}


def _tune_loop(block, optimizer, scheduler, block_inputs, block_target_outputs, importance: torch.Tensor, kwargs,
               batch_size: int, epochs: int, num_samples: int, forward_fn=None, plateau_tol: float = 0.0) -> int:
    """Shared epoch loop of :func:`tune_nonfact` and :func:`tune_fact`.

    Minimises the weighted block reconstruction loss with gradient accumulation over ``batch_size`` samples and
    logs the per-element loss every epoch. ``forward_fn(idx)`` replaces the full block forward when given
    (:func:`mlp_only_forward`). With ``plateau_tol > 0`` the loop stops once an epoch improves the loss by less
    than that fraction (the cosine schedule keeps its full length and is simply truncated).

    Returns
    -------
    int
        Number of epochs run.
    """
    device = block_target_outputs.device
    numel = block_target_outputs.numel()
    t0 = time.time()
    prev = None
    run = 0
    for epoch in range(epochs):
        data_idx = torch.randperm(num_samples, device="cpu", dtype=torch.long)
        epoch_loss = torch.zeros(1, device=device)
        for i in range(num_samples):
            idx = data_idx[i].item()
            y = forward_fn(idx) if forward_fn is not None else block(block_inputs[idx:idx + 1], **kwargs)[0]
            loss = fused_weighted_mse(y, block_target_outputs[idx:idx + 1], importance)
            epoch_loss += loss.detach()
            (loss / batch_size).backward()
            if (i + 1) % batch_size == 0 or (i + 1) == num_samples:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        run = epoch + 1
        cur = (epoch_loss / numel).item()
        plateau = plateau_tol > 0 and prev is not None and (prev - cur) / max(abs(prev), 1e-30) < plateau_tol
        msg = f"\t\t(Epoch {epoch+1:02d}/{epochs:02d}) Block Loss: {cur:.4e}"
        if plateau:
            msg += " | plateau: stop"
        if epoch == epochs - 1 or plateau:
            msg += f" | {time.time() - t0:.0f}s"
        print(msg)
        if plateau:
            break
        prev = cur
    return run


@torch.enable_grad()
def tune_nonfact(block, block_inputs, block_target_outputs, importance: torch.Tensor, kwargs, quant_config,
                 target_name: str | None = None, epochs_scale: float = 1.0):
    """Tune the still full-precision linear layers of ``block`` to absorb the quantisation error so far.

    ``target_name`` (the layer about to be binarised) enables the MLP-only forward of :func:`mlp_only_forward`;
    ``epochs_scale`` multiplies ``nonfact_epochs`` (:func:`tuning_epoch_weights`).
    """
    set_seed(quant_config['seed'])
    batch_size = quant_config['nonfact_batch_size']
    epochs = scaled_epochs(quant_config['nonfact_epochs'], epochs_scale)
    num_samples = quant_config['num_calib_samples']
    total_steps = math.ceil(num_samples / batch_size) * epochs
    lr = quant_config['nonfact_lr']
    forward_fn = mlp_only_forward(block, block_inputs, kwargs, target_name)
    params = []
    for module in block.modules():
        if isinstance(module, nn.Linear):
            module.weight.requires_grad = True
            params.append(module.weight)
    assert len(params) > 0, "No linear layers found in the block"
    optimizer = AdamW(params, lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4 * lr)
    _tune_loop(block, optimizer, scheduler, block_inputs, block_target_outputs, importance, kwargs, batch_size, epochs,
               num_samples, forward_fn=forward_fn, plateau_tol=float(quant_config.get('tune_plateau_tol', 0.0) or 0.0))
    del forward_fn
    for p in params:
        p.requires_grad = False
    block.zero_grad(set_to_none=True)
    del params, optimizer


def _to_device(results: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in results.items()}


@torch.no_grad()
def factorize_and_replace(layer, name, rank, quant_config, cache: ArtifactCache | None = None,
                          input_factor: torch.Tensor | None = None, eig_cache: EigCache | None = None):
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
    eig_cache : EigCache, optional
        Eigendecompositions of curvature factors shared by several layers of the block (fresh input factor of
        q/k/v and gate/up), see :func:`fresh_input_factor`.
    """
    set_seed(quant_config['seed'])
    lx_orig = find_layers(layer)[name]
    new_module = lx_orig
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. Iterative Factorization and Module Conversion ---
    # one read-only copy of the weight serves the factoriser, the memo key and the error report
    W_res = lx_orig.weight.data.clone()
    weight_for_factorization = W_res

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
    spectrum_diag: dict | None = {} if quant_config.get('block_diagnostics', False) else None

    if factor_results is None:
        if quant_config['admm_type'] == 'dbf':
            if i_cov is not None or o_cov is not None:
                raise ValueError("curvature='kron' is only supported with admm_type='nanoquant'")
            factor_results = factorize_admm_dbf(W_res.to(device), i_norm.to(device), lx_orig.o_norm.to(device),
                                                mid_rank=rank, iters=quant_config['admm_outer_iters'],
                                                is_transpose=is_transpose)
        elif quant_config['admm_type'] == 'nanoquant':
            eigh_dtype = getattr(torch, quant_config.get('kron_eigh_dtype', 'float64'))
            spectrum = SpectrumSpec.from_config(quant_config)
            factor_results = factorize_admm_nanoquant(
                W_res.to(device), i_norm.to(device), lx_orig.o_norm.to(device), mid_rank=rank,
                outer_iters=quant_config['admm_outer_iters'], inner_iters=quant_config['admm_inner_iters'],
                reg=quant_config['admm_reg'],
                is_transpose=is_transpose, rho_scheduler=quant_config['admm_penalty_scheduler'],
                print_admm_steps=quant_config['admm_print_steps'],
                i_cov=None if i_cov is None else i_cov.to(device),
                o_cov=None if o_cov is None else o_cov.to(device),
                eigh_dtype=eigh_dtype, mid_scale=bool(quant_config.get('admm_mid_scale', False)),
                spectrum=spectrum, eig_cache=eig_cache, diagnostics=spectrum_diag)
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
        gaps = {k: v for k, v in (spectrum_diag or {}).items() if v}  # empty on an ADMM memo / EigCache hit
        if gaps:
            print("\t\tcurvature spectrum, replaced middle: " + " | ".join(
                f"{k}: n {v['middle']} log(AM/GM) {v['log_am_gm']:.3f} log(GM/HM) {v['log_gm_hm']:.3f}"
                for k, v in gaps.items()))
    # The dense factors are no longer needed for this layer: free the memory.
    for buf_name in ('i_cov', 'o_cov'):
        if hasattr(lx_orig, buf_name):
            delattr(lx_orig, buf_name)
    del i_cov, o_cov, i_cov_stale

    # Assemble final factorization results
    final_factor_results = argparse.Namespace(**factor_results)
    final_factor_results.cache_hit = cache_hit

    # Replace module class and convert
    do_tuning = quant_config['tune_fact']
    new_module.__class__ = NanoQuantLinear
    new_module.__quant_convert__(do_train=do_tuning, rank=rank, factor_results=final_factor_results)

    # --- 2. Finalization ---
    if not do_tuning and new_module.bias is not None and hasattr(lx_orig, 'bias') and lx_orig.bias is not None:
        new_module.bias.data.copy_(lx_orig.bias.data)

    W_final = factor_results["W_final"]
    recon_error_raw = (W_final - weight_for_factorization.to(W_final.device, W_final.dtype)).square().sum().item()
    original_norm_sq = weight_for_factorization.square().sum().item()
    per_el_error = recon_error_raw / W_res.numel()
    if original_norm_sq > 0:
        normalized_error = recon_error_raw / original_norm_sq
        tag = " (memo hit)" if cache_hit else ""
        print(
            f"\t\tADMM weight recon error: raw={recon_error_raw:.4f}, norm={normalized_error:.4f}, per_el={per_el_error:.4e}, ADMM time={admm_time:.2f}s{tag}"
        )

    del W_res, weight_for_factorization, lx_orig
    cleanup_memory()

    return new_module, final_factor_results


def _hard_sign(x: torch.Tensor) -> torch.Tensor:
    """Deployed sign convention of the binary factors: ``+1`` for ``x >= 0`` and ``-1`` otherwise."""
    return torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))


@torch.enable_grad()
def tune_fact(block, target_linear, block_inputs, block_target_outputs, importance: torch.Tensor, kwargs,
              quant_config, target_name: str | None = None, epochs_scale: float = 1.0):
    """Tune the latent binary factors and scales of ``target_linear`` (STE forward), then harden it.

    The number of sign flips relative to the ADMM initialisation is logged as a diagnostic. ``target_name`` enables
    the MLP-only forward of :func:`mlp_only_forward`; ``epochs_scale`` multiplies ``fact_epochs``.
    """
    set_seed(quant_config['seed'])
    forward_fn = mlp_only_forward(block, block_inputs, kwargs, target_name)
    batch_size = quant_config['fact_batch_size']
    epochs = scaled_epochs(quant_config['fact_epochs'], epochs_scale)
    num_samples = quant_config['num_calib_samples']
    total_steps = math.ceil(num_samples / batch_size) * epochs
    binary_lr = quant_config['fact_binary_lr']
    scale_lr = quant_config['fact_scale_lr']
    bias_lr = quant_config['fact_bias_lr']
    param_config = get_param_group_config(block, binary_lr=binary_lr, scale_lr=scale_lr, bias_lr=bias_lr)
    optimizer = AdamW(param_config, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4 * scale_lr)
    with torch.no_grad():
        init_signs = {n: _hard_sign(p.detach()) for n, p in target_linear.named_parameters() if "latent" in n}
    _tune_loop(block, optimizer, scheduler, block_inputs, block_target_outputs, importance, kwargs, batch_size, epochs,
               num_samples, forward_fn=forward_fn, plateau_tol=float(quant_config.get('tune_plateau_tol', 0.0) or 0.0))
    del forward_fn
    with torch.no_grad():
        flips = sum(int((_hard_sign(getattr(target_linear, n).detach()) != s).sum().item())
                    for n, s in init_signs.items())
        total = sum(s.numel() for s in init_signs.values())
    if total:
        print(f"\t\tsign flips during factor tuning: {flips}/{total} ({flips / total:.3e})")
    del init_signs
    # harden latent binary weights
    target_linear.finalize()
    del param_config, optimizer
