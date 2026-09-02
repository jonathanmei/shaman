# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from ..utils.utils import cleanup_memory

# Gradient Scaling Factor to prevent underflow (Numerical Stability)
GRAD_SCALE_FACTOR = 1e6
# --- Hyperparameters ---
# tracks the 99.9th percentile to filter outliers
PERCENTILE = 0.999
# Supported curvature estimates: diagonal (legacy) or Kronecker-factored (nearest Kronecker product)
CURVATURE_TYPES = ("diag", "kron")


# -----------------------------------------------------------------------------
# Core Logic: Robust Max Calculation
# -----------------------------------------------------------------------------
def _get_robust_batch_tau(norms: torch.Tensor, percentile: float = 0.999) -> torch.Tensor:
    """
    Returns the k-th largest value (robust max) as a 0-dim tensor on the same device.
    Avoids GPU->CPU sync from `.item()` inside hooks.
    """
    n = norms.numel()
    if n == 0:
        return norms.new_zeros(())
    k = max(1, int(n * (1.0 - percentile)))
    return torch.topk(norms.reshape(-1), k).values[-1]


def _clip_tokens(x: torch.Tensor, threshold) -> torch.Tensor:
    """Clip the L2 norm of every row (token) of ``x`` to at most ``threshold``.

    Parameters
    ----------
    x : torch.Tensor
        Token matrix of shape ``(tokens, features)``.
    threshold : torch.Tensor or float
        Maximum allowed per-token norm.

    Returns
    -------
    torch.Tensor
        Clipped copy of ``x``.
    """
    norms = torch.norm(x, dim=1, keepdim=True)
    clip_scales = torch.clamp(threshold / (norms + 1e-8), max=1.0)
    return x * clip_scales


# -----------------------------------------------------------------------------
# Hook Functions: Two-Phase & Online Robust Hooks
# -----------------------------------------------------------------------------
def _phase1_robust_profiling_hook(module, inputs, outputs, layer_name, global_stats, is_forward=True):
    if is_forward:
        x = inputs[0].detach().flatten(0, -2).float()
        key = "i_max"
    else:
        x = outputs[0].detach().flatten(0, -2).float() * GRAD_SCALE_FACTOR
        key = "o_max"

    norms = torch.norm(x, dim=1, keepdim=True)
    tau = _get_robust_batch_tau(norms, PERCENTILE)

    prev = global_stats[key].get(layer_name, None)
    global_stats[key][layer_name] = tau if prev is None else torch.maximum(prev, tau)


def _fixed_clipping_hook(module, inputs, outputs, layer_name, stats_dict, thresholds, stats_device, is_forward=True):
    stats_dev = torch.device(stats_device)

    if is_forward:
        x = inputs[0].detach().flatten(0, -2).float()
        out_key = "i_norm"
    else:
        x = outputs[0].detach().flatten(0, -2).float() * GRAD_SCALE_FACTOR
        out_key = "o_norm"

    thresh = thresholds["i" if is_forward else "o"].get(layer_name, 1e9)
    x_clipped = _clip_tokens(x, thresh)

    update = x_clipped.square().mean(dim=0)
    if not is_forward:
        update = update / GRAD_SCALE_FACTOR

    if update.device != stats_dev:
        update = update.to(stats_dev)
    stats_dict[out_key][layer_name].add_(update)


def _online_clipping_hook(module, inputs, outputs, layer_name, stats_dict, run_states, stats_device, is_forward=True):
    stats_dev = torch.device(stats_device)

    if is_forward:
        x = inputs[0].detach().flatten(0, -2).float()
        key = "i_norm"
    else:
        x = outputs[0].detach().flatten(0, -2).float() * GRAD_SCALE_FACTOR
        key = "o_norm"

    norms = torch.norm(x, dim=1, keepdim=True)
    tau = _get_robust_batch_tau(norms, PERCENTILE)

    state = run_states[layer_name][key]
    gmax = state["global_max"]

    if gmax is None:
        gmax = tau
    elif tau > gmax:
        correction = (tau / (gmax + 1e-8)).square()
        stats_dict[key][layer_name].mul_(correction.to(stats_dev))
        gmax = tau

    state["global_max"] = gmax

    update = _clip_tokens(x, gmax).square().mean(dim=0)

    if not is_forward:
        update /= GRAD_SCALE_FACTOR

    stats_dict[key][layer_name].add_(update.to(stats_dev))


# -----------------------------------------------------------------------------
# Hook Functions: DBF Strategy
# -----------------------------------------------------------------------------
def _dbf_hook(module, inputs, outputs, layer_name, stats_dict, stats_device, is_forward=True):
    stats_dev = torch.device(stats_device)

    if is_forward:
        x = inputs[0].detach().flatten(0, -2).float()
        out_key = "i_norm"
    else:
        x = outputs[0].detach().flatten(0, -2).float()
        out_key = "o_norm"

    update = x.square().mean(dim=0)
    if not is_forward:
        update *= GRAD_SCALE_FACTOR

    if update.device != stats_dev:
        update = update.to(stats_dev)
    stats_dict[out_key][layer_name].add_(update)


# -----------------------------------------------------------------------------
# Kronecker-factored curvature: nearest Kronecker product of the per-token empirical Fisher
# -----------------------------------------------------------------------------
@torch.no_grad()
def nkp_update(
    x: torch.Tensor,
    delta: torch.Tensor,
    L_prev: torch.Tensor | None = None,
    R_prev: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Un-normalised alternating-least-squares contribution of a token batch to the Kronecker factors.

    For per-token gradients ``G_t = delta_t x_t^T`` the empirical Fisher is
    ``F = sum_t vec(G_t) vec(G_t)^T``. Minimising ``||F - L kron R||_F`` in one factor while the other
    is held at its previous-pass value gives weighted covariances

    ``L = sum_t (x_t^T R_prev x_t) delta_t delta_t^T``,  ``R = sum_t (delta_t^T L_prev delta_t) x_t x_t^T``.

    With ``L_prev = R_prev = None`` (identity) this is exactly the Shampoo pair
    ``sum_t G_t G_t^T`` and ``sum_t G_t^T G_t``.

    Parameters
    ----------
    x : torch.Tensor
        Layer inputs, shape ``(tokens, in_features)``.
    delta : torch.Tensor
        Output gradients, shape ``(tokens, out_features)``.
    L_prev : torch.Tensor, optional
        Previous-pass output-side factor ``(out, out)``; ``None`` means identity.
    R_prev : torch.Tensor, optional
        Previous-pass input-side factor ``(in, in)``; ``None`` means identity.

    Returns
    -------
    tuple of torch.Tensor
        ``(L, R)`` symmetric PSD contributions of shapes ``(out, out)`` and ``(in, in)``.
    """
    x = x.float()
    delta = delta.float()
    if R_prev is None:
        w_L = x.square().sum(dim=1)
    else:
        w_L = ((x @ R_prev.to(x.device, torch.float32)) * x).sum(dim=1)
    if L_prev is None:
        w_R = delta.square().sum(dim=1)
    else:
        w_R = ((delta @ L_prev.to(delta.device, torch.float32)) * delta).sum(dim=1)

    L = delta.mT @ (delta * w_L.unsqueeze(1))
    R = x.mT @ (x * w_R.unsqueeze(1))
    return 0.5 * (L + L.mT), 0.5 * (R + R.mT)


def _frobenius_normalize(M: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """Return ``M`` scaled to unit Frobenius norm (a zero matrix is returned unchanged)."""
    return M / M.norm().clamp(min=eps)


@torch.no_grad()
def nkp_fit(x: torch.Tensor, delta: torch.Tensor, num_iters: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """In-memory reference of the multi-pass (Jacobi) nearest-Kronecker-product fit.

    Both factors are updated in every pass from the previous pass's factors, starting from identity,
    and each is normalised to unit Frobenius norm after every pass. This mirrors exactly what the
    streaming calibration hooks compute over the calibration set.

    Parameters
    ----------
    x, delta : torch.Tensor
        All tokens' inputs ``(tokens, in)`` and output gradients ``(tokens, out)``.
    num_iters : int
        Number of passes.

    Returns
    -------
    tuple of torch.Tensor
        Unit-Frobenius-norm factors ``(L, R)``.
    """
    L_prev = R_prev = None
    for _ in range(num_iters):
        L, R = nkp_update(x, delta, L_prev, R_prev)
        L_prev, R_prev = _frobenius_normalize(L), _frobenius_normalize(R)
    return L_prev, R_prev


def _factor_bytes(m: nn.Linear) -> int:
    """fp32 bytes of one layer's Kronecker factors for the accumulator *and* the previous-pass copy."""
    return 4 * 2 * (m.in_features**2 + m.out_features**2)


def _partition_layers(linear_layers: dict[str, nn.Linear], budget_bytes: int) -> list[list[str]]:
    """Greedy contiguous grouping of layers so that each group's factors fit within ``budget_bytes``.

    A layer larger than the budget forms its own group. Layer order is preserved.

    Parameters
    ----------
    linear_layers : dict
        ``name -> nn.Linear`` in model order.
    budget_bytes : int
        Memory budget for one group's accumulators plus previous-pass factors.

    Returns
    -------
    list of list of str
    """
    groups: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for n, m in linear_layers.items():
        b = _factor_bytes(m)
        if cur and cur_bytes + b > budget_bytes:
            groups.append(cur)
            cur, cur_bytes = [], 0
        cur.append(n)
        cur_bytes += b
    if cur:
        groups.append(cur)
    return groups


def _kron_clip(x: torch.Tensor, layer_name: str, side: str, clip_state, acc: dict) -> torch.Tensor:
    """Robust per-token clipping for the Kronecker path.

    Pass 1 behaves like the ``online`` strategy (running robust max with retroactive rescaling of the
    accumulators); later passes reuse the frozen pass-1 thresholds. Because ``G_t`` is linear in both
    ``x_t`` and ``delta_t``, a threshold change rescales *both* dense accumulators by ``c**2``.
    """
    if clip_state is None:
        return x
    state = clip_state[layer_name][side]
    gmax = state["global_max"]
    if state["frozen"]:
        return _clip_tokens(x, gmax) if gmax is not None else x

    tau = _get_robust_batch_tau(torch.norm(x, dim=1, keepdim=True), PERCENTILE)
    if gmax is None:
        gmax = tau
    elif tau > gmax:
        correction = (tau / (gmax + 1e-8)).square()
        for key in ("i_cov", "o_cov"):
            acc[key][layer_name].mul_(correction.to(acc[key][layer_name].device))
        gmax = tau
    state["global_max"] = gmax
    return _clip_tokens(x, gmax)


def _kron_forward_hook(module, inputs, outputs, layer_name, run_states, clip_state, acc):
    """Stash the (clipped) layer input for the matching backward hook.

    Only runs when autograd is enabled, i.e. in the gradient-checkpoint recompute that immediately
    precedes the backward pass (or in a plain forward when checkpointing is off), so the stash is
    consumed right away and never holds every layer's activations at once.
    """
    if not torch.is_grad_enabled():
        return
    x = inputs[0].detach().flatten(0, -2).float()
    x = _kron_clip(x, layer_name, "i", clip_state, acc)
    run_states[layer_name]["x"] = x


def _kron_backward_hook(module, grad_input, grad_output, layer_name, run_states, clip_state, acc, prev, sq_sums):
    """Accumulate one Jacobi ALS contribution ``nkp_update(x, delta, L_prev, R_prev)`` for this layer."""
    x = run_states[layer_name].pop("x", None)
    if x is None:
        return
    delta = grad_output[0].detach().flatten(0, -2).float() * GRAD_SCALE_FACTOR
    delta = _kron_clip(delta, layer_name, "o", clip_state, acc)

    L_prev = prev["o_cov"].get(layer_name)
    R_prev = prev["i_cov"].get(layer_name)
    L, R = nkp_update(x, delta, L_prev, R_prev)

    acc_L = acc["o_cov"][layer_name]
    acc_R = acc["i_cov"][layer_name]
    acc_L.add_(L.to(acc_L.device))
    acc_R.add_(R.to(acc_R.device))
    # running mean-square statistics used to put the factors on the legacy i_norm / o_norm scale
    sq_sums[layer_name]["i"] += x.square().mean().item()
    sq_sums[layer_name]["o"] += delta.square().mean().item() / GRAD_SCALE_FACTOR
    sq_sums[layer_name]["n"] += 1


def _collect_kron_stats(model, dataloader, dev, linear_layers: dict[str, nn.Linear], strategy: str, nkp_iters: int,
                        stats_device: str, use_truefisher: bool, model_offload: bool,
                        gpu_budget_gb: float = 0.0) -> dict:
    """Multi-pass streaming estimate of the nearest Kronecker product of the per-token empirical Fisher.

    Every ALS iteration updates **both** factors of every layer from the previous iteration's
    (unit-Frobenius-norm) factors; iteration 1 starts from identity. After the final iteration the
    factors are rescaled so that their mean diagonals match the legacy per-token second moments, which
    keeps ``i_norm``/``o_norm`` (their diagonals) on the familiar scale for downstream users.

    Memory/traffic strategy. With ``gpu_budget_gb <= 0`` every layer's accumulator lives on
    ``stats_device`` (CPU) and each token batch's contribution is transferred there per hook call — for
    large models this moves tens of GB per calibration sample and dominates run time. With a positive
    budget the layers are partitioned into groups whose accumulators and previous-pass factors fit within
    the budget on the compute device; each iteration then runs one calibration pass **per group** with
    everything resident on ``dev`` and flushes the finished factors to ``stats_device``. Because the
    previous-pass factors are fixed within an iteration, the result is identical to the single-pass
    computation.
    """
    if nkp_iters < 1:
        raise ValueError("nkp_iters must be >= 1")
    clip_state = None
    if strategy != "dbf":
        clip_state = defaultdict(lambda: {
            "i": {"global_max": None, "frozen": False},
            "o": {"global_max": None, "frozen": False},
        })

    if gpu_budget_gb > 0:
        groups = _partition_layers(linear_layers, int(gpu_budget_gb * 1e9))
        acc_device = dev
    else:
        groups = [list(linear_layers)]
        acc_device = stats_device

    prev = {"i_cov": {}, "o_cov": {}}
    sq_sums = None
    for it in range(nkp_iters):
        sq_sums = defaultdict(lambda: {"i": 0.0, "o": 0.0, "n": 0})
        new = {"i_cov": {}, "o_cov": {}}
        for g, names in enumerate(groups):
            group_bytes = sum(_factor_bytes(linear_layers[n]) for n in names)
            print(f">>> Kronecker curvature: NKP pass {it + 1}/{nkp_iters}, layer group {g + 1}/{len(groups)} "
                  f"({len(names)} layers, {group_bytes / 2**30:.1f} GiB on {acc_device})")
            acc = {
                "i_cov": {
                    n: torch.zeros(linear_layers[n].in_features, linear_layers[n].in_features, dtype=torch.float32,
                                   device=acc_device)
                    for n in names
                },
                "o_cov": {
                    n: torch.zeros(linear_layers[n].out_features, linear_layers[n].out_features,
                                   dtype=torch.float32, device=acc_device)
                    for n in names
                },
            }
            prev_group = {
                key: {n: prev[key][n].to(acc_device) for n in names if n in prev[key]}
                for key in ("i_cov", "o_cov")
            }
            run_states = defaultdict(dict)
            handles = []
            for n in names:
                m = linear_layers[n]
                handles.append(
                    m.register_forward_hook(
                        partial(_kron_forward_hook, layer_name=n, run_states=run_states, clip_state=clip_state,
                                acc=acc)))
                handles.append(
                    m.register_full_backward_hook(
                        partial(_kron_backward_hook, layer_name=n, run_states=run_states, clip_state=clip_state,
                                acc=acc, prev=prev_group, sq_sums=sq_sums)))

            _run_calibration_loop(dataloader, model, dev, model_offload, use_truefisher)

            for h in handles:
                h.remove()
            for key in ("i_cov", "o_cov"):
                for n in names:
                    new[key][n] = _frobenius_normalize(acc[key][n]).to(stats_device)
            del acc, prev_group
            if torch.cuda.is_available():
                cleanup_memory()

        if clip_state is not None:
            for states in clip_state.values():
                states["i"]["frozen"] = True
                states["o"]["frozen"] = True
        prev = new

    # Put the final (unit-norm) factors on the legacy scale: mean diag(R) = E_t[x^2], mean diag(L) = E_t[delta^2]
    for n in linear_layers:
        stats = sq_sums[n]
        if stats["n"] == 0:
            continue
        for key, side in (("i_cov", "i"), ("o_cov", "o")):
            M = prev[key][n]
            mean_diag = M.diagonal().mean()
            target = stats[side] / stats["n"]
            if mean_diag > 0 and target > 0:
                M.mul_(target / mean_diag)
    return prev


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def _run_calibration_loop(dataloader, model, dev, model_offload, use_truefisher):
    """Common calibration loop used by all strategies."""
    def _to_batch(x):
        # preserves your behavior (you used unsqueeze(0) assuming per-sample tensors)
        return x.unsqueeze(0) if x.dim() == 1 else x

    for batch in tqdm(dataloader):
        batch_dev = _to_batch(batch.to(dev, non_blocking=True))
        hidden = _get_last_hidden_state(model, batch_dev, model_offload)
        loss = _calculate_loss(None if model_offload else hidden, hidden if model_offload else None, model, batch_dev,
                               use_truefisher, model_offload)
        loss.backward()
        model.zero_grad(set_to_none=True)


def _get_last_hidden_state(model, batch, model_offload):
    if model.config.model_type == "opt":
        attention_mask = (batch != model.config.pad_token_id).long()
        return model(input_ids=batch, attention_mask=attention_mask).logits if model_offload else \
               model.model.decoder(input_ids=batch, attention_mask=attention_mask)[0]
    return model(batch).logits if model_offload else model.model(batch)[0]


def _calculate_loss(embs, lm_logits, model, batch, use_truefisher, model_offload):
    if use_truefisher:
        from cut_cross_entropy import linear_cross_entropy
        with torch.inference_mode():
            step = 1024
            labels = [
                torch.multinomial(torch.softmax(model.lm_head(embs[:, i:i + step]), dim=-1)[0], 1).reshape(1, -1)
                for i in range(0, embs.shape[1], step)
            ]
            labels = torch.cat(labels, dim=1)
        return linear_cross_entropy(embs, model.lm_head.weight, labels)

    if model_offload:
        return F.cross_entropy(
            lm_logits[:, :-1, :].reshape(-1, lm_logits.size(-1)),
            batch[:, 1:].to(lm_logits.device).reshape(-1),
        )
    from cut_cross_entropy import linear_cross_entropy
    return linear_cross_entropy(embs, model.lm_head.weight, batch.to(embs.device), shift=1)


def get_shrunk_stats(raw_stats: dict, shrinkage: float = 0.0) -> dict:
    """
    Creates a new statistics dictionary with covariance shrinkage applied.

    Diagonal statistics (``i_norm``/``o_norm``) are shrunk toward their mean,
    ``H_new = (1 - shrinkage) * H + shrinkage * mean(H)``. Dense Kronecker factors
    (``i_cov``/``o_cov``, when present) are shrunk toward the scaled identity
    ``(1 - shrinkage) * C + shrinkage * mean(diag(C)) * I``, whose diagonal is exactly the vector
    formula above; the diagonal statistics are then re-derived from the shrunk dense factors so that
    the weight preconditioner is the diagonal of the dense curvature.

    Args:
        raw_stats (dict): Raw calibration statistics from `collect_stats()`
        shrinkage (float): Shrinkage strength (0.0 = no shrinkage, 1.0 = full shrinkage)

    Returns:
        dict: A new dictionary containing the shrunk statistics.
    """
    # 1. Create a deep copy to prevent "Double Dipping" or polluting the raw data.
    #    We use .clone() on tensors to allocate new memory (approx. 7MB total, negligible).
    shrunk_stats = {
        'i_norm': {
            k: v.clone()
            for k, v in raw_stats['i_norm'].items()
        },
        'o_norm': {
            k: v.clone()
            for k, v in raw_stats['o_norm'].items()
        },
        'stats_device': raw_stats.get('stats_device', 'cpu')
    }
    has_dense = 'i_cov' in raw_stats and 'o_cov' in raw_stats
    if has_dense:
        for key in ('i_cov', 'o_cov'):
            shrunk_stats[key] = {k: v.clone() for k, v in raw_stats[key].items()}

    apply = 0.0 < shrinkage < 1.0
    if apply:
        print(f"Applying covariance shrinkage (strength: {shrinkage})...")

    if has_dense:
        for key_cov, key_vec in (('i_cov', 'i_norm'), ('o_cov', 'o_norm')):
            for layer_name, cov in shrunk_stats[key_cov].items():
                if cov.numel() == 0:
                    continue
                if apply:
                    mean_diag = cov.diagonal().mean()
                    cov.mul_(1.0 - shrinkage)
                    cov.diagonal().add_(mean_diag * shrinkage)
                shrunk_stats[key_vec][layer_name] = cov.diagonal().clone()
        return shrunk_stats

    # 2. Return early if no shrinkage is needed (just return the copy).
    if not apply:
        return shrunk_stats

    # 3. Apply the shrinkage formula to the copied tensors.
    #    H_new = (1 - shrinkage) * H + shrinkage * mean(H)
    for key in ['i_norm', 'o_norm']:
        for layer_name, tensor in shrunk_stats[key].items():
            if tensor.numel() == 0:
                continue

            mean_val = tensor.mean()
            # In-place modification is safe here because 'tensor' is already a clone.
            tensor.mul_(1.0 - shrinkage).add_(mean_val * shrinkage)

    return shrunk_stats


def register_stats(model, stats: dict):
    """
    Registers the calibration statistics (i_norm, o_norm and, when present, the dense
    Kronecker factors i_cov, o_cov) as non-persistent buffers on every linear layer.

    Args:
        model (nn.Module): The PyTorch model to modify.
        stats (dict): The dictionary containing processed 'i_norm' and 'o_norm' (and optional 'i_cov'/'o_cov').

    Returns:
        model: The model with registered buffers.
    """
    # 1. Identify target linear layers (excluding the LM head).
    linear_layers = {name: m for name, m in model.named_modules() if isinstance(m, nn.Linear) and "lm_head" not in name}

    device = stats.get('stats_device', 'cpu')

    print(f"Attaching statistics to {len(linear_layers)} layers...")

    for name, m in linear_layers.items():
        # 2. Register Input Norm (i_norm)
        if name in stats['i_norm']:
            # persistent=False means these buffers won't be saved in the model's state_dict
            # (checkpoints), keeping the file size small.
            m.register_buffer("i_norm", stats['i_norm'][name], persistent=False)
        else:
            # Fallback: If stats are missing, register a tensor of ones (Identity op).
            m.register_buffer("i_norm", torch.ones(m.weight.shape[1], device=device), persistent=False)
        # 3. Register Output Norm (o_norm)
        if name in stats['o_norm']:
            m.register_buffer("o_norm", stats['o_norm'][name], persistent=False)
        else:
            m.register_buffer("o_norm", torch.ones(m.weight.shape[0], device=device), persistent=False)
        # 4. Dense Kronecker factors (only when the kron curvature was collected for this layer)
        for key in ("i_cov", "o_cov"):
            if key in stats and name in stats[key]:
                m.register_buffer(key, stats[key][name], persistent=False)

    dense_note = " (+ dense i_cov/o_cov)" if "i_cov" in stats else ""
    print(f"Registered i_norm and o_norm buffers{dense_note} to {len(linear_layers)} layers")
    return model


# -----------------------------------------------------------------------------
# MAIN CALIBRATION FUNCTION
# -----------------------------------------------------------------------------
def collect_stats(model, dataloader, dev, use_truefisher=False, model_offload=False, vram_limit_gb=50, save_plots=False,
                  strategy='online', curvature='diag', nkp_iters=3, stats_device=None, gpu_budget_gb=0.0):
    """
    Main entry point for NanoQuant calibration statistics collection.
    Collects raw calibration statistics without applying shrinkage.

    Parameters
    ----------
    curvature : {"diag", "kron"}
        ``"diag"`` (legacy) collects per-feature second moments ``i_norm``/``o_norm``.
        ``"kron"`` additionally collects dense Kronecker factors ``i_cov`` (in x in) and ``o_cov``
        (out x out) as the nearest Kronecker product of the per-token empirical Fisher, fitted by
        ``nkp_iters`` alternating passes over the calibration data; ``i_norm``/``o_norm`` are then
        their diagonals. The clipping behaviour follows ``strategy`` (``"dbf"`` = no clipping).
    nkp_iters : int
        Number of ALS iterations for ``curvature="kron"``.
    stats_device : str, optional
        Where the returned statistics live. Defaults to ``dev`` (or CPU when offloading); the dense
        ``kron`` statistics default to CPU because they are large.
    gpu_budget_gb : float
        For ``curvature="kron"``: accumulate on ``dev`` for groups of layers whose factors fit within
        this budget (one calibration pass per group and iteration), instead of streaming every
        contribution to ``stats_device``. ``0`` keeps the legacy per-hook CPU accumulation.
    """
    if curvature not in CURVATURE_TYPES:
        raise ValueError(f"Unknown curvature '{curvature}'. Choose from {CURVATURE_TYPES}.")
    if stats_device is None:
        stats_device = 'cpu' if (model_offload or curvature == 'kron') else dev
    print(f"Calibration Strategy: {strategy.upper()} | Curvature: {curvature.upper()} | "
          f"Offload: {model_offload} (Limit: {vram_limit_gb}GB)")

    linear_layers = {name: m for name, m in model.named_modules() if isinstance(m, nn.Linear) and "lm_head" not in name}

    if not model_offload:
        model.to(dev)

    model.train()
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        if curvature == 'kron':
            # The kron hooks rely on the re-entrant recompute (forward hook fires right before backward).
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            except TypeError:
                model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()

    # =========================================================
    # CURVATURE: KRON (multi-pass nearest Kronecker product)
    # =========================================================
    if curvature == 'kron':
        if strategy not in ('online', 'two_phase', 'dbf'):
            raise ValueError(f"Unknown strategy: {strategy}")
        factors = _collect_kron_stats(model, dataloader, dev, linear_layers, strategy, nkp_iters, stats_device,
                                      use_truefisher, model_offload, gpu_budget_gb=gpu_budget_gb)
        raw_stats = {
            'i_norm': {n: factors['i_cov'][n].diagonal().clone() for n in linear_layers},
            'o_norm': {n: factors['o_cov'][n].diagonal().clone() for n in linear_layers},
            'i_cov': factors['i_cov'],
            'o_cov': factors['o_cov'],
            'n_samples': len(dataloader),
            'stats_device': stats_device,
        }
        return raw_stats

    stats = {
        'i_norm': defaultdict(lambda: torch.zeros(0, dtype=torch.float32, device=stats_device)),
        'o_norm': defaultdict(lambda: torch.zeros(0, dtype=torch.float32, device=stats_device)),
    }
    handles = []

    # =========================================================
    # STRATEGY: TWO-PHASE (Robust Profiling + Fixed Clipping)
    # =========================================================
    if strategy == 'two_phase':
        print(">>> Phase 1: Robust Profiling (Percentile-based Tau discovery)...")

        global_profiling = {"i_max": {}, "o_max": {}}

        for n, m in linear_layers.items():
            handles.append(
                m.register_forward_hook(
                    partial(_phase1_robust_profiling_hook, layer_name=n, global_stats=global_profiling,
                            is_forward=True)))
            handles.append(
                m.register_full_backward_hook(
                    partial(_phase1_robust_profiling_hook, layer_name=n, global_stats=global_profiling,
                            is_forward=False)))

        _run_calibration_loop(dataloader, model, dev, model_offload, use_truefisher)

        for h in handles:
            h.remove()
        handles = []

        thresholds = {
            'i': {
                n: v.item()
                for n, v in global_profiling["i_max"].items()
            },
            'o': {
                n: v.item()
                for n, v in global_profiling["o_max"].items()
            },
        }
        del global_profiling

        print(">>> Phase 2: Sanitized Calibration...")

        for n, m in linear_layers.items():
            stats['i_norm'][n] = torch.zeros(m.weight.shape[1], device=stats_device)
            stats['o_norm'][n] = torch.zeros(m.weight.shape[0], device=stats_device)
            handles.append(
                m.register_forward_hook(
                    partial(_fixed_clipping_hook, layer_name=n, stats_dict=stats, thresholds=thresholds,
                            stats_device=stats_device, is_forward=True)))
            handles.append(
                m.register_full_backward_hook(
                    partial(_fixed_clipping_hook, layer_name=n, stats_dict=stats, thresholds=thresholds,
                            stats_device=stats_device, is_forward=False)))

    # =========================================================
    # STRATEGY: ONLINE (Cumulative Monotonic Update)
    # =========================================================
    elif strategy == 'online':
        print(">>> Single Pass: Online Cumulative Preconditioning...")

        run_states = defaultdict(lambda: {'i_norm': {'global_max': None}, 'o_norm': {'global_max': None}})

        for n, m in linear_layers.items():
            stats['i_norm'][n] = torch.zeros(m.weight.shape[1], device=stats_device)
            stats['o_norm'][n] = torch.zeros(m.weight.shape[0], device=stats_device)
            handles.append(
                m.register_forward_hook(
                    partial(_online_clipping_hook, layer_name=n, stats_dict=stats, run_states=run_states,
                            stats_device=stats_device, is_forward=True)))
            handles.append(
                m.register_full_backward_hook(
                    partial(_online_clipping_hook, layer_name=n, stats_dict=stats, run_states=run_states,
                            stats_device=stats_device, is_forward=False)))

    # =========================================================
    # STRATEGY: DBF (Direct Buffer Accumulation)
    # =========================================================
    elif strategy == 'dbf':
        print(">>> DBF Strategy: Direct Accumulation...")

        for n, m in linear_layers.items():
            stats['i_norm'][n] = torch.zeros(m.weight.shape[1], dtype=torch.float32, device=stats_device)
            stats['o_norm'][n] = torch.zeros(m.weight.shape[0], dtype=torch.float32, device=stats_device)

            handles.append(
                m.register_forward_hook(
                    partial(_dbf_hook, layer_name=n, stats_dict=stats, stats_device=stats_device, is_forward=True)))
            handles.append(
                m.register_full_backward_hook(
                    partial(_dbf_hook, layer_name=n, stats_dict=stats, stats_device=stats_device, is_forward=False)))

    else:
        for h in handles:
            h.remove()
        raise ValueError(f"Unknown strategy: {strategy}")

    # =========================================================
    # Common calibration loop for all strategies
    # =========================================================
    _run_calibration_loop(dataloader, model, dev, model_offload, use_truefisher)

    # =========================================================
    # Finalization
    # =========================================================
    for h in handles:
        h.remove()

    if torch.cuda.is_available():
        cleanup_memory()

    print("Finalizing raw statistics...")

    n_samples = len(dataloader)

    # Create a copy of raw statistics to return
    raw_stats = {'i_norm': {}, 'o_norm': {}, 'n_samples': n_samples, 'stats_device': stats_device}

    for name, m in linear_layers.items():
        if name in stats['i_norm']:
            raw_stats['i_norm'][name] = stats['i_norm'][name]
            if strategy != "dbf":
                raw_stats['i_norm'][name] /= n_samples
        else:
            raw_stats['i_norm'][name] = torch.ones(m.weight.shape[1], device=stats_device)

        if name in stats['o_norm']:
            raw_stats['o_norm'][name] = stats['o_norm'][name]
            if strategy != "dbf":
                raw_stats['o_norm'][name] /= n_samples
        else:
            raw_stats['o_norm'][name] = torch.ones(m.weight.shape[0], device=stats_device)

    del stats
    return raw_stats
