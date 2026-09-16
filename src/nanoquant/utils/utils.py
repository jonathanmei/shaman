# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import gc
import inspect
import os
import random

import numpy as np
import torch
from torch import nn


def set_seed(seed, use_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if use_deterministic:
        if torch.cuda.is_available():
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def cleanup_memory(verbose=False) -> None:
    caller_name = ""
    try:
        caller_name = f" (from {inspect.stack()[1].function})"
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    del_vars = [k for k in list(globals().keys()) if k.startswith("_tmp_")]
    for k in del_vars:
        globals().pop(k, None)
    gc.collect()

    if torch.cuda.is_available():
        # https://discuss.pytorch.org/t/how-to-delete-a-tensor-in-gpu-to-free-up-memory/48879/33
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        memory_after = total_reserved_mem()
        if verbose:
            print(f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GiB"
                  f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GiB)")


def stage_devices(quant_config: dict, dev) -> list[str]:
    """Devices for the layer-sharded stages (rank probe, calibration layer groups).

    Parameters
    ----------
    quant_config : dict
        ``parallel_devices``: number of GPUs to use; ``0`` = every visible one, ``1`` = the single ``dev``.
    dev : str or torch.device
        The pipeline's compute device.

    Returns
    -------
    list of str
        ``[dev]`` on CPU or with one usable device, else ``["cuda:0", ..., "cuda:G-1"]``.
    """
    dev = str(dev)
    if torch.device(dev).type != "cuda":
        return [dev]
    want = int(quant_config.get("parallel_devices", 0) or 0)
    avail = torch.cuda.device_count()
    n = avail if want <= 0 else min(want, avail)
    if n <= 1:
        return [dev]
    return [f"cuda:{i}" for i in range(n)]


def device_context(device):
    """Context manager making ``device`` the thread-local current CUDA device (a no-op for CPU devices).

    Worker threads start with device 0 as their current device; kernels that pick the device from that thread-local
    state (Triton launchers, RNG forks in checkpoint recompute) must run under this context when their tensors live
    on another GPU.

    Parameters
    ----------
    device : str or torch.device
        Device the thread works on.

    Returns
    -------
    context manager
    """
    d = torch.device(str(device))
    if d.type != "cuda":
        return contextlib.nullcontext()
    return torch.cuda.device(d.index if d.index is not None else torch.cuda.current_device())


def admm_side_device(quant_config: dict, device) -> str | None:
    """Second device for the B side of the two-device ADMM (``admm_parallel_sides``), or ``None`` for the serial path.

    Parameters
    ----------
    quant_config : dict
        ``admm_parallel_sides``: run each ADMM iteration's B-update on another GPU.
    device : str or torch.device
        Device of the A side (the weight's device).

    Returns
    -------
    str or None
        The neighbouring CUDA device (``cuda:1`` for ``cuda:0``) when the knob is on and at least two devices are
        visible; ``None`` otherwise (CPU, single GPU, or knob off).
    """
    if not quant_config.get("admm_parallel_sides", False):
        return None
    d = torch.device(str(device))
    if d.type != "cuda" or torch.cuda.device_count() < 2:
        return None
    if d.index is not None:
        idx = d.index
    else:
        idx = torch.cuda.current_device() if torch.cuda.is_initialized() else 0
    return f"cuda:{(idx + 1) % torch.cuda.device_count()}"


def find_layers(module, layers=None, name=''):
    """
    Recursively finds all instances of specified layers in a module.
    """
    if layers is None:
        layers = [nn.Linear]

    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def get_layers_to_factorize(model_type: str) -> list[str]:
    """Returns a list of sublayers to factorize based on the model architecture."""
    if model_type in ["llama", "mistral", "mixtral", "mobilellm", "qwen3"] or model_type.startswith("gemma"):
        ret = [
            'self_attn.q_proj',
            'self_attn.v_proj',
            'self_attn.o_proj',
            'self_attn.k_proj',
            'mlp.gate_proj',
            'mlp.up_proj',
            'mlp.down_proj',
        ]
    elif model_type == "opt":
        ret = [
            'self_attn.q_proj',
            'self_attn.v_proj',
            'self_attn.out_proj',
            'self_attn.k_proj',
            'fc1',
            'fc2',
        ]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return ret


def get_decoder_layers(model):
    """
    Returns the list of decoder layers based on the model architecture.
    """
    model_type = model.config.model_type
    if model_type in ["llama", "mistral", "mixtral", "mobilellm", "qwen3"] or model_type.startswith("gemma"):
        return model.model.layers
    elif model_type == "opt":
        return model.model.decoder.layers
    elif model_type == "gpt2":
        return model.transformer.h
    raise AttributeError(f"Could not find decoder layers for model architecture '{model_type}'.")


def get_decoder_layer_cls_name(model: nn.Module) -> list[str]:
    """Helper to get the class name of the decoder blocks (to prevent accelerate from splitting blocks)."""
    try:
        layers = get_decoder_layers(model)
        if layers:
            return [layers[0].__class__.__name__]
    except AttributeError:
        pass
    return []


def has_mid_scale(quant_config) -> bool:
    """Whether the factorisation exports a per-rank middle scale (``scale_mid``).

    Parameters
    ----------
    quant_config : dict
        Quantisation configuration. The ``dbf`` ADMM always has a middle scale; the ``nanoquant``
        ADMM has one when ``admm_mid_scale`` is set. Configs predating the flag are treated as ``False``.

    Returns
    -------
    bool
    """
    return quant_config.get('admm_type') == 'dbf' or bool(quant_config.get('admm_mid_scale', False))


SCALE_BITS = 16
RANK_STEP = 32
RANK_BUDGETS = ("uniform", "parity", "full")
# how the per-layer bit budget is set under a non-uniform budget: a sensitivity curve measured at calibration time
# by short ADMM solves ("admm") or a whitened-SVD proxy ("svd"); "none" = the uniform rule only
RANK_SENSITIVITIES = ("none", "admm", "svd")
# power-law exponent of the fitted sensitivity curve J(r) = exp(a) r^-beta: clamp range and single-probe default
BETA_RANGE = (0.05, 8.0)
BETA_DEFAULT = 1.0


def layer_bits(in_features: int, out_features: int, rank: int, num_scales: int) -> int:
    """Storage bits of one factorised layer: ``rank (in + out)`` binary entries plus 16-bit scales.

    Parameters
    ----------
    in_features, out_features : int
        Layer shape.
    rank : int
        Factorisation rank.
    num_scales : int
        2 (pre, post) or 3 (pre, mid, post).

    Returns
    -------
    int
    """
    scale_entries = in_features + out_features + (rank if num_scales == 3 else 0)
    return rank * (in_features + out_features) + SCALE_BITS * scale_entries


def _continuous_rank(a: int, b: int, bits: float, num_scales: int = 2):
    """Rank at which a ``(a x b)`` layer with ``num_scales`` 16-bit scale vectors costs exactly ``bits`` per weight.

    Standard (2 scales: pre, post):  bits = [rank (a + b) + 16 (a + b)] / (a b)
    Middle scale (3 scales):         bits = [rank (a + b) + 16 (a + b + rank)] / (a b)
    """
    if bits is None or a * b == 0:
        return None
    total_budget_bits = a * b * bits
    param_sum = a + b
    if num_scales == 3:
        return (total_budget_bits - SCALE_BITS * param_sum) / (param_sum + SCALE_BITS)
    return (total_budget_bits / param_sum) - SCALE_BITS


def _floor_rank(rank, in_features: int, out_features: int, min_rank: int = RANK_STEP,
                max_rank: int | None = None) -> int:
    """Legacy rounding: floor to a multiple of 32, at least ``min_rank``, at most ``max_rank`` (default ``min(in, out)``)."""
    curr_rank = int(rank) if rank is not None else 0
    curr_rank = (curr_rank // RANK_STEP) * RANK_STEP
    curr_rank = max(curr_rank, min_rank)
    return min(curr_rank, min(in_features, out_features) if max_rank is None else max_rank)


def uniform_rank(in_features: int, out_features: int, bits: float, num_scales: int) -> int:
    """Rank of the paper's uniform rule: the bit-exact continuous rank floored to a multiple of 32 (at least 32).

    Parameters
    ----------
    in_features, out_features : int
        Layer shape.
    bits : float
        Target bits per weight.
    num_scales : int
        2 or 3.

    Returns
    -------
    int
    """
    return _floor_rank(_continuous_rank(in_features, out_features, bits, num_scales), in_features, out_features)


def rank_ceiling(in_features: int, out_features: int) -> int:
    """Largest rank a layer may receive: ``min(in, out)`` (beyond it the sign product's real rank saturates)."""
    return min(in_features, out_features)


def parse_probe_ranks(spec: str) -> list[float]:
    """Parse ``"0.5,1.0,1.5"`` into the list of multiples of the uniform rank at which a layer is probed.

    Raises
    ------
    ValueError
        On a malformed or non-positive entry, or an empty list.
    """
    values: list[float] = []
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            v = float(item)
        except ValueError as e:
            raise ValueError(f"rank_probe_ranks entry '{item}' is not a number") from e
        if v <= 0:
            raise ValueError(f"rank_probe_ranks entry '{item}' must be > 0")
        values.append(v)
    if not values:
        raise ValueError("rank_probe_ranks must list at least one positive multiple of the uniform rank")
    return values


def fit_power_law(probes: dict[int, float]) -> tuple[float, float]:
    """Least-squares fit of ``log J = a - beta log r`` through the probed ``{rank: J}`` points.

    ``beta`` is clamped to :data:`BETA_RANGE` (a curve that rises with rank through noise gets the minimum
    slope, never a negative one) and the level ``a`` is re-fitted at the clamped slope. With a single probe
    the slope is :data:`BETA_DEFAULT`.

    Parameters
    ----------
    probes : dict
        ``rank -> J`` with ``J > 0`` (curvature-weighted weight error at that rank).

    Returns
    -------
    tuple
        ``(a, beta)`` such that the predicted loss is ``exp(a) * rank ** -beta``.

    Raises
    ------
    ValueError
        On an empty dict or a non-positive rank / loss.
    """
    pts = [(int(r), float(j)) for r, j in probes.items()]
    if not pts:
        raise ValueError("fit_power_law needs at least one probe")
    if any(r <= 0 or not (j > 0) for r, j in pts):
        raise ValueError(f"fit_power_law needs positive ranks and losses, got {pts}")
    x = np.log(np.array([r for r, _ in pts], dtype=np.float64))
    y = np.log(np.array([j for _, j in pts], dtype=np.float64))
    xm, ym = x.mean(), y.mean()
    var = float(((x - xm) ** 2).sum())
    if len(pts) == 1 or var == 0.0:
        beta = BETA_DEFAULT
    else:
        beta = -float(((x - xm) * (y - ym)).sum() / var)
    beta = min(max(beta, BETA_RANGE[0]), BETA_RANGE[1])
    a = float(ym + beta * xm)
    return a, float(beta)


def predicted_loss(curve: tuple[float, float], rank: int) -> float:
    """``exp(a) * rank ** -beta`` for a fitted ``(a, beta)`` curve."""
    a, beta = curve
    return float(np.exp(a - beta * np.log(rank)))


def depth_multipliers(shapes: dict[str, tuple[int, int]], n_blocks: int, depth_ramp: float) -> dict[str, float]:
    """Depth prior ``m_l = exp(ramp (b_l / (B - 1) - 1/2))`` per layer (block index ``b_l`` of ``B`` blocks).

    The ramp supplies the end-to-end depth weighting that the local Gauss-Newton sensitivity misses: late-block
    errors land on the logits through fewer layers of repair. ``ramp`` is the log-ratio of the last block's
    multiplier to the first block's (0 = flat).

    Parameters
    ----------
    shapes : dict
        ``"<block>.<name>" -> (in_features, out_features)``.
    n_blocks : int
        Number of decoder blocks.
    depth_ramp : float
        Log-ratio of the last block's multiplier to the first block's.
    """
    mult: dict[str, float] = {}
    for key in shapes:
        blk = int(key.split(".", 1)[0])
        mult[key] = float(np.exp(depth_ramp * (blk / (n_blocks - 1) - 0.5))) if n_blocks > 1 else 1.0
    return mult


def _step_bits(in_features: int, out_features: int, num_scales: int) -> int:
    """Bits added by one 32-rank step of a layer."""
    return RANK_STEP * (in_features + out_features) + (SCALE_BITS * RANK_STEP if num_scales == 3 else 0)


def _total_bit_target(shapes: dict[str, tuple[int, int]], bits: float, num_scales: int, budget: str,
                      legacy: dict[str, int]) -> int:
    """Total bits to spend: what the uniform rule spends after flooring (``parity``) or exactly ``bits`` per weight."""
    if budget == "parity":
        return sum(layer_bits(a, b, legacy[k], num_scales) for k, (a, b) in shapes.items())
    return int(bits * sum(a * b for a, b in shapes.values()))


def allocate_ranks_measured(shapes: dict[str, tuple[int, int]], curves: dict[str, tuple[float, float]], bits: float,
                            num_scales: int, budget: str, legacy: dict[str, int],
                            mult: dict[str, float] | None = None) -> dict[str, int]:
    """Rank allocation by marginal predicted loss per bit from measured per-layer sensitivity curves.

    Every layer starts at rank 32; one 32-step at a time is given to the layer with the largest predicted loss
    decrease per added bit, ``(J_l(r) - J_l(r + 32)) / step_bits_l``, until no step fits under the bit target
    (``budget="parity"``: the bits the uniform rule spends; ``"full"``: exactly ``bits`` per weight) or the
    ceiling ``min(in, out)``. For decreasing convex curves this greedy fill is the optimum of the discretised
    separable knapsack. Layers without a curve keep their uniform rank (their bits stay reserved).

    Parameters
    ----------
    shapes : dict
        ``"<block>.<name>" -> (in_features, out_features)``.
    curves : dict
        ``"<block>.<name>" -> (a, beta)`` from :func:`fit_power_law`; predicted loss ``exp(a) rank^-beta``.
    bits : float
        Target bits per weight.
    num_scales : int
        2 or 3.
    budget : str
        ``"parity"`` or ``"full"``.
    legacy : dict
        Ranks of the uniform rule (parity target and fallback).
    mult : dict, optional
        Multiplicative prior on each layer's curve (see :func:`depth_multipliers`).

    Returns
    -------
    dict
        ``"<block>.<name>" -> rank``.
    """
    if budget not in ("parity", "full"):
        raise ValueError(f"Unknown rank_budget for allocate_ranks_measured: {budget}")
    lo = RANK_STEP
    hi = {k: rank_ceiling(a, b) for k, (a, b) in shapes.items()}
    step_bits = {k: _step_bits(a, b, num_scales) for k, (a, b) in shapes.items()}
    total_target = _total_bit_target(shapes, bits, num_scales, budget, legacy)
    level = {k: (curves[k][0] + (float(np.log(mult[k])) if mult else 0.0), curves[k][1])
             for k in shapes if k in curves}
    ranks = {k: (lo if k in level else min(legacy[k], hi[k])) for k in shapes}
    total = sum(layer_bits(a, b, ranks[k], num_scales) for k, (a, b) in shapes.items())

    def gain(k: str) -> float:
        r = ranks[k]
        return (predicted_loss(level[k], r) - predicted_loss(level[k], r + RANK_STEP)) / step_bits[k]

    gains = {k: gain(k) for k in level}
    while True:
        cands = [k for k in level if ranks[k] + RANK_STEP <= hi[k] and total + step_bits[k] <= total_target]
        if not cands:
            break
        k = max(cands, key=gains.__getitem__)
        ranks[k] += RANK_STEP
        total += step_bits[k]
        gains[k] = gain(k)
    return ranks


def _format_measured_summary(shapes: dict[str, tuple[int, int]], ranks: dict[str, int], legacy: dict[str, int],
                             curves: dict[str, tuple[float, float]], num_scales: int) -> str:
    """Per-type and per-block summary of a measured allocation against the uniform rule."""
    by_type: dict[str, list[str]] = {}
    for k in shapes:
        by_type.setdefault(k.split(".", 1)[1].rsplit(".", 1)[-1], []).append(k)
    lines = ["Rank allocation (measured): type | median beta | mean rank measured / uniform"]
    for t, keys in by_type.items():
        betas = [curves[k][1] for k in keys if k in curves]
        med = float(np.median(betas)) if betas else float("nan")
        lines.append(f"    {t:<10s} | {med:6.2f} | {np.mean([ranks[k] for k in keys]):7.0f} / "
                     f"{np.mean([legacy[k] for k in keys]):5.0f}")
    blocks = sorted({int(k.split('.', 1)[0]) for k in shapes})
    ratios = []
    for b in blocks:
        keys = [k for k in shapes if k.startswith(f"{b}.")]
        got = sum(layer_bits(*shapes[k], ranks[k], num_scales) for k in keys)
        ref = sum(layer_bits(*shapes[k], legacy[k], num_scales) for k in keys)
        ratios.append(got / ref if ref else float("nan"))
    lines.append("    bits per block relative to uniform: " + " ".join(f"{r:.2f}" for r in ratios))
    return "\n".join(lines)


def calculate_ranks(model, layers_to_analyze, quant_config, sensitivity: dict | None = None):
    """Per-layer factorisation ranks for the bit target ``quant_config["bits"]``.

    ``rank_budget="uniform"`` (default) is the paper's rule: each layer's rank is the bit-exact continuous rank
    floored to a multiple of 32 (at least 32). ``"parity"`` / ``"full"`` allocate the same total bits (or exactly
    ``bits`` per weight) by marginal predicted loss per bit from the sensitivity curves measured at calibration
    time (:func:`nanoquant.core.rank_probe.measure_sensitivity`, :func:`allocate_ranks_measured`), with
    ``rank_depth_ramp`` as an optional depth prior. Until those curves exist (``sensitivity`` is ``None``, e.g. the
    accounting printed before calibration) the uniform rule is returned.

    Parameters
    ----------
    model : nn.Module
        Model whose decoder blocks provide the layer shapes.
    layers_to_analyze : list of str
        Sub-layer names within each block.
    quant_config : dict
        Quantisation configuration.
    sensitivity : dict, optional
        Probe artifact ``{"probes": ..., "curves": {key: (a, beta)}, "meta": ...}``.

    Returns
    -------
    dict
        ``"<block index>.<sub-layer name>" -> rank``.
    """
    num_scales = 3 if has_mid_scale(quant_config) else 2
    bits = quant_config['bits']
    budget = quant_config.get('rank_budget', 'uniform') or 'uniform'
    if budget not in RANK_BUDGETS:
        raise ValueError(f"Unknown rank_budget: {budget}")
    measured = quant_config.get('rank_sensitivity', 'none') or 'none'
    if measured not in RANK_SENSITIVITIES:
        raise ValueError(f"Unknown rank_sensitivity: {measured}")
    ramp = float(quant_config.get('rank_depth_ramp', 0.0) or 0.0)
    if budget == 'uniform' and (ramp or measured != 'none'):
        raise ValueError("rank_depth_ramp / rank_sensitivity require rank_budget='parity' or 'full'")
    if budget != 'uniform' and measured == 'none':
        raise ValueError("rank_budget='parity' / 'full' requires a measured rank_sensitivity ('admm' or 'svd')")

    print(f"Rank calculation: Bits = ({bits:.2f}), Scales: {num_scales}, budget: {budget}"
          + (f", measured sensitivity ({measured})" if measured != 'none' else "")
          + (f", depth ramp {ramp:g}" if ramp else ""))
    blocks = get_decoder_layers(model)
    shapes: dict[str, tuple[int, int]] = {}
    for i, layer in enumerate(blocks):
        subset = find_layers(layer)
        for name in layers_to_analyze:
            if name in subset:
                shapes[f"{i}.{name}"] = (subset[name].in_features, subset[name].out_features)
    legacy = {k: uniform_rank(a, b, bits, num_scales) for k, (a, b) in shapes.items()}
    if budget == 'uniform':
        return legacy
    if sensitivity is None:
        print("Rank allocation: measured sensitivity pending (probe runs after calibration); "
              "showing the uniform rule")
        return legacy
    curves = sensitivity["curves"]
    mult = depth_multipliers(shapes, len(blocks), ramp) if ramp else None
    ranks = allocate_ranks_measured(shapes, curves, bits, num_scales, budget, legacy, mult=mult)
    print(_format_measured_summary(shapes, ranks, legacy, curves, num_scales))
    changed = sum(ranks[k] != legacy[k] for k in ranks)
    print(f"Rank allocation: {changed}/{len(ranks)} layers differ from the uniform rule; "
          f"ranks {min(ranks.values())}..{max(ranks.values())}")
    return ranks
