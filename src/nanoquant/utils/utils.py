# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

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


def _floor_rank(rank, in_features: int, out_features: int, min_rank: int = RANK_STEP) -> int:
    """Legacy rounding: floor to a multiple of 32, at least ``min_rank``, at most ``min(in, out)``."""
    curr_rank = int(rank) if rank is not None else 0
    curr_rank = (curr_rank // RANK_STEP) * RANK_STEP
    curr_rank = max(curr_rank, min_rank)
    return min(curr_rank, min(in_features, out_features))


def parse_type_weights(spec: str) -> dict[str, float]:
    """Parse ``"v_proj:1.2,down_proj:1.15"`` into ``{"v_proj": 1.2, "down_proj": 1.15}``.

    Keys are the last component of a layer name (``q_proj``, ``down_proj``, ...) or the full sub-layer name
    (``self_attn.q_proj``); values are positive bit-budget multipliers.

    Raises
    ------
    ValueError
        On a malformed entry or a non-positive weight.
    """
    weights: dict[str, float] = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"rank_type_weights entry '{item}' must be 'name:weight'")
        name, value = item.rsplit(":", 1)
        try:
            w = float(value)
        except ValueError as e:
            raise ValueError(f"rank_type_weights entry '{item}': weight is not a number") from e
        if w <= 0:
            raise ValueError(f"rank_type_weights entry '{item}': weight must be > 0")
        weights[name.strip()] = w
    return weights


def _type_weight(name: str, weights: dict[str, float]) -> float:
    if name in weights:
        return weights[name]
    return weights.get(name.rsplit(".", 1)[-1], 1.0)


def allocate_ranks(shapes: dict[str, tuple[int, int]], n_blocks: int, bits: float, num_scales: int,
                   depth_ramp: float, type_weights: dict[str, float], budget: str,
                   legacy: dict[str, int]) -> dict[str, int]:
    """Budget-matched non-uniform rank allocation.

    Every layer ``l`` gets a bit-budget multiplier ``m_l = exp(ramp (b_l/(B-1) - 1/2)) * w_type(l)`` (block index
    ``b_l`` of ``B`` blocks), normalised so that the continuous budgets sum to ``bits`` times the number of
    factorised weights. Ranks start at the floored continuous targets and are then moved in steps of 32 toward
    the bit target: ``budget="parity"`` targets the bits the legacy uniform rule actually spends (so arms are
    comparable at identical actual bpw), ``budget="full"`` targets ``bits`` per weight exactly (spending the
    remainder that the 32-multiple rounding leaves idle). Steps are taken on the layer whose rank is furthest
    from its continuous target in relative terms, never below 32 nor above ``min(in, out)``.

    Parameters
    ----------
    shapes : dict
        ``"<block>.<name>" -> (in_features, out_features)`` in schedule order.
    n_blocks : int
        Number of decoder blocks.
    bits : float
        Target bits per weight.
    num_scales : int
        2 or 3.
    depth_ramp : float
        Log-ratio of the last block's multiplier to the first block's (0 = flat).
    type_weights : dict
        Per-layer-type multipliers (see :func:`parse_type_weights`).
    budget : str
        ``"parity"`` or ``"full"``.
    legacy : dict
        Ranks of the uniform rule (defines the parity target).

    Returns
    -------
    dict
        ``"<block>.<name>" -> rank``.
    """
    if budget not in ("parity", "full"):
        raise ValueError(f"Unknown rank_budget for allocate_ranks: {budget}")
    known = {k.split(".", 1)[1] for k in shapes} | {k.split(".", 1)[1].rsplit(".", 1)[-1] for k in shapes}
    unknown = [t for t in type_weights if t not in known]
    if unknown:
        raise ValueError(f"rank_type_weights refer to layer types not in the model: {unknown}")
    # bit-budget multipliers, normalised to leave the total budget unchanged
    mult: dict[str, float] = {}
    for key in shapes:
        blk, name = key.split(".", 1)
        depth = np.exp(depth_ramp * (int(blk) / (n_blocks - 1) - 0.5)) if n_blocks > 1 else 1.0
        mult[key] = float(depth) * _type_weight(name, type_weights)
    weights_total = sum(a * b for a, b in shapes.values())
    norm = weights_total / sum(mult[k] * a * b for k, (a, b) in shapes.items())
    target = {k: max(_continuous_rank(a, b, bits * mult[k] * norm, num_scales), 1.0) for k, (a, b) in shapes.items()}
    ranks = {k: _floor_rank(target[k], a, b) for k, (a, b) in shapes.items()}
    lo, hi = RANK_STEP, {k: min(a, b) for k, (a, b) in shapes.items()}
    step_bits = {k: RANK_STEP * (a + b) + (SCALE_BITS * RANK_STEP if num_scales == 3 else 0)
                 for k, (a, b) in shapes.items()}
    if budget == "parity":
        total_target = sum(layer_bits(a, b, legacy[k], num_scales) for k, (a, b) in shapes.items())
    else:
        total_target = int(bits * weights_total)
    total = sum(layer_bits(a, b, ranks[k], num_scales) for k, (a, b) in shapes.items())

    def rel_excess(k):
        return (ranks[k] - target[k]) / target[k]

    while total > total_target:
        cands = [k for k in shapes if ranks[k] - RANK_STEP >= lo]
        if not cands:
            break
        k = max(cands, key=rel_excess)
        ranks[k] -= RANK_STEP
        total -= step_bits[k]
    while True:
        cands = [k for k in shapes if ranks[k] + RANK_STEP <= hi[k] and total + step_bits[k] <= total_target]
        if not cands:
            break
        k = min(cands, key=rel_excess)
        ranks[k] += RANK_STEP
        total += step_bits[k]
    return ranks


def calculate_ranks(model, layers_to_analyze, quant_config):
    """Per-layer factorisation ranks for the bit target ``quant_config["bits"]``.

    With the defaults (``rank_budget="uniform"``, no depth ramp, no type weights) this is the legacy rule: each
    layer's rank is the bit-exact continuous rank floored to a multiple of 32 (at least 32). ``rank_budget`` of
    ``"parity"`` or ``"full"`` enables the non-uniform allocation of :func:`allocate_ranks` driven by
    ``rank_depth_ramp`` and ``rank_type_weights``.

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
    ramp = float(quant_config.get('rank_depth_ramp', 0.0) or 0.0)
    type_weights = parse_type_weights(quant_config.get('rank_type_weights', '') or '')

    print(f"Rank calculation: Bits = ({bits:.2f}), Scales: {num_scales}, budget: {budget}"
          + (f", depth ramp {ramp:g}" if ramp else "") + (f", type weights {type_weights}" if type_weights else ""))
    blocks = get_decoder_layers(model)
    shapes: dict[str, tuple[int, int]] = {}
    for i, layer in enumerate(blocks):
        subset = find_layers(layer)
        for name in layers_to_analyze:
            if name in subset:
                shapes[f"{i}.{name}"] = (subset[name].in_features, subset[name].out_features)
    legacy = {k: _floor_rank(_continuous_rank(a, b, bits, num_scales), a, b) for k, (a, b) in shapes.items()}
    if budget == 'uniform':
        if ramp or type_weights:
            raise ValueError("rank_depth_ramp / rank_type_weights require rank_budget='parity' or 'full'")
        return legacy
    ranks = allocate_ranks(shapes, len(blocks), bits, num_scales, ramp, type_weights, budget, legacy)
    changed = sum(ranks[k] != legacy[k] for k in ranks)
    print(f"Rank allocation: {changed}/{len(ranks)} layers differ from the uniform rule; "
          f"ranks {min(ranks.values())}..{max(ranks.values())}")
    return ranks
