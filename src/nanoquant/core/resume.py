# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Block-level checkpoint and resume for the sequential block-reconstruction loop.

After block ``i`` is reconstructed its compressed state (packed ±1 factors, scales, tuned
full-precision weights) is stored under the block's chain key, together with a rolling *progress*
record holding the activations that block ``i+1`` needs (``compressed_inputs`` from the quantised
prefix and ``original_inputs`` from the FP prefix). A later run with the same chain restores the
longest completed prefix and continues.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from ..utils.cache import ArtifactCache

PROGRESS_KIND = "progress"
BLOCK_KIND = "block"


def compressed_state_dict(model: nn.Module) -> OrderedDict:
    """State dict of ``model`` in which ``NanoQuantLinear`` modules use their packed representation.

    Parameters
    ----------
    model : nn.Module
        Any module (whole model or a single decoder block).

    Returns
    -------
    OrderedDict
        Parameters of plain modules plus the packed state of every ``NanoQuantLinear``.
    """
    from ..modules.linear import NanoQuantLinear

    state = OrderedDict()
    for name, param in model.named_parameters():
        module_path = name.rsplit('.', 1)[0] if '.' in name else ''
        try:
            module = model.get_submodule(module_path) if module_path else model
            if not isinstance(module, NanoQuantLinear):
                state[name] = param.data
        except AttributeError:
            state[name] = param.data
    for name, mod in model.named_modules():
        if isinstance(mod, NanoQuantLinear):
            state.update(mod.state_dict(prefix=name + '.' if name else ''))
    return state


def block_state(block: nn.Module) -> OrderedDict:
    """CPU copy of :func:`compressed_state_dict` for one block (safe to ``torch.save``)."""
    return OrderedDict((k, v.detach().to("cpu").clone()) for k, v in compressed_state_dict(block).items())


def restore_block(block: nn.Module, state: dict) -> None:
    """Load a checkpointed block state into a fresh full-precision ``block`` in place.

    Every ``nn.Linear`` whose packed factors are present in ``state`` is converted to
    ``NanoQuantLinear`` (same recipe as ``load_utils.load_compressed_model``), with the middle scale
    enabled iff ``<name>.scale_mid`` is stored. The packed ``V``/``U`` are unpacked by
    ``NanoQuantLinear._load_from_state_dict``.

    Raises
    ------
    RuntimeError
        If unexpected keys are found or non-factor keys are missing.
    """
    from ..modules.linear import NanoQuantLinear

    for name, module in list(block.named_modules()):
        if type(module) is nn.Linear:
            base = f"{name}." if name else ""
            if base + "V_packed" in state:
                module.__class__ = NanoQuantLinear
                rank = int(state[base + "V_shape"][0])
                module.init_for_inference(rank=rank, has_scale_mid=(base + "scale_mid") in state,
                                          has_latent=(base + "V_latent") in state)
    result = block.load_state_dict(state, strict=False)
    # the packed V/U are consumed by the custom loader and therefore reported as "missing" by torch
    missing = [k for k in result.missing_keys if not (k.endswith((".V", ".U")) or k in ("V", "U"))]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"block restore mismatch: missing={missing[:5]} unexpected={result.unexpected_keys[:5]}")


def save_block_checkpoint(cache: ArtifactCache, key: str, block: nn.Module) -> None:
    """Persist ``block`` under its chain ``key``."""
    cache.save(BLOCK_KIND, key, block_state(block))


def save_progress(cache: ArtifactCache, root_key: str, block_idx: int, compressed_inputs: torch.Tensor,
                  original_inputs: torch.Tensor) -> None:
    """Persist the activations entering block ``block_idx + 1`` (rolling, one record per chain)."""
    cache.save(PROGRESS_KIND, root_key, {
        "block_idx": int(block_idx),
        "compressed_inputs": compressed_inputs.detach().to("cpu"),
        "original_inputs": original_inputs.detach().to("cpu"),
    })


def load_progress(cache: ArtifactCache, root_key: str) -> dict | None:
    return cache.load(PROGRESS_KIND, root_key)


def completed_prefix(cache: ArtifactCache, keys: list[str]) -> int:
    """Number of leading blocks whose checkpoints exist (0 when none or the cache is disabled)."""
    if not cache.enabled:
        return 0
    k = 0
    for key in keys:
        if not cache.exists(BLOCK_KIND, key):
            break
        k += 1
    return k


def restore_prefix(cache: ArtifactCache, root_key: str, keys: list[str],
                   blocks) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    """Restore the longest resumable prefix of ``blocks`` in place.

    Resumption needs both the block checkpoints ``0..k-1`` and a progress record for block ``k-1``
    (the activations entering block ``k``); the prefix is truncated to what the progress record covers.

    Returns
    -------
    tuple
        ``(k, compressed_inputs, original_inputs)``; ``k == 0`` means start from scratch.
    """
    k = completed_prefix(cache, keys)
    if k == 0:
        return 0, None, None
    prog = load_progress(cache, root_key)
    if prog is None:
        return 0, None, None
    k = min(k, int(prog["block_idx"]) + 1)
    if k <= 0:
        return 0, None, None
    for i in range(k):
        restore_block(blocks[i], cache.load(BLOCK_KIND, keys[i]))
    return k, prog["compressed_inputs"], prog["original_inputs"]
