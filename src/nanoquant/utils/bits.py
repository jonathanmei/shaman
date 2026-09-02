# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Bits-per-weight (bpw) accounting.

Two views are provided:

* :func:`static_accounting` – what the rank-budget rule (``utils.calculate_ranks``) will produce for a
  model and config *before* quantising: per-layer ranks and bits, and the resulting bpw over the
  factorised weights. This mirrors the paper's effective-bit definition (Appendix F, Eq. 60:
  total bits of the decoder linear layers divided by their number of weights), i.e. embeddings,
  norms and the LM head are excluded.
* :func:`model_accounting` – what a quantised model actually holds: ±1 entries of ``U``/``V`` count
  1 bit, every scale entry 16 bits, and all remaining parameters their storage width. Reports the
  factorised-layer bpw (comparable to the paper) and a whole-model bpw / checkpoint size.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .utils import calculate_ranks, find_layers, get_decoder_layers, has_mid_scale

SCALE_BITS = 16


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


def static_accounting(model: nn.Module, layers_to_factorize: Iterable[str], quant_config: dict) -> dict:
    """Predict per-layer ranks/bits and the factorised-layer bpw from the rank-budget rule.

    Parameters
    ----------
    model : nn.Module
        Full-precision model (only layer shapes are read).
    layers_to_factorize : iterable of str
        Sub-layer names within each decoder block (see ``utils.get_layers_to_factorize``).
    quant_config : dict
        Quantisation config (``bits``, ``admm_type``, ``admm_mid_scale``).

    Returns
    -------
    dict
        ``{"target_bits", "num_scales", "layers": {name: {...}}, "factorized_weights", "factorized_bits",
        "factorized_bpw"}``.
    """
    layers_to_factorize = list(layers_to_factorize)
    ranks = calculate_ranks(model, layers_to_factorize, quant_config)
    num_scales = 3 if has_mid_scale(quant_config) else 2
    layers: dict[str, dict] = {}
    tot_bits = tot_w = 0
    for i, block in enumerate(get_decoder_layers(model)):
        subset = find_layers(block)
        for name in layers_to_factorize:
            if name not in subset:
                continue
            lx = subset[name]
            rank = ranks[f"{i}.{name}"]
            bits = layer_bits(lx.in_features, lx.out_features, rank, num_scales)
            weights = lx.in_features * lx.out_features
            layers[f"{i}.{name}"] = {"in": lx.in_features, "out": lx.out_features, "rank": rank, "bits": bits,
                                     "bpw": bits / weights}
            tot_bits += bits
            tot_w += weights
    return {
        "target_bits": quant_config["bits"],
        "num_scales": num_scales,
        "layers": layers,
        "factorized_weights": tot_w,
        "factorized_bits": tot_bits,
        "factorized_bpw": tot_bits / tot_w if tot_w else float("nan"),
    }


@torch.no_grad()
def model_accounting(model: nn.Module) -> dict:
    """Count the bits actually stored by a (partially) quantised model.

    Parameters
    ----------
    model : nn.Module
        Model containing ``NanoQuantLinear`` modules (others are counted at their parameter width).

    Returns
    -------
    dict
        ``factorized_*`` totals over the ``NanoQuantLinear`` layers (bpw comparable to the paper),
        ``other_bits``/``other_params`` for everything else, ``model_bpw`` and ``checkpoint_mib``.
    """
    from ..modules.linear import NanoQuantLinear

    layers: dict[str, dict] = {}
    fact_bits = fact_w = 0
    other_bits = other_params = 0
    quant_modules = set()
    for name, mod in model.named_modules():
        if not isinstance(mod, NanoQuantLinear):
            continue
        quant_modules.add(mod)
        binary = 0
        for attr in ("V", "U", "V_latent", "U_latent"):
            p = getattr(mod, attr, None)
            if p is not None:
                binary += p.numel()
        scales = 0
        for attr in ("scale_pre", "scale_mid", "scale_post"):
            p = getattr(mod, attr, None)
            if p is not None:
                scales += p.numel()
        bias_bits = 0
        if getattr(mod, "bias", None) is not None:
            bias_bits = mod.bias.numel() * mod.bias.element_size() * 8
        bits = binary + SCALE_BITS * scales + bias_bits
        weights = mod.in_features * mod.out_features
        rank = getattr(mod, "rank", None)
        layers[name] = {"in": mod.in_features, "out": mod.out_features, "rank": rank, "binary_bits": binary,
                        "scale_bits": SCALE_BITS * scales, "bits": bits, "bpw": bits / weights}
        fact_bits += bits
        fact_w += weights
    seen = set()
    for name, mod in model.named_modules():
        if mod in quant_modules:
            continue
        for pname, p in mod.named_parameters(recurse=False):
            if id(p) in seen:  # tied weights count once
                continue
            seen.add(id(p))
            other_params += p.numel()
            other_bits += p.numel() * p.element_size() * 8
    total_bits = fact_bits + other_bits
    total_params = fact_w + other_params
    return {
        "layers": layers,
        "factorized_weights": fact_w,
        "factorized_bits": fact_bits,
        "factorized_bpw": fact_bits / fact_w if fact_w else float("nan"),
        "other_params": other_params,
        "other_bits": other_bits,
        "model_bpw": total_bits / total_params if total_params else float("nan"),
        "checkpoint_mib": total_bits / 8 / 2**20,
    }


def format_accounting(acc: dict, title: str = "bpw accounting") -> str:
    """Human-readable multi-line summary of a :func:`static_accounting` / :func:`model_accounting` result."""
    lines = [f"[{title}] factorized layers: {acc['factorized_weights'] / 1e6:.1f}M weights -> "
             f"{acc['factorized_bits'] / 8 / 2**20:.1f} MiB, {acc['factorized_bpw']:.4f} bpw"]
    if "target_bits" in acc:
        lines[0] += f" (target {acc['target_bits']}, {acc['num_scales']} scales)"
    if "model_bpw" in acc:
        lines.append(f"[{title}] whole model: {acc['model_bpw']:.3f} bpw incl. {acc['other_params'] / 1e6:.1f}M "
                     f"non-factorized params; checkpoint ~ {acc['checkpoint_mib']:.0f} MiB")
    # one line per distinct (shape, rank) to keep the summary short
    seen: dict[tuple, int] = {}
    for info in acc["layers"].values():
        key = (info["in"], info["out"], info.get("rank"), round(info["bpw"], 4))
        seen[key] = seen.get(key, 0) + 1
    for (i, o, r, bpw), n in sorted(seen.items()):
        lines.append(f"[{title}]   {n:3d} x ({i} -> {o}) rank={r}: {bpw:.4f} bpw")
    return "\n".join(lines)
