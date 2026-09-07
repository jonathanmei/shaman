# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostics and utilities for the continuous latent factors of :class:`NanoQuantLinear` layers.

A factorised layer deploys ``U = sign(U_latent)`` and ``V = sign(V_latent)``. During latent-aware tuning
(block-level ``tune_fact`` and model-level KD with ``model_kd_mode="scales_latent"``) the latents move and
some signs flip; these helpers count the flips against a reference sign pattern, summarise the sign
margins ``|latent|`` (how close entries are to flipping), rescale latents without changing the forward pass
and drop them once they are no longer needed.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
from torch import nn

from ..modules.linear import NanoQuantLinear

LATENT_NAMES: tuple[str, ...] = ("U_latent", "V_latent")


def hard_sign(x: torch.Tensor) -> torch.Tensor:
    """Deployed sign convention of the binary factors: ``+1`` for ``x >= 0`` and ``-1`` otherwise."""
    return torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))


def layer_type(name: str) -> str:
    """Last component of a module path, e.g. ``"model.layers.3.mlp.down_proj" -> "down_proj"``."""
    return name.rsplit(".", 1)[-1] if name else name


def iter_latent_layers(model: nn.Module) -> Iterator[tuple[str, NanoQuantLinear]]:
    """Yield ``(name, module)`` for every :class:`NanoQuantLinear` that still holds latent factors."""
    for name, module in model.named_modules():
        if isinstance(module, NanoQuantLinear) and module.has_latent:
            yield name, module


def sign_flips(latent: torch.Tensor, reference_sign: torch.Tensor) -> int:
    """Number of entries whose deployed sign differs from ``reference_sign`` (a ±1 tensor of the same shape)."""
    return int((hard_sign(latent.detach()) != reference_sign).sum().item())


@torch.no_grad()
def latent_flip_stats(model: nn.Module) -> dict:
    """Sign flips of every layer's latents relative to its stored hardened ``U``/``V``.

    The hardened factors are exactly the signs the latents had when the layer was last finalised, so
    this measures how many bits a latent-aware stage has flipped since then.

    Parameters
    ----------
    model : nn.Module
        Model (or block) containing :class:`NanoQuantLinear` layers with retained latents and hardened
        ``U``/``V``.

    Returns
    -------
    dict
        ``{"flipped", "total", "fraction", "layers", "zero_flip_layers", "by_type": {type: {...}}}``.
    """
    by_type: dict[str, dict[str, int]] = {}
    layers = zero = 0
    for name, module in iter_latent_layers(model):
        flips = sign_flips(module.U_latent, module.U) + sign_flips(module.V_latent, module.V)
        total = module.U_latent.numel() + module.V_latent.numel()
        entry = by_type.setdefault(layer_type(name), {"flipped": 0, "total": 0})
        entry["flipped"] += flips
        entry["total"] += total
        layers += 1
        zero += int(flips == 0)
    flipped = sum(e["flipped"] for e in by_type.values())
    total = sum(e["total"] for e in by_type.values())
    for e in by_type.values():
        e["fraction"] = e["flipped"] / e["total"] if e["total"] else 0.0
    return {"flipped": flipped, "total": total, "fraction": flipped / total if total else 0.0, "layers": layers,
            "zero_flip_layers": zero, "by_type": by_type}


@torch.no_grad()
def latent_margin_stats(model: nn.Module, thresholds: Iterable[float] = (1e-3, 1e-2)) -> dict:
    """Sign margins ``|latent|`` per layer type: median and the fraction below each threshold.

    Parameters
    ----------
    model : nn.Module
        Model containing layers with retained latents.
    thresholds : iterable of float
        Margin thresholds; an entry below a threshold flips once the optimiser has moved it by that much.

    Returns
    -------
    dict
        ``{"by_type": {type: {"median": float, "below": {thr: fraction}}}, "median": float, "below": {thr: fraction}}``
        where the type-level median is the mean of the per-layer medians.
    """
    thresholds = tuple(float(t) for t in thresholds)
    acc: dict[str, dict] = {}
    all_medians: list[float] = []
    all_below = {t: 0 for t in thresholds}
    all_total = 0
    for name, module in iter_latent_layers(model):
        entry = acc.setdefault(layer_type(name), {"medians": [], "below": {t: 0 for t in thresholds}, "total": 0})
        for attr in LATENT_NAMES:
            mag = getattr(module, attr).detach().abs().float()
            med = mag.median().item()
            entry["medians"].append(med)
            all_medians.append(med)
            entry["total"] += mag.numel()
            all_total += mag.numel()
            for t in thresholds:
                n = int((mag < t).sum().item())
                entry["below"][t] += n
                all_below[t] += n
    by_type = {
        k: {"median": sum(v["medians"]) / len(v["medians"]),
            "below": {t: v["below"][t] / v["total"] for t in thresholds}}
        for k, v in acc.items()
    }
    return {"by_type": by_type, "median": sum(all_medians) / len(all_medians) if all_medians else float("nan"),
            "below": {t: all_below[t] / all_total if all_total else float("nan") for t in thresholds}}


@torch.no_grad()
def normalize_latents(model: nn.Module, eps: float = 1e-12) -> None:
    """Rescale every latent row to unit mean magnitude (in place).

    Only the signs of the latents enter the forward pass, so a positive per-row scaling leaves the
    deployed model unchanged while making an optimiser step of size ``lr`` mean the same flip budget in
    every layer and row.
    """
    for _, module in iter_latent_layers(model):
        for attr in LATENT_NAMES:
            p = getattr(module, attr)
            scale = p.detach().abs().float().mean(dim=1, keepdim=True).clamp_min(eps)
            p.data.copy_((p.data.float() / scale).to(p.dtype))


def drop_latents(model: nn.Module) -> None:
    """Delete the latent factors of every :class:`NanoQuantLinear` in ``model``."""
    for module in model.modules():
        if isinstance(module, NanoQuantLinear):
            module.drop_latent()


def format_flip_stats(stats: dict, title: str = "latent flips") -> str:
    """One-line summary of :func:`latent_flip_stats`."""
    parts = [f"{k} {v['fraction']:.3e}" for k, v in sorted(stats["by_type"].items())]
    return (f"[{title}] {stats['flipped']}/{stats['total']} ({stats['fraction']:.3e}); "
            f"{stats['zero_flip_layers']}/{stats['layers']} layers without flips; by type: " + ", ".join(parts))


def format_margin_stats(stats: dict, title: str = "latent margins") -> str:
    """One-line summary of :func:`latent_margin_stats`."""
    below = ", ".join(f"<{t:g}: {f:.3e}" for t, f in stats["below"].items())
    parts = [f"{k} median {v['median']:.3e}" for k, v in sorted(stats["by_type"].items())]
    return f"[{title}] median |latent| {stats['median']:.3e}; fraction {below}; by type: " + ", ".join(parts)
