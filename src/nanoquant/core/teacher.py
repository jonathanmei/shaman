# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Teacher logits for model-level knowledge distillation: RAM cache, disk memmap, or online."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from ..utils.cache import ArtifactCache

TEACHER_MODES = ("ram", "disk", "online")


def _logits_of(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    out = model(batch)
    return out.logits if hasattr(out, "logits") else out


class TeacherLogits:
    """Provide the FP teacher's logits for calibration sample ``idx`` in one of three modes.

    Parameters
    ----------
    mode : {"ram", "disk", "online"}
        ``"ram"`` precomputes every sample's logits and keeps them on the host (legacy; ~0.6 GB per
        2048-token sample for a 150k vocabulary). ``"online"`` keeps the teacher on ``dev`` and
        recomputes logits at every step (no extra memory, one extra forward per step). ``"disk"``
        precomputes into an ``int16``-backed bf16 ``numpy.memmap`` in the artifact cache, keyed by
        ``key`` (model + data only, so it is shared by every run on that model/data).
    fp_model : nn.Module
        Full-precision teacher (moved to ``dev`` here; never moved back, as in the legacy code).
    samples : Sequence[torch.Tensor]
        Calibration batches, each ``(1, seqlen)`` on ``dev``.
    dev : str
        Device of the student.
    cache : ArtifactCache, optional
        Required for ``"disk"``.
    key : str, optional
        Teacher key (see ``utils.cache.teacher_key``); required for ``"disk"``.
    """
    def __init__(self, mode: str, fp_model: nn.Module, samples: Sequence[torch.Tensor], dev: str,
                 cache: ArtifactCache | None = None, key: str | None = None):
        if mode not in TEACHER_MODES:
            raise ValueError(f"Unknown teacher mode '{mode}'. Choose from {TEACHER_MODES}.")
        self.mode = mode
        self.dev = dev
        self.samples = samples
        self.model = fp_model.eval().to(dev)
        self._ram: dict[int, torch.Tensor] = {}
        self._mm = None
        if mode == "ram":
            self._precompute_ram()
        elif mode == "disk":
            if cache is None or not cache.enabled or key is None:
                raise ValueError("teacher mode 'disk' requires an enabled ArtifactCache and a key")
            self._open_or_fill_memmap(cache.path("teacher", key, ".bin"))

    @torch.no_grad()
    def _precompute_ram(self) -> None:
        for idx, batch in enumerate(tqdm(self.samples, desc="Precomputing teacher logits (FP model)")):
            self._ram[idx] = _logits_of(self.model, batch).detach().cpu()

    @torch.no_grad()
    def _open_or_fill_memmap(self, path: Path) -> None:
        meta_path = path.with_suffix(".json")
        n = len(self.samples)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._mm = np.memmap(path, dtype=np.int16, mode="r", shape=tuple(meta["shape"]))
            if self._mm.shape[0] != n:
                raise ValueError(f"teacher memmap holds {self._mm.shape[0]} samples, expected {n}")
            print(f"[cache] hit  teacher {path.name}")
            return
        print(f"[cache] miss teacher {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        first = _logits_of(self.model, self.samples[0]).detach()
        shape = (n, *first.shape[1:])
        mm = np.memmap(path, dtype=np.int16, mode="w+", shape=shape)
        mm[0] = first.to(torch.bfloat16).cpu().view(torch.int16).numpy()[0]
        for idx in tqdm(range(1, n), desc="Precomputing teacher logits (FP model) to disk"):
            logits = _logits_of(self.model, self.samples[idx]).detach().to(torch.bfloat16).cpu()
            mm[idx] = logits.view(torch.int16).numpy()[0]
        mm.flush()
        del mm
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"shape": list(shape), "dtype": "bfloat16"}))
        tmp.replace(meta_path)
        self._mm = np.memmap(path, dtype=np.int16, mode="r", shape=shape)

    @torch.no_grad()
    def get(self, idx: int, batch: torch.Tensor) -> torch.Tensor:
        """Teacher logits for sample ``idx`` (``batch`` is the same ``(1, seqlen)`` tensor), on ``dev``."""
        if self.mode == "ram":
            return self._ram[idx].to(self.dev, non_blocking=True)
        if self.mode == "disk":
            arr = np.array(self._mm[idx])  # one contiguous slice
            return torch.from_numpy(arr).view(torch.bfloat16).unsqueeze(0).to(self.dev)
        return _logits_of(self.model, batch).detach()
