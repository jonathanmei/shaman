# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed artifact cache for the NanoQuant pipeline.

The quantisation pipeline has a layered dependency structure, so one monolithic run key would be
far too strict. Each stage gets its own key built from the exact subset of configuration fields
(and, where the inputs are tensors, the raw bytes of those tensors) that determine its output:

* ``stats_key``   – raw calibration statistics (independent of shrinkage and of everything downstream);
* ``chain_keys``  – one key per decoder block, chained so that block ``i`` depends on blocks ``< i``;
* ``kd_key``      – model-level knowledge distillation on top of the last block;
* ``admm_key``    – a single layer's ADMM solution, keyed by the *bytes* of the weight matrix and
  curvature tensors it consumes, so reuse is automatically valid whenever those inputs are identical;
* ``teacher_key`` – the FP teacher's logits (model + data only).

Every key also folds in a fingerprint of the source files implementing that stage, so algorithmic
code changes invalidate stale artifacts automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch

PACKAGE_DIR = Path(__file__).resolve().parents[1]

# --- field subsets (single source of truth; tests assert sensitivity/insensitivity against these) ---
STATS_FIELDS: tuple[str, ...] = ("model_id", "seqlen", "calib_dataset", "num_calib_samples", "seed", "calib_strategy",
                                 "curvature", "kron_nkp_iters")
ADMM_FIELDS: tuple[str, ...] = ("admm_type", "admm_outer_iters", "admm_inner_iters", "admm_reg",
                                "admm_penalty_scheduler", "admm_mid_scale", "kron_eigh_dtype", "seed")
BLOCK_FIELDS: tuple[str, ...] = ("calib_shrinkage", "bits") + ADMM_FIELDS + (
    "tune_nonfact", "nonfact_lr", "nonfact_batch_size", "nonfact_epochs", "tune_fact", "fact_binary_lr",
    "fact_scale_lr", "fact_bias_lr", "fact_batch_size", "fact_epochs")
KD_FIELDS: tuple[str, ...] = ("model_kd_lr", "model_kd_batch_size", "model_kd_epochs")
TEACHER_FIELDS: tuple[str, ...] = ("model_id", "seqlen", "calib_dataset", "num_calib_samples", "seed")

# source files whose algorithm determines each stage's output (relative to the ``nanoquant`` package)
SOURCE_GROUPS: dict[str, tuple[str, ...]] = {
    "stats": ("core/importance.py",),
    "admm": ("core/admm_nq.py", "core/admm_dbf.py", "core/compress_block.py"),
    "blocks": ("core/admm_nq.py", "core/admm_dbf.py", "core/compress_block.py", "core/compress_model.py",
               "modules/linear.py"),
    "kd": ("core/compress_model.py", "core/teacher.py"),
}


def _subset(cfg: dict, fields: Iterable[str]) -> dict:
    return {f: cfg.get(f) for f in fields}


def hash_json(obj: Any) -> str:
    """SHA-256 hex digest of the canonical JSON serialisation of ``obj``."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def hash_tensors(*tensors: torch.Tensor | None) -> str:
    """SHA-256 hex digest of the shapes, dtypes and raw bytes of ``tensors`` (``None`` is allowed).

    Parameters
    ----------
    *tensors : torch.Tensor or None
        Tensors on any device; they are copied to CPU and made contiguous for hashing.

    Returns
    -------
    str
    """
    h = hashlib.sha256()
    for t in tensors:
        if t is None:
            h.update(b"None")
            continue
        t = t.detach().to("cpu").contiguous()
        h.update(f"{tuple(t.shape)}|{t.dtype}".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def fingerprint_sources(group: str) -> str:
    """BLAKE2b digest of the source files listed in ``SOURCE_GROUPS[group]``."""
    h = hashlib.blake2b(digest_size=16)
    for rel in SOURCE_GROUPS[group]:
        h.update(rel.encode())
        h.update((PACKAGE_DIR / rel).read_bytes())
    return h.hexdigest()


def stats_key(cfg: dict) -> str:
    """Key of the raw calibration statistics (shrinkage is applied at load time and is *not* part of it)."""
    return hash_json({"fields": _subset(cfg, STATS_FIELDS), "code": fingerprint_sources("stats")})


def chain_root(cfg: dict) -> str:
    """Key of the block-reconstruction chain before any block (calibration key + all block-level settings)."""
    return hash_json({
        "stats": stats_key(cfg),
        "fields": _subset(cfg, BLOCK_FIELDS),
        "code": fingerprint_sources("blocks"),
    })


def chain_keys(cfg: dict, n_blocks: int) -> list[str]:
    """Per-block keys; block ``i`` depends on the chain root and on all blocks ``< i``."""
    keys: list[str] = []
    prev = chain_root(cfg)
    for i in range(n_blocks):
        prev = hash_json({"prev": prev, "block": i})
        keys.append(prev)
    return keys


def kd_key(cfg: dict, n_blocks: int) -> str:
    """Key of the model-level KD stage on top of the fully reconstructed block chain."""
    return hash_json({
        "last_block": chain_keys(cfg, n_blocks)[-1],
        "fields": _subset(cfg, KD_FIELDS),
        "code": fingerprint_sources("kd"),
    })


def teacher_key(cfg: dict) -> str:
    """Key of the FP teacher logits: depends on the model and the calibration data only."""
    return hash_json({"fields": _subset(cfg, TEACHER_FIELDS)})


def admm_key(W: torch.Tensor, i_norm: torch.Tensor, o_norm: torch.Tensor, i_cov: torch.Tensor | None,
             o_cov: torch.Tensor | None, rank: int, cfg: dict) -> str:
    """Content-addressed key of one layer's ADMM solution.

    Parameters
    ----------
    W : torch.Tensor
        Weight matrix exactly as fed to the factoriser (already reflects earlier stages).
    i_norm, o_norm : torch.Tensor
        Shrunk diagonal curvature.
    i_cov, o_cov : torch.Tensor or None
        Shrunk dense Kronecker factors (``None`` on the diagonal path).
    rank : int
        Factorisation rank.
    cfg : dict
        Quantisation config; only ``ADMM_FIELDS`` are used.
    """
    return hash_json({
        "tensors": hash_tensors(W, i_norm, o_norm, i_cov, o_cov),
        "rank": int(rank),
        "fields": _subset(cfg, ADMM_FIELDS),
        "code": fingerprint_sources("admm"),
    })


def atomic_save(obj: Any, path: Path) -> None:
    """``torch.save`` to ``path`` via a temporary file and an atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class ArtifactCache:
    """Directory-backed store of pipeline artifacts, addressed by ``(kind, key)``.

    Parameters
    ----------
    root : str or None
        Cache directory. ``None`` or ``""`` disables the cache (every method becomes a no-op /
        miss), which reproduces the legacy behaviour exactly.
    """
    def __init__(self, root: str | os.PathLike | None):
        self.root = Path(root).expanduser() if root else None

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def path(self, kind: str, key: str, suffix: str = ".pt") -> Path:
        if not self.enabled:
            raise RuntimeError("cache is disabled")
        return self.root / kind / f"{key}{suffix}"

    def exists(self, kind: str, key: str, suffix: str = ".pt") -> bool:
        return self.enabled and self.path(kind, key, suffix).exists()

    def save(self, kind: str, key: str, obj: Any) -> Path | None:
        """Store ``obj`` (any ``torch.save``-able, ``weights_only``-loadable object). Returns the path."""
        if not self.enabled:
            return None
        p = self.path(kind, key)
        atomic_save({"kind": kind, "key": key, "obj": obj}, p)
        return p

    def load(self, kind: str, key: str, map_location: str = "cpu") -> Any | None:
        """Return the stored object or ``None`` on a miss. A stored key mismatch raises ``ValueError``."""
        if not self.exists(kind, key):
            return None
        payload = torch.load(self.path(kind, key), map_location=map_location, weights_only=True)
        if payload.get("kind") != kind or payload.get("key") != key:
            raise ValueError(f"cache file {self.path(kind, key)} belongs to ({payload.get('kind')}, "
                             f"{str(payload.get('key'))[:12]}...), expected ({kind}, {key[:12]}...)")
        return payload["obj"]

    def load_or_compute(self, kind: str, key: str, fn: Callable[[], Any]) -> Any:
        """Return the cached object for ``(kind, key)`` or compute, store and return it."""
        obj = self.load(kind, key)
        if obj is not None:
            print(f"[cache] hit  {kind} {key[:12]}")
            return obj
        if self.enabled:
            print(f"[cache] miss {kind} {key[:12]}")
        obj = fn()
        self.save(kind, key, obj)
        return obj


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(PACKAGE_DIR),
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def append_ledger(root: str | os.PathLike, record: dict) -> Path:
    """Append one JSON line to ``<root>/results.jsonl`` with timestamp, git commit and Slurm job id added."""
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "results.jsonl"
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        **record,
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return path


def read_ledger(root: str | os.PathLike) -> list[dict]:
    """Parse ``<root>/results.jsonl`` (empty list if absent)."""
    path = Path(root).expanduser() / "results.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
