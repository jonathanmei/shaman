"""Small dense Ising instances: energies, brute force, and the RQAOA variable eliminations.

Conventions
-----------
Spins ``s_i`` are ``+1`` / ``-1``; the energy is ``E(s) = const + sum_i h_i s_i + sum_{i<j} J_ij s_i s_j`` with a
symmetric, zero-diagonal ``J``. Qubit ``q`` holds spin ``q``; measurement bit ``b_q = (1 - s_q) / 2``, so bit ``1``
is spin ``-1``. Basis-state indices put qubit 0 in the most significant position (``index = sum_q b_q 2^(n-1-q)``)
and bitstrings list qubit 0 first. Everything here is numpy-only and sized for ``n <= ~20``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator


def spin_table(n: int) -> np.ndarray:
    """All ``2^n`` spin configurations as a ``(2^n, n)`` array of ``+-1`` in basis-index order.

    Parameters
    ----------
    n : int
        Number of spins.

    Returns
    -------
    numpy.ndarray
        Row ``idx`` is :func:`index_to_spins` of ``idx``.
    """
    idx = np.arange(2 ** n, dtype=np.int64)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    return (1 - 2 * bits).astype(np.int8)


def index_to_spins(idx: int, n: int) -> list[int]:
    """Spins of basis state ``idx`` (qubit 0 most significant)."""
    return [1 - 2 * ((idx >> (n - 1 - q)) & 1) for q in range(n)]


def spins_to_index(s: Sequence[int]) -> int:
    """Basis-state index of a spin configuration."""
    idx = 0
    for v in s:
        idx = (idx << 1) | (1 if v < 0 else 0)
    return idx


def spins_to_bitstring(s: Sequence[int]) -> str:
    """Bitstring (qubit 0 first, bit 1 = spin -1) of a spin configuration."""
    return "".join("1" if v < 0 else "0" for v in s)


def bitstring_to_spins(bits: str) -> list[int]:
    """Inverse of :func:`spins_to_bitstring`."""
    return [-1 if b == "1" else 1 for b in bits]


class IsingInstance(BaseModel):
    """A dense Ising model ``E(s) = const + h . s + sum_{i<j} J_ij s_i s_j``.

    Parameters
    ----------
    h : list of float
        Linear fields.
    J : list of list of float
        Symmetric coupling matrix with zero diagonal.
    const : float
        Energy offset.
    labels : list of int, optional
        Original variable ids (RQAOA reductions renumber positions but keep labels). Defaults to ``range(n)``.
    meta : dict
        Free-form provenance (layer, tile entries, ADMM bits, ...), carried through reductions and JSON.
    """

    h: list[float]
    J: list[list[float]]
    const: float = 0.0
    labels: list[int] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_shapes(self) -> IsingInstance:
        n = len(self.h)
        J = np.asarray(self.J, dtype=np.float64)
        if J.shape != (n, n):
            raise ValueError(f"J must be ({n}, {n}), got {J.shape}")
        if n and not np.allclose(J, J.T, rtol=0.0, atol=1e-12 * max(1.0, float(np.abs(J).max()))):
            raise ValueError("J must be symmetric")
        if n and np.any(np.diagonal(J) != 0.0):
            raise ValueError("J must have a zero diagonal")
        if self.labels is None:
            self.labels = list(range(n))
        elif len(self.labels) != n:
            raise ValueError(f"labels must have length {n}, got {len(self.labels)}")
        return self

    @property
    def n(self) -> int:
        """Number of spins."""
        return len(self.h)

    def h_array(self) -> np.ndarray:
        """Fields as a float64 array."""
        return np.asarray(self.h, dtype=np.float64)

    def J_array(self) -> np.ndarray:
        """Couplings as a float64 array."""
        return np.asarray(self.J, dtype=np.float64)

    def energy(self, s: Sequence[int]) -> float:
        """Energy of one spin configuration."""
        v = np.asarray(s, dtype=np.float64)
        return float(self.const + self.h_array() @ v + 0.5 * v @ self.J_array() @ v)

    def all_energies(self) -> np.ndarray:
        """Energies of all ``2^n`` basis states (the QAOA cost diagonal), in basis-index order."""
        S = spin_table(self.n).astype(np.float64)
        return self.const + S @ self.h_array() + 0.5 * np.einsum("bi,bi->b", S @ self.J_array(), S)

    def brute_force(self) -> tuple[list[int], float]:
        """Exact minimiser ``(spins, energy)``; ties resolve to the lowest basis index."""
        e = self.all_energies()
        idx = int(np.argmin(e))
        return index_to_spins(idx, self.n), float(e[idx])

    def scaled(self, factor: float) -> IsingInstance:
        """The same model with ``h``, ``J`` and ``const`` multiplied by ``factor``."""
        return IsingInstance(h=(self.h_array() * factor).tolist(), J=(self.J_array() * factor).tolist(),
                             const=self.const * factor, labels=list(self.labels or []), meta=dict(self.meta))

    def to_json(self, path: str | Path) -> None:
        """Write the instance as JSON."""
        Path(path).write_text(json.dumps(self.model_dump(), indent=1))

    @classmethod
    def from_json(cls, path: str | Path) -> IsingInstance:
        """Read an instance written by :meth:`to_json`."""
        return cls.model_validate(json.loads(Path(path).read_text()))


def cost_scale(inst: IsingInstance) -> float:
    """Largest magnitude among fields and couplings (``1.0`` for an empty model); used to normalise QAOA angles."""
    vals = [float(np.abs(inst.h_array()).max()) if inst.n else 0.0,
            float(np.abs(inst.J_array()).max()) if inst.n else 0.0]
    m = max(vals)
    return m if m > 0 else 1.0


class Elimination(BaseModel):
    """One RQAOA reduction step, recorded by variable *label*.

    ``kind="pair"`` fixes ``s_i = sign * s_j``; ``kind="single"`` fixes ``s_i = sign``.
    """

    kind: Literal["pair", "single"]
    i: int
    j: int | None = None
    sign: int

    @model_validator(mode="after")
    def _check(self) -> Elimination:
        if self.sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1")
        if (self.kind == "pair") != (self.j is not None):
            raise ValueError("pair eliminations need j; single eliminations must not have one")
        return self


def _drop(inst: IsingInstance, h: np.ndarray, J: np.ndarray, const: float, pos: int) -> IsingInstance:
    keep = [k for k in range(inst.n) if k != pos]
    labels = [inst.labels[k] for k in keep] if inst.labels is not None else None
    Jr = J[np.ix_(keep, keep)]
    np.fill_diagonal(Jr, 0.0)
    return IsingInstance(h=h[keep].tolist(), J=Jr.tolist(), const=float(const), labels=labels, meta=dict(inst.meta))


def eliminate_pair(inst: IsingInstance, i_pos: int, j_pos: int, sign: int) -> tuple[IsingInstance, Elimination]:
    """Impose ``s_i = sign * s_j`` and remove variable ``i`` (positions in the current instance).

    Returns
    -------
    tuple
        The reduced instance (``n - 1`` spins, labels preserved) and the :class:`Elimination` record.
    """
    if i_pos == j_pos:
        raise ValueError("pair elimination needs two distinct variables")
    h, J = inst.h_array(), inst.J_array()
    const = inst.const + sign * J[i_pos, j_pos]
    h[j_pos] += sign * h[i_pos]
    for k in range(inst.n):
        if k not in (i_pos, j_pos):
            J[j_pos, k] += sign * J[i_pos, k]
            J[k, j_pos] = J[j_pos, k]
    rec = Elimination(kind="pair", i=inst.labels[i_pos], j=inst.labels[j_pos], sign=int(sign))
    return _drop(inst, h, J, const, i_pos), rec


def eliminate_single(inst: IsingInstance, i_pos: int, sign: int) -> tuple[IsingInstance, Elimination]:
    """Fix ``s_i = sign`` and remove variable ``i`` (position in the current instance)."""
    h, J = inst.h_array(), inst.J_array()
    const = inst.const + sign * h[i_pos]
    h = h + sign * J[i_pos]
    rec = Elimination(kind="single", i=inst.labels[i_pos], sign=int(sign))
    return _drop(inst, h, J, const, i_pos), rec


def back_substitute(records: Sequence[Elimination], final: dict[int, int]) -> dict[int, int]:
    """Undo a chain of eliminations: extend ``final`` (label -> spin of the reduced model) to every label."""
    full = dict(final)
    for rec in reversed(records):
        if rec.kind == "pair":
            full[rec.i] = rec.sign * full[rec.j]
        else:
            full[rec.i] = rec.sign
    return full
