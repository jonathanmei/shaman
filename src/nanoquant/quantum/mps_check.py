"""Matrix-product-state cross-check of the warm-started QAOA circuit with quimb (optional ``quantum`` extra).

Builds exactly the circuit of :mod:`nanoquant.quantum.ionq_backend` on a ``quimb.tensor.CircuitMPS`` with a bond
dimension cap and returns the energy and the ``<Z_i>``, ``<Z_i Z_j>`` correlations by local expectation values.
quimb's conventions match qiskit's: ``RZ(t) = exp(-i t Z / 2)``, ``RZZ(t) = exp(-i t Z Z / 2)``. Non-adjacent
two-qubit gates are applied through quimb's automatic swap-and-split, which is where the bond dimension matters on an
all-to-all cost layer.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .ising import IsingInstance

try:  # pragma: no cover - exercised only with the optional extra installed
    import quimb as qu
    import quimb.tensor as qtn
except ImportError:  # pragma: no cover
    qu = qtn = None  # type: ignore[assignment]


def _require_quimb() -> None:
    if qtn is None:
        raise ImportError("quimb is required: install the 'quantum' extra (uv sync --extra quantum)")


def build_circuit_mps(inst: IsingInstance, theta: Sequence[float], gammas: Sequence[float], betas: Sequence[float],
                      max_bond: int | None = 64, cutoff: float = 1e-10):
    """The warm-started QAOA circuit as a ``CircuitMPS`` (bond dimension capped at ``max_bond``)."""
    _require_quimb()
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")
    n = inst.n
    if len(theta) != n:
        raise ValueError(f"theta has {len(theta)} entries for {n} spins")
    h, J = inst.h_array(), inst.J_array()
    circ = qtn.CircuitMPS(n, max_bond=max_bond, cutoff=cutoff)
    for q, t in enumerate(theta):
        circ.apply_gate("RY", float(t), q)
    for g, b in zip(gammas, betas):
        for i in range(n):
            if h[i] != 0.0:
                circ.apply_gate("RZ", float(2.0 * g * h[i]), i)
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0.0:
                    circ.apply_gate("RZZ", float(2.0 * g * J[i, j]), i, j)
        for q, t in enumerate(theta):
            circ.apply_gate("RY", float(-t), q)
            circ.apply_gate("RZ", float(2.0 * b), q)
            circ.apply_gate("RY", float(t), q)
    return circ


def mps_expectations(inst: IsingInstance, theta: Sequence[float], gammas: Sequence[float], betas: Sequence[float],
                     max_bond: int | None = 64, pairs: Sequence[tuple[int, int]] | None = None,
                     cutoff: float = 1e-10) -> tuple[float, np.ndarray, np.ndarray, int]:
    """``(energy, z, zz, max bond dimension reached)`` of the MPS-simulated circuit.

    Parameters
    ----------
    inst : IsingInstance
        Model.
    theta, gammas, betas : sequence of float
        Warm-start and QAOA angles.
    max_bond : int or None
        Bond dimension cap (``None`` = exact, exponential).
    pairs : sequence of (int, int), optional
        Pairs to evaluate (``None`` = all); the energy sums those pairs only.
    cutoff : float
        Singular-value cutoff of the MPS compression.
    """
    _require_quimb()
    circ = build_circuit_mps(inst, theta, gammas, betas, max_bond=max_bond, cutoff=cutoff)
    n = inst.n
    Z = qu.pauli("Z")
    ZZ = Z & Z
    z = np.array([float(np.real(circ.local_expectation(Z, (i,)))) for i in range(n)])
    pair_list = [(i, j) for i in range(n) for j in range(i + 1, n)] if pairs is None else [tuple(p) for p in pairs]
    zz = np.zeros((n, n))
    for i, j in pair_list:
        v = float(np.real(circ.local_expectation(ZZ, (i, j))))
        zz[i, j] = zz[j, i] = v
    energy = inst.const + float(inst.h_array() @ z) + 0.5 * float(np.sum(inst.J_array() * zz))
    return energy, z, zz, int(circ.psi.max_bond())
