"""Warm-started QAOA circuits as qiskit circuits, sampled on IonQ backends (optional ``quantum`` extra).

The circuit implements exactly :func:`nanoquant.quantum.rqaoa.qaoa_state`:

* warm start: ``RY(theta_q)`` on every qubit;
* cost layer ``exp(-i gamma H_C)``: ``RZ(2 gamma h_i)`` and ``RZZ(2 gamma J_ij)`` (``RZ(l) = exp(-i l Z / 2)``);
* mixer ``exp(-i beta (sin theta_q X_q + cos theta_q Z_q)) = RY(theta_q) RZ(2 beta) RY(-theta_q)``.

qiskit reports bitstrings little-endian (qubit 0 last); :func:`to_counts` reverses them into the convention of
:mod:`nanoquant.quantum.ising` (qubit 0 first).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from .ising import IsingInstance

try:  # pragma: no cover - exercised only with the optional extra installed
    from qiskit import QuantumCircuit
except ImportError:  # pragma: no cover
    QuantumCircuit = None  # type: ignore[assignment]


def _require_qiskit() -> None:
    if QuantumCircuit is None:
        raise ImportError("qiskit is required: install the 'quantum' extra (uv sync --extra quantum)")


def build_circuit(inst: IsingInstance, theta: Sequence[float], gammas: Sequence[float], betas: Sequence[float],
                  measure: bool = True):
    """The warm-started QAOA circuit of ``inst`` (see the module docstring for the gate decomposition).

    Parameters
    ----------
    inst : IsingInstance
        Model.
    theta : sequence of float
        Warm-start angles.
    gammas, betas : sequence of float
        QAOA angles, ``p = len(gammas)``.
    measure : bool
        Append measurements of every qubit.

    Returns
    -------
    qiskit.QuantumCircuit
    """
    _require_qiskit()
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")
    n = inst.n
    if len(theta) != n:
        raise ValueError(f"theta has {len(theta)} entries for {n} spins")
    h, J = inst.h_array(), inst.J_array()
    qc = QuantumCircuit(n, n if measure else 0)
    for q, t in enumerate(theta):
        qc.ry(float(t), q)
    for g, b in zip(gammas, betas):
        for i in range(n):
            if h[i] != 0.0:
                qc.rz(float(2.0 * g * h[i]), i)
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0.0:
                    qc.rzz(float(2.0 * g * J[i, j]), i, j)
        for q, t in enumerate(theta):
            qc.ry(float(-t), q)
            qc.rz(float(2.0 * b), q)
            qc.ry(float(t), q)
    if measure:
        qc.measure(range(n), range(n))
    return qc


def to_counts(qiskit_counts: dict[str, int], n: int) -> dict[str, int]:
    """Convert qiskit's little-endian count keys into qubit-0-first bitstrings."""
    out: dict[str, int] = {}
    for key, c in qiskit_counts.items():
        bits = key.replace(" ", "")[-n:][::-1]
        out[bits] = out.get(bits, 0) + int(c)
    return out


def two_qubit_gate_count(inst: IsingInstance, p: int = 1) -> int:
    """Number of ``RZZ`` gates per layer times ``p`` (the depth budget check for hardware)."""
    J = inst.J_array()
    return int(p * np.count_nonzero(np.triu(J, 1)))


class IonQSampler:
    """Sample QAOA circuits on an IonQ backend through ``qiskit-ionq``.

    Parameters
    ----------
    target : str
        ``"simulator"``, ``"qpu.aria-1"``, ``"qpu.forte-1"``, ... (the IonQ backend name without the ``ionq_``
        prefix).
    noise_model : str or None
        For the simulator: ``"aria-1"``, ``"forte-1"``, ... (``None`` = ideal).
    token : str or None
        API key; defaults to ``IONQ_API_KEY``, then to qiskit-ionq's own ``QISKIT_IONQ_API_TOKEN``.
    """

    def __init__(self, target: str = "simulator", noise_model: str | None = None, token: str | None = None):
        _require_qiskit()
        from qiskit_ionq import IonQProvider

        provider = IonQProvider(token or os.environ.get("IONQ_API_KEY"))
        self.target = target
        self.noise_model = noise_model
        self.backend = provider.get_backend(f"ionq_{target}")
        self.job_ids: list[str] = []

    def sample(self, inst: IsingInstance, theta: Sequence[float], gammas: Sequence[float], betas: Sequence[float],
               shots: int, rng: np.random.Generator) -> dict[str, int]:
        """Submit one circuit and return counts (qubit 0 first). ``rng`` is unused (hardware randomness)."""
        qc = build_circuit(inst, theta, gammas, betas)
        kwargs = {"shots": int(shots)}
        if self.noise_model and self.target == "simulator":
            kwargs["noise_model"] = self.noise_model
        job = self.backend.run(qc, **kwargs)
        self.job_ids.append(str(job.job_id()))
        return to_counts(job.result().get_counts(), inst.n)
