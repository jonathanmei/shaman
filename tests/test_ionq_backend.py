"""The qiskit circuit must reproduce the numpy statevector exactly (skipped without the ``quantum`` extra)."""

import numpy as np
import pytest

from nanoquant.quantum import ising as I
from nanoquant.quantum import rqaoa as Q

qiskit = pytest.importorskip("qiskit")
from qiskit.quantum_info import Statevector

from nanoquant.quantum import ionq_backend as B


def _inst(n: int, seed: int) -> I.IsingInstance:
    rng = np.random.default_rng(seed)
    h = 0.3 * rng.normal(size=n)
    J = rng.normal(size=(n, n))
    J = np.triu(J, 1)
    J = J + J.T
    return I.IsingInstance(h=h.tolist(), J=J.tolist(), const=1.7)


def _qiskit_probs(qc, n: int) -> np.ndarray:
    """Probabilities re-indexed to qubit-0-most-significant order."""
    sv = Statevector(qc)
    probs = np.zeros(2 ** n)
    for idx, amp in enumerate(sv.data):  # qiskit index: qubit 0 is the least significant bit
        bits = format(idx, f"0{n}b")[::-1]
        probs[int(bits, 2)] += abs(amp) ** 2
    return probs


@pytest.mark.parametrize("p", [0, 1, 2])
def test_circuit_matches_statevector(p):
    n = 5
    inst = _inst(n, seed=p)
    theta = Q.warm_start_angles([1, -1, 1, 1, -1], [0.9, 0.3, 0.0, 0.6, 1.0])
    rng = np.random.default_rng(1)
    gammas, betas = rng.uniform(0, 1, p).tolist(), rng.uniform(0, np.pi, p).tolist()
    qc = B.build_circuit(inst, theta, gammas, betas, measure=False)
    ref = Q.probabilities(Q.qaoa_state(inst, theta, gammas, betas))
    assert np.allclose(_qiskit_probs(qc, n), ref, atol=1e-10)


def test_counts_conversion_and_gate_count():
    inst = _inst(4, seed=3)
    assert B.to_counts({"0010": 7, "1000": 3}, 4) == {"0100": 7, "0001": 3}
    assert B.two_qubit_gate_count(inst, p=1) == 6
    qc = B.build_circuit(inst, [0.0] * 4, [0.5], [0.2])
    assert qc.count_ops()["rzz"] == 6 and qc.count_ops()["measure"] == 4
