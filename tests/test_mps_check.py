"""The quimb MPS circuit must reproduce the statevector (skipped without the ``quantum`` extra)."""

import numpy as np
import pytest

from nanoquant.quantum import ising as I
from nanoquant.quantum import rqaoa as Q

pytest.importorskip("quimb")
from nanoquant.quantum import mps_check as M


def _inst(n: int, seed: int, coupling: float) -> I.IsingInstance:
    rng = np.random.default_rng(seed)
    h = rng.normal(size=n)
    J = coupling * rng.normal(size=(n, n))
    J = np.triu(J, 1)
    J = J + J.T
    return I.IsingInstance(h=h.tolist(), J=J.tolist(), const=0.2)


def _reference(inst, theta, gammas, betas):
    probs = Q.probabilities(Q.qaoa_state(inst, theta, gammas, betas))
    S = I.spin_table(inst.n).astype(float)
    zz = (S * probs[:, None]).T @ S
    np.fill_diagonal(zz, 0.0)
    return float(probs @ inst.all_energies()), probs @ S, zz


@pytest.mark.parametrize("p", [1, 2])
def test_mps_with_large_bond_matches_statevector(p):
    n = 8
    inst = _inst(n, seed=p, coupling=0.5)
    rng = np.random.default_rng(p)
    theta = Q.warm_start_angles(rng.choice([-1, 1], size=n), rng.uniform(0, 1, size=n))
    gammas, betas = rng.uniform(0, 1.5, size=p).tolist(), rng.uniform(0, np.pi, size=p).tolist()
    e_ref, z_ref, zz_ref = _reference(inst, theta, gammas, betas)
    e, z, zz, chi = M.mps_expectations(inst, theta, gammas, betas, max_bond=64)
    assert chi <= 16  # 8 qubits: exact MPS needs at most 2^4
    assert e == pytest.approx(e_ref, abs=1e-8)
    assert np.allclose(z, z_ref, atol=1e-8)
    assert np.allclose(zz, zz_ref, atol=1e-8)


def test_small_bond_is_approximate_but_close_for_weak_couplings():
    n = 8
    inst = _inst(n, seed=5, coupling=0.05)
    rng = np.random.default_rng(5)
    theta = Q.warm_start_angles(rng.choice([-1, 1], size=n), rng.uniform(0, 1, size=n))
    gammas, betas = [0.8, 1.1], [0.7, 2.0]
    _, _, zz_ref = _reference(inst, theta, gammas, betas)
    e, _, zz, chi = M.mps_expectations(inst, theta, gammas, betas, max_bond=2, pairs=[(0, 7), (3, 4)])
    assert chi <= 2
    assert abs(e - (inst.const + inst.h_array() @ _reference(inst, theta, gammas, betas)[1]
                    + inst.J[0][7] * zz_ref[0, 7] + inst.J[3][4] * zz_ref[3, 4])) < 5e-2
    assert abs(zz[0, 7] - zz_ref[0, 7]) < 5e-2 and zz[1, 2] == 0.0
