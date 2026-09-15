"""The product-operator (Pauli propagation) engine against the statevector: exact where it must be, controlled
where it truncates (numpy only)."""

import numpy as np
import pytest

from nanoquant.quantum import ising as I
from nanoquant.quantum import pauli_prop as PP
from nanoquant.quantum import rqaoa as Q


def _inst(n: int, seed: int, coupling: float) -> I.IsingInstance:
    """Fields of order one, couplings of order ``coupling`` (the tile regime is ~0.03-0.1)."""
    rng = np.random.default_rng(seed)
    h = rng.normal(size=n)
    J = coupling * rng.normal(size=(n, n))
    J = np.triu(J, 1)
    J = J + J.T
    return I.IsingInstance(h=h.tolist(), J=J.tolist(), const=0.3)


def _reference(inst, theta, gammas, betas):
    probs = Q.probabilities(Q.qaoa_state(inst, theta, gammas, betas))
    S = I.spin_table(inst.n).astype(float)
    z = probs @ S
    zz = (S * probs[:, None]).T @ S
    np.fill_diagonal(zz, 0.0)
    return float(probs @ inst.all_energies()), z, zz


def _angles(seed: int, p: int, theta_n: int):
    rng = np.random.default_rng(seed)
    theta = Q.warm_start_angles(rng.choice([-1, 1], size=theta_n), rng.uniform(0, 1, size=theta_n))
    return theta, rng.uniform(0, 1.5, size=p).tolist(), rng.uniform(0, np.pi, size=p).tolist()


@pytest.mark.parametrize("n,seed", [(5, 0), (7, 1), (9, 2)])
def test_p1_is_exact_without_any_truncation(n, seed):
    inst = _inst(n, seed, coupling=0.8)  # strong couplings: p = 1 must still be exact
    theta, g, b = _angles(seed, 1, n)
    e_ref, z_ref, zz_ref = _reference(inst, theta, g, b)
    e, z, zz = PP.expectations(inst, theta, g, b, k=0)
    assert e == pytest.approx(e_ref, abs=1e-10)
    assert np.allclose(z, z_ref, atol=1e-10)
    assert np.allclose(zz, zz_ref, atol=1e-10)


@pytest.mark.parametrize("p", [2, 3])
def test_untruncated_engine_is_exact_at_depth(p):
    n = 5
    inst = _inst(n, seed=3, coupling=0.7)
    theta, g, b = _angles(4, p, n)
    e_ref, z_ref, zz_ref = _reference(inst, theta, g, b)
    e, z, zz = PP.expectations(inst, theta, g, b, k=(p - 1) * n)  # k >= (p - 1)(n - |roots|): no truncation
    assert e == pytest.approx(e_ref, abs=1e-9)
    assert np.allclose(z, z_ref, atol=1e-9)
    assert np.allclose(zz, zz_ref, atol=1e-9)


def test_truncation_error_shrinks_with_order_in_the_weak_coupling_regime():
    n = 8
    inst = _inst(n, seed=5, coupling=0.08)
    theta, g, b = _angles(6, 2, n)
    e_ref, z_ref, zz_ref = _reference(inst, theta, g, b)
    errs = []
    for k in (0, 1, 2):
        e, z, zz = PP.expectations(inst, theta, g, b, k=k)
        errs.append(max(abs(e - e_ref), np.abs(z - z_ref).max(), np.abs(zz - zz_ref).max()))
    # measured ladder on this instance: 8e-2, 4e-3, 2e-4 (about 20x per order), machine precision at k = 6
    assert errs[0] > 5 * errs[1] > 25 * errs[2]
    assert errs[1] < 1e-2 and errs[2] < 1e-3


def test_strong_couplings_need_higher_order():
    n = 7
    inst = _inst(n, seed=7, coupling=0.8)
    theta, g, b = _angles(8, 2, n)
    _, z_ref, _ = _reference(inst, theta, g, b)
    _, z1, _ = PP.expectations(inst, theta, g, b, k=1)
    _, z_all, _ = PP.expectations(inst, theta, g, b, k=n)  # p = 2: one truncated layer, k = n is exact
    assert np.abs(z1 - z_ref).max() > 1e-2  # documents the regime where k = 1 is not enough
    assert np.allclose(z_all, z_ref, atol=1e-9)


def test_active_site_restriction_and_pair_subset():
    n = 10
    inst = _inst(n, seed=9, coupling=0.05)
    theta, g, b = _angles(10, 2, n)
    _, z_ref, zz_ref = _reference(inst, theta, g, b)
    pairs = PP.candidate_pairs(inst, per_spin=3)
    assert all(i < j for i, j in pairs) and len(pairs) <= 3 * n
    e, z, zz = PP.expectations(inst, theta, g, b, k=1, max_active=4, pairs=pairs)
    _, _, zz_full = PP.expectations(inst, theta, g, b, k=1, pairs=pairs)
    err_active = max(abs(zz[i, j] - zz_ref[i, j]) for i, j in pairs)
    err_full = max(abs(zz_full[i, j] - zz_ref[i, j]) for i, j in pairs)
    assert err_full < err_active < 3e-2  # restricting the active sites costs accuracy, controllably
    for i, j in pairs:
        assert zz[j, i] == zz[i, j]
    mask = np.zeros((n, n), bool)
    for i, j in pairs:
        mask[i, j] = mask[j, i] = True
    assert np.all(zz[~mask] == 0.0)
    partial = inst.const + inst.h_array() @ z + sum(inst.J[i][j] * zz[i, j] for i, j in pairs)
    assert e == pytest.approx(partial)
    assert np.allclose(z, z_ref, atol=1e-2)


def test_expectations_are_real_and_batched_consistently():
    n = 6
    inst = _inst(n, seed=11, coupling=0.1)
    theta, g, b = _angles(12, 2, n)
    _, z, zz = PP.expectations(inst, theta, g, b, k=1)
    # one pair at a time must agree with the batched evaluation
    for i, j in [(0, 1), (2, 5)]:
        _, _, zz1 = PP.expectations(inst, theta, g, b, k=1, pairs=[(i, j)])
        assert zz1[i, j] == pytest.approx(zz[i, j], abs=1e-12)
    assert np.all(np.abs(z) <= 1 + 1e-9)
