"""Tests for the warm-started QAOA statevector simulator and the recursive (RQAOA) loop (numpy only)."""

import numpy as np
import pytest

from nanoquant.quantum import ising as I
from nanoquant.quantum import rqaoa as Q


def _dense_instance(n: int, seed: int) -> I.IsingInstance:
    rng = np.random.default_rng(seed)
    h = 0.3 * rng.normal(size=n)
    J = rng.normal(size=(n, n))
    J = np.triu(J, 1)
    J = J + J.T
    return I.IsingInstance(h=h.tolist(), J=J.tolist())


def test_warm_start_angles_encode_sign_and_confidence():
    theta = Q.warm_start_angles([1, -1, 1], [1.0, 1.0, 0.0], eps=0.1)
    p1 = np.sin(np.asarray(theta) / 2) ** 2  # probability of bit 1 (spin -1)
    assert p1[0] == pytest.approx(0.1)  # confident +1 -> mostly bit 0
    assert p1[1] == pytest.approx(0.9)  # confident -1 -> mostly bit 1
    assert p1[2] == pytest.approx(0.5)  # no confidence -> |+>


def test_p0_reproduces_the_warm_start_bits():
    signs = [1, -1, -1, 1, 1]
    theta = Q.warm_start_angles(signs, [1.0, 0.8, 0.6, 0.9, 1.0], eps=0.1)
    inst = _dense_instance(5, seed=0)
    state = Q.qaoa_state(inst, theta, gammas=[], betas=[])
    probs = Q.probabilities(state)
    assert probs.sum() == pytest.approx(1.0)
    assert I.index_to_spins(int(np.argmax(probs)), 5) == signs
    z = Q.expect_z(probs, 5)
    assert np.all(np.sign(z) == np.asarray(signs))
    assert z[0] == pytest.approx(0.8)  # c = 1, eps = 0.1 -> <Z> = 1 - 2 eps


def test_mixer_leaves_warm_state_invariant_up_to_phase():
    theta = Q.warm_start_angles([1, -1, 1], [0.7, 0.2, 0.9])
    psi = Q.warm_state(theta)
    out = Q.apply_mixer(psi, theta, beta=0.37)
    overlap = abs(np.vdot(psi, out))
    assert overlap == pytest.approx(1.0)


def test_cost_layer_is_diagonal_phase():
    inst = _dense_instance(4, seed=1)
    theta = Q.warm_start_angles([1] * 4, [0.0] * 4)
    psi = Q.warm_state(theta)
    out = Q.apply_cost(psi, inst.all_energies(), gamma=0.5)
    assert np.allclose(np.abs(out), np.abs(psi))
    assert np.allclose(out, psi * np.exp(-0.5j * inst.all_energies()))


def test_expectation_matches_probability_average():
    inst = _dense_instance(5, seed=2)
    theta = Q.warm_start_angles([1, 1, -1, -1, 1], [0.5] * 5)
    e = Q.expectation(inst, theta, gammas=[0.4], betas=[0.3])
    probs = Q.probabilities(Q.qaoa_state(inst, theta, [0.4], [0.3]))
    assert e == pytest.approx(float(probs @ inst.all_energies()))


def test_optimized_p1_beats_p0_from_an_unconfident_start():
    inst = _dense_instance(6, seed=3)
    theta = Q.warm_start_angles([1] * 6, [0.0] * 6)  # plain |+>^n
    e0 = Q.expectation(inst, theta, [], [])
    gammas, betas, e1 = Q.optimize_angles(inst, theta, p=1, seed=0, n_starts=4)
    assert len(gammas) == 1 and len(betas) == 1
    assert e1 < e0 - 1e-6
    assert Q.expectation(inst, theta, gammas, betas) == pytest.approx(e1)


def test_statevector_sampler_counts_and_correlations():
    inst = _dense_instance(4, seed=4)
    theta = Q.warm_start_angles([1, -1, 1, -1], [1.0] * 4, eps=0.0)  # a product basis state
    counts = Q.StatevectorSampler().sample(inst, theta, [], [], shots=100, rng=np.random.default_rng(0))
    assert counts == {"0101": 100}
    z, zz = Q.correlations(counts, 4)
    assert np.allclose(z, [1, -1, 1, -1])
    assert zz[0, 1] == pytest.approx(-1.0) and zz[0, 2] == pytest.approx(1.0)
    assert np.allclose(np.diag(zz), 0.0)


@pytest.mark.parametrize("seed", [0, 2, 5])
def test_rqaoa_recovers_the_optimum_from_a_mostly_right_warm_start(seed):
    """The ADMM-like case: most bits right and confident, a few wrong ones flagged by low confidence."""
    inst = _dense_instance(8, seed=seed)
    best_s, best_e = inst.brute_force()
    start = [-s if q < 3 else s for q, s in enumerate(best_s)]
    theta = Q.warm_start_angles(start, [0.2] * 3 + [0.9] * 5)
    assert inst.energy(start) > best_e
    res = Q.rqaoa(inst, theta, Q.StatevectorSampler(), shots=4000, seed=0, n_stop=4, p=1)
    assert res.energy == pytest.approx(best_e)
    assert inst.energy(res.spins) == pytest.approx(best_e)
    assert res.labels == list(range(8))
    assert len(res.steps) == 4  # 8 -> 4 spins, one elimination per step
    assert res.steps[0].n == 8 and res.steps[-1].n == 5
    assert res.final_brute_force_n == 4


@pytest.mark.parametrize("seed", [1, 3, 4])
def test_rqaoa_recovers_the_optimum_from_plus_state(seed):
    inst = _dense_instance(8, seed=seed)
    _, best_e = inst.brute_force()
    theta = Q.warm_start_angles([1] * 8, [0.0] * 8)
    res = Q.rqaoa(inst, theta, Q.StatevectorSampler(), shots=4000, seed=0, n_stop=4, p=1)
    assert res.energy == pytest.approx(best_e)


@pytest.mark.parametrize("p", [1, 2])
def test_pauli_correlator_rqaoa_recovers_the_optimum(p):
    inst = _dense_instance(8, seed=2)
    best_s, best_e = inst.brute_force()
    start = [-s if q < 3 else s for q, s in enumerate(best_s)]
    theta = Q.warm_start_angles(start, [0.2] * 3 + [0.9] * 5)
    # p = 1 is exact at any k; at p = 2 the second-order truncation is accurate to ~1e-4 on these couplings
    res = Q.rqaoa(inst, theta, Q.PauliPropCorrelator(k=2), shots=0, seed=0, n_stop=4, p=p, n_starts=2)
    assert res.energy == pytest.approx(best_e)
    assert all(step.reoptimized for step in res.steps)


def test_carried_and_fixed_angles_paths_run():
    inst = _dense_instance(7, seed=3)
    theta = Q.warm_start_angles([1] * 7, [0.0] * 7)
    res = Q.rqaoa(inst, theta, Q.PauliPropCorrelator(k=1, pair_top=3), shots=0, seed=0, n_stop=3, p=1,
                  reoptimize_every=2, n_starts=2)
    assert [s.reoptimized for s in res.steps] == [True, False, True, False]
    assert res.steps[1].expectation is None and res.steps[0].expectation is not None
    fixed = Q.normalized_angles(inst, res.steps[0].gammas, res.steps[0].betas)
    res2 = Q.rqaoa(inst, theta, Q.StatevectorSampler(), shots=500, seed=0, n_stop=3, fixed_angles=fixed.tolist())
    assert not any(s.reoptimized for s in res2.steps)
    assert inst.energy(res2.spins) == pytest.approx(res2.energy)
    assert res.seconds > 0 and all(s.seconds >= 0 for s in res.steps)


def test_statevector_refuses_large_instances():
    n = Q.MAX_STATEVECTOR_N + 1
    inst = I.IsingInstance(h=[0.1] * n, J=np.zeros((n, n)).tolist())
    with pytest.raises(ValueError):
        Q.expectation(inst, [0.0] * n, [0.1], [0.1])
    _, z, _ = __import__("nanoquant.quantum.pauli_prop", fromlist=["expectations"]).expectations(
        inst, [0.0] * n, [0.1], [0.1], k=0, pairs=[])
    assert np.allclose(z, 1.0)  # all spins up, no couplings: <Z> = cos(0) = 1 regardless of angles


def test_rqaoa_result_json_round_trip(tmp_path):
    inst = _dense_instance(5, seed=6)
    theta = Q.warm_start_angles([1] * 5, [0.0] * 5)
    res = Q.rqaoa(inst, theta, Q.StatevectorSampler(), shots=500, seed=1, n_stop=3, p=1)
    path = tmp_path / "solution.json"
    res.to_json(path)
    assert Q.RQAOAResult.from_json(path) == res
