"""Tests for the Ising instance container, brute force, and the RQAOA variable eliminations (numpy only)."""

import itertools

import numpy as np
import pytest

from nanoquant.quantum import ising as I


def _random_instance(n: int, seed: int) -> I.IsingInstance:
    rng = np.random.default_rng(seed)
    h = rng.normal(size=n)
    J = rng.normal(size=(n, n))
    J = np.triu(J, 1)
    J = J + J.T
    return I.IsingInstance(h=h.tolist(), J=J.tolist(), const=float(rng.normal()))


def _direct_energy(inst: I.IsingInstance, s) -> float:
    s = np.asarray(s, dtype=float)
    J = np.asarray(inst.J)
    return inst.const + float(np.dot(inst.h, s)) + 0.5 * float(s @ J @ s)


def test_energy_matches_direct_formula():
    inst = _random_instance(6, seed=0)
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = rng.choice([-1, 1], size=6)
        assert inst.energy(s) == pytest.approx(_direct_energy(inst, s))


def test_all_energies_index_convention_and_brute_force():
    inst = _random_instance(6, seed=2)
    e = inst.all_energies()
    assert e.shape == (64,)
    for idx in range(64):
        s = I.index_to_spins(idx, 6)
        assert I.spins_to_index(s) == idx
        assert e[idx] == pytest.approx(inst.energy(s))
    # qubit 0 is the leftmost bit; bit 1 <-> spin -1
    assert I.index_to_spins(0b100000, 6)[0] == -1
    assert I.spins_to_bitstring([-1, 1, 1, 1, 1, 1]) == "100000"
    best_s, best_e = inst.brute_force()
    ref = min((_direct_energy(inst, s), s) for s in itertools.product([-1, 1], repeat=6))
    assert best_e == pytest.approx(ref[0])
    assert inst.energy(best_s) == pytest.approx(ref[0])


def test_json_round_trip(tmp_path):
    inst = _random_instance(5, seed=3)
    inst.meta = {"layer": "27.mlp.down_proj", "tile": [[1, 2], [3, 4]]}
    path = tmp_path / "inst.json"
    inst.to_json(path)
    back = I.IsingInstance.from_json(path)
    assert back == inst


def test_validation_rejects_asymmetric_or_wrong_size():
    with pytest.raises(ValueError):
        I.IsingInstance(h=[0.0, 0.0], J=[[0.0, 1.0], [0.5, 0.0]])
    with pytest.raises(ValueError):
        I.IsingInstance(h=[0.0, 0.0, 0.0], J=[[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError):
        I.IsingInstance(h=[0.0, 0.0], J=[[1.0, 0.0], [0.0, 0.0]])


@pytest.mark.parametrize("sign", [1, -1])
def test_pair_elimination_preserves_energies(sign):
    inst = _random_instance(6, seed=4)
    i, j = 4, 1
    red, rec = I.eliminate_pair(inst, i, j, sign)
    assert red.n == 5
    assert rec.kind == "pair" and rec.i == 4 and rec.j == 1 and rec.sign == sign
    assert red.labels == [0, 1, 2, 3, 5]
    for s_red in itertools.product([-1, 1], repeat=5):
        full = dict(zip(red.labels, s_red))
        full[i] = sign * full[j]
        s_full = [full[k] for k in range(6)]
        assert red.energy(s_red) == pytest.approx(inst.energy(s_full))


@pytest.mark.parametrize("sign", [1, -1])
def test_single_elimination_preserves_energies(sign):
    inst = _random_instance(5, seed=5)
    red, rec = I.eliminate_single(inst, 2, sign)
    assert red.n == 4 and rec.kind == "single" and rec.i == 2 and rec.j is None
    for s_red in itertools.product([-1, 1], repeat=4):
        full = dict(zip(red.labels, s_red))
        full[2] = sign
        assert red.energy(s_red) == pytest.approx(inst.energy([full[k] for k in range(5)]))


def test_back_substitution_reconstructs_full_assignment():
    inst = _random_instance(6, seed=6)
    red1, r1 = I.eliminate_pair(inst, 5, 0, -1)  # s5 = -s0
    red2, r2 = I.eliminate_single(red1, 2, 1)  # label 2 = +1
    red3, r3 = I.eliminate_pair(red2, 0, 1, 1)  # positions 0, 1 of labels [0, 1, 3, 4]: s0 = s1
    assert red3.labels == [1, 3, 4]
    final = dict(zip(red3.labels, [1, -1, 1]))
    full = I.back_substitute([r1, r2, r3], final)
    assert sorted(full) == list(range(6))
    assert full[2] == 1
    assert full[5] == -full[0]
    assert full[r3.i] == r3.sign * full[r3.j]
    # energies agree through the chain
    assert red3.energy([final[k] for k in red3.labels]) == pytest.approx(inst.energy([full[k] for k in range(6)]))


def test_scaled_instance_scales_energies():
    inst = _random_instance(4, seed=7)
    sc = inst.scaled(0.25)
    s = [1, -1, -1, 1]
    assert sc.energy(s) == pytest.approx(0.25 * inst.energy(s))
    assert I.cost_scale(sc) == pytest.approx(0.25 * I.cost_scale(inst))
