"""Warm-started QAOA on a numpy statevector, and the recursive QAOA (RQAOA) loop around it.

The single solver path of the proof of concept:

* **warm start** - qubit ``q`` starts in ``RY(theta_q)|0>`` with ``theta_q`` from the classical (ADMM) sign and a
  confidence in ``[0, 1]`` (:func:`warm_start_angles`); the mixer rotates about each qubit's warm-start axis, so with
  no QAOA layer the circuit reproduces the warm-start distribution;
* **p = 1 QAOA** per step with angles optimised on the statevector (Nelder-Mead, seeded multistart);
* **recursion** - the largest ``|<Z_i Z_j>|`` (or ``|<Z_i>|``) fixes one variable, the model shrinks by one spin
  (:func:`nanoquant.quantum.ising.eliminate_pair`), and the loop repeats until ``n_stop`` spins are brute-forced.

Samples come from a :class:`Sampler`; the statevector one is exact, the IonQ one lives in ``ionq_backend``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel
from scipy.optimize import minimize

from .ising import (
    Elimination,
    IsingInstance,
    back_substitute,
    bitstring_to_spins,
    cost_scale,
    eliminate_pair,
    eliminate_single,
    index_to_spins,
    spin_table,
    spins_to_bitstring,
)


# ----------------------------------------------------------------------------------------------------
# Warm start
# ----------------------------------------------------------------------------------------------------
def warm_start_angles(signs: Sequence[int], confidence: Sequence[float], eps: float = 0.1) -> list[float]:
    """Per-qubit ``RY`` angles encoding classical signs with a confidence.

    Qubit ``q`` gets ``P(bit 1) = 1/2 - s_q c_q (1/2 - eps)``, so a fully confident spin sits at probability ``eps``
    of being flipped (the regularisation of Egger et al.'s warm-start QAOA) and ``c_q = 0`` is ``|+>``.

    Parameters
    ----------
    signs : sequence of int
        Classical spins ``+-1``.
    confidence : sequence of float
        Confidence in ``[0, 1]`` per spin.
    eps : float
        Regularisation in ``[0, 1/2)``; ``0`` puts confident spins in exact basis states (mixer then cannot move them).

    Returns
    -------
    list of float
        ``theta_q = 2 arcsin(sqrt(P(bit 1)))``.
    """
    s = np.asarray(signs, dtype=np.float64)
    c = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0)
    if s.shape != c.shape:
        raise ValueError("signs and confidence must have the same length")
    p1 = 0.5 - s * c * (0.5 - eps)
    return (2.0 * np.arcsin(np.sqrt(np.clip(p1, 0.0, 1.0)))).tolist()


def warm_state(theta: Sequence[float]) -> np.ndarray:
    """Product state ``prod_q RY(theta_q)|0>`` as a length-``2^n`` complex vector (qubit 0 most significant)."""
    psi = np.ones(1, dtype=np.complex128)
    for t in theta:
        psi = np.kron(psi, np.array([np.cos(t / 2), np.sin(t / 2)], dtype=np.complex128))
    return psi


# ----------------------------------------------------------------------------------------------------
# Circuit layers on the statevector
# ----------------------------------------------------------------------------------------------------
def apply_cost(state: np.ndarray, energies: np.ndarray, gamma: float) -> np.ndarray:
    """``exp(-i gamma H_C)`` on a statevector, ``H_C`` diagonal with the given energies."""
    return state * np.exp(-1j * gamma * energies)


def apply_mixer(state: np.ndarray, theta: Sequence[float], beta: float) -> np.ndarray:
    """Warm-start mixer ``prod_q exp(-i beta (sin theta_q X_q + cos theta_q Z_q))``.

    Each qubit's warm-start state is the ``+1`` eigenstate of its mixer term, so the warm state is invariant.
    """
    n = len(theta)
    psi = state.reshape((2,) * n)
    c, s = np.cos(beta), np.sin(beta)
    for q, t in enumerate(theta):
        H = np.array([[np.cos(t), np.sin(t)], [np.sin(t), -np.cos(t)]], dtype=np.complex128)
        U = c * np.eye(2) - 1j * s * H
        psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)
    return psi.reshape(-1)


def qaoa_state(inst: IsingInstance, theta: Sequence[float], gammas: Sequence[float],
               betas: Sequence[float]) -> np.ndarray:
    """Statevector after the warm start and ``p = len(gammas)`` cost/mixer layers."""
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")
    if len(theta) != inst.n:
        raise ValueError(f"theta has {len(theta)} entries for {inst.n} spins")
    energies = inst.all_energies()
    psi = warm_state(theta)
    for g, b in zip(gammas, betas):
        psi = apply_mixer(apply_cost(psi, energies, g), theta, b)
    return psi


def probabilities(state: np.ndarray) -> np.ndarray:
    """Measurement probabilities of a statevector."""
    p = np.abs(state) ** 2
    return p / p.sum()


def expect_z(probs: np.ndarray, n: int) -> np.ndarray:
    """``<Z_q>`` for every qubit from basis-state probabilities."""
    return probs @ spin_table(n).astype(np.float64)


def expectation(inst: IsingInstance, theta: Sequence[float], gammas: Sequence[float],
                betas: Sequence[float]) -> float:
    """Energy expectation of the QAOA state."""
    probs = probabilities(qaoa_state(inst, theta, gammas, betas))
    return float(probs @ inst.all_energies())


def optimize_angles(inst: IsingInstance, theta: Sequence[float], p: int = 1, seed: int = 0, n_starts: int = 8,
                    maxiter: int = 400) -> tuple[list[float], list[float], float]:
    """Nelder-Mead over ``(gammas, betas)`` from ``n_starts`` seeded random starts on the statevector.

    The cost is normalised by :func:`nanoquant.quantum.ising.cost_scale` so the ``gamma`` search range is
    problem-independent; the returned ``gammas`` are in the units of ``inst``.

    Returns
    -------
    tuple
        ``(gammas, betas, expectation)`` of the best start.
    """
    scale = cost_scale(inst)
    norm = inst.scaled(1.0 / scale)
    energies = norm.all_energies()
    psi0 = warm_state(theta)

    def value(x: np.ndarray) -> float:
        psi = psi0
        for g, b in zip(x[:p], x[p:]):
            psi = apply_mixer(apply_cost(psi, energies, g), theta, b)
        return float(probabilities(psi) @ energies)

    rng = np.random.default_rng(seed)
    best_x, best_v = None, np.inf
    for _ in range(max(1, n_starts)):
        x0 = np.concatenate([rng.uniform(0.0, 2 * np.pi, size=p), rng.uniform(0.0, np.pi, size=p)])
        res = minimize(value, x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-7})
        if res.fun < best_v:
            best_x, best_v = res.x, float(res.fun)
    gammas = (best_x[:p] / scale).tolist()
    betas = best_x[p:].tolist()
    return gammas, betas, expectation(inst, theta, gammas, betas)


# ----------------------------------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------------------------------
class Sampler(Protocol):
    """Anything that returns measurement counts ``{bitstring: shots}`` of a warm-started QAOA circuit."""

    def sample(self, inst: IsingInstance, theta: Sequence[float], gammas: Sequence[float], betas: Sequence[float],
               shots: int, rng: np.random.Generator) -> dict[str, int]:
        ...


class StatevectorSampler:
    """Exact simulator: multinomial shots from the statevector probabilities."""

    def sample(self, inst: IsingInstance, theta: Sequence[float], gammas: Sequence[float], betas: Sequence[float],
               shots: int, rng: np.random.Generator) -> dict[str, int]:
        probs = probabilities(qaoa_state(inst, theta, gammas, betas))
        draws = rng.multinomial(shots, probs)
        return {spins_to_bitstring(index_to_spins(int(i), inst.n)): int(c) for i, c in enumerate(draws) if c > 0}


def correlations(counts: dict[str, int], n: int) -> tuple[np.ndarray, np.ndarray]:
    """``<Z_i>`` and ``<Z_i Z_j>`` (zero diagonal) estimated from measurement counts."""
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("no shots")
    z = np.zeros(n)
    zz = np.zeros((n, n))
    for bits, c in counts.items():
        s = np.asarray(bitstring_to_spins(bits), dtype=np.float64)
        z += c * s
        zz += c * np.outer(s, s)
    z /= total
    zz /= total
    np.fill_diagonal(zz, 0.0)
    return z, zz


# ----------------------------------------------------------------------------------------------------
# Recursive QAOA
# ----------------------------------------------------------------------------------------------------
class RQAOAStep(BaseModel):
    """One recursion step: the model size, the optimised angles, and the variable it fixed."""

    n: int
    gammas: list[float]
    betas: list[float]
    expectation: float
    elimination: Elimination
    correlation: float


class RQAOAResult(BaseModel):
    """Final assignment (indexed by the original labels in ``labels`` order) and the recursion trace."""

    labels: list[int]
    spins: list[int]
    energy: float
    steps: list[RQAOAStep]
    final_brute_force_n: int

    def to_json(self, path: str | Path) -> None:
        """Write the result as JSON."""
        Path(path).write_text(json.dumps(self.model_dump(), indent=1))

    @classmethod
    def from_json(cls, path: str | Path) -> RQAOAResult:
        """Read a result written by :meth:`to_json`."""
        return cls.model_validate(json.loads(Path(path).read_text()))


def rqaoa(inst: IsingInstance, theta: Sequence[float], sampler: Sampler, shots: int, seed: int = 0, n_stop: int = 4,
          p: int = 1, n_starts: int = 8) -> RQAOAResult:
    """Warm-started recursive QAOA.

    Parameters
    ----------
    inst : IsingInstance
        Model to minimise.
    theta : sequence of float
        Warm-start angles, one per spin (see :func:`warm_start_angles`).
    sampler : Sampler
        Source of measurement counts (statevector or hardware).
    shots : int
        Shots per recursion step.
    seed : int
        Seeds the angle multistart and the statevector sampler.
    n_stop : int
        Remaining spins that are brute-forced instead of recursed.
    p : int
        QAOA depth per step.
    n_starts : int
        Nelder-Mead multistart count per step.

    Returns
    -------
    RQAOAResult
        Spins in the order of ``inst.labels``, their energy under ``inst``, and the per-step trace.
    """
    rng = np.random.default_rng(seed)
    cur, th = inst, list(theta)
    records: list[Elimination] = []
    steps: list[RQAOAStep] = []
    step = 0
    while cur.n > n_stop:
        gammas, betas, val = optimize_angles(cur, th, p=p, seed=seed + step, n_starts=n_starts)
        counts = sampler.sample(cur, th, gammas, betas, shots, rng)
        z, zz = correlations(counts, cur.n)
        i_s = int(np.argmax(np.abs(z)))
        iu, ju = np.triu_indices(cur.n, 1)
        k = int(np.argmax(np.abs(zz[iu, ju])))
        i_p, j_p, v_p = int(iu[k]), int(ju[k]), float(zz[iu[k], ju[k]])
        if abs(v_p) >= abs(z[i_s]):
            sign = 1 if v_p >= 0 else -1
            cur, rec = eliminate_pair(cur, i_p, j_p, sign)
            del th[i_p]
            corr = v_p
        else:
            sign = 1 if z[i_s] >= 0 else -1
            cur, rec = eliminate_single(cur, i_s, sign)
            del th[i_s]
            corr = float(z[i_s])
        records.append(rec)
        steps.append(RQAOAStep(n=cur.n + 1, gammas=gammas, betas=betas, expectation=val, elimination=rec,
                               correlation=corr))
        step += 1
    best_s, _ = cur.brute_force()
    full = back_substitute(records, dict(zip(cur.labels, best_s)))
    spins = [int(full[lab]) for lab in inst.labels]
    return RQAOAResult(labels=list(inst.labels), spins=spins, energy=inst.energy(spins), steps=steps,
                       final_brute_force_n=cur.n)
