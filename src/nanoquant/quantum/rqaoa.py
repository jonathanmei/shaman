"""Warm-started QAOA and the recursive (RQAOA) loop around it, over a pluggable correlator.

The single solver path of the proof of concept:

* **warm start** - qubit ``q`` starts in ``RY(theta_q)|0>`` with ``theta_q`` from the classical (ADMM) sign and a
  confidence in ``[0, 1]`` (:func:`warm_start_angles`); the mixer rotates about each qubit's warm-start axis, so with
  no QAOA layer the circuit reproduces the warm-start distribution;
* **QAOA** at depth ``p`` with angles optimised on the chosen correlator (Nelder-Mead, seeded multistart, angles
  carried between recursion steps);
* **recursion** - the largest ``|<Z_i Z_j>|`` (or ``|<Z_i>|``) fixes one variable, the model shrinks by one spin
  (:func:`nanoquant.quantum.ising.eliminate_pair`), and the loop repeats until ``n_stop`` spins are brute-forced.

A :class:`Correlator` supplies the energy expectation and the correlations: :class:`StatevectorCorrelator` (exact,
``n <= 22``, also wraps any counts :class:`Sampler` such as the IonQ backend) or
:class:`PauliPropCorrelator` (:mod:`nanoquant.quantum.pauli_prop`, any ``n``, exact at ``p = 1``).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

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

MAX_STATEVECTOR_N = 22


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


def _check_statevector_size(n: int) -> None:
    if n > MAX_STATEVECTOR_N:
        raise ValueError(f"statevector simulation is limited to {MAX_STATEVECTOR_N} spins (got {n}); "
                         "use PauliPropCorrelator")


def qaoa_state(inst: IsingInstance, theta: Sequence[float], gammas: Sequence[float],
               betas: Sequence[float]) -> np.ndarray:
    """Statevector after the warm start and ``p = len(gammas)`` cost/mixer layers."""
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")
    if len(theta) != inst.n:
        raise ValueError(f"theta has {len(theta)} entries for {inst.n} spins")
    _check_statevector_size(inst.n)
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
    """Energy expectation of the QAOA state (statevector)."""
    probs = probabilities(qaoa_state(inst, theta, gammas, betas))
    return float(probs @ inst.all_energies())


# ----------------------------------------------------------------------------------------------------
# Sampling and correlators
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


@runtime_checkable
class Correlator(Protocol):
    """Energy expectation and correlations of the warm-started QAOA state, however they are obtained."""

    def energy(self, inst: IsingInstance, theta: Sequence[float], gammas: Sequence[float],
               betas: Sequence[float]) -> float:
        ...

    def correlations(self, inst: IsingInstance, theta: Sequence[float], gammas: Sequence[float],
                     betas: Sequence[float], shots: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        ...


class StatevectorCorrelator:
    """Exact statevector energies; correlations from a counts :class:`Sampler` (statevector by default, or
    hardware). Limited to ``MAX_STATEVECTOR_N`` spins."""

    def __init__(self, sampler: Sampler | None = None):
        self.sampler = sampler if sampler is not None else StatevectorSampler()

    def energy(self, inst: IsingInstance, theta: Sequence[float], gammas: Sequence[float],
               betas: Sequence[float]) -> float:
        return expectation(inst, theta, gammas, betas)

    def energy_function(self, inst: IsingInstance, theta: Sequence[float], p: int) -> Callable[[np.ndarray], float]:
        """Fast closure over precomputed energies and warm state for the angle optimiser."""
        _check_statevector_size(inst.n)
        energies = inst.all_energies()
        psi0 = warm_state(theta)

        def value(x: np.ndarray) -> float:
            psi = psi0
            for g, b in zip(x[:p], x[p:]):
                psi = apply_mixer(apply_cost(psi, energies, g), theta, b)
            return float(probabilities(psi) @ energies)

        return value

    def correlations(self, inst: IsingInstance, theta: Sequence[float], gammas: Sequence[float],
                     betas: Sequence[float], shots: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        return correlations(self.sampler.sample(inst, theta, gammas, betas, shots, rng), inst.n)


class PauliPropCorrelator:
    """Exact (``p = 1``) or perturbatively truncated (``p > 1``) expectations by Pauli propagation, any ``n``.

    Parameters
    ----------
    k : int
        Total perturbative order (see :func:`nanoquant.quantum.pauli_prop.expectations`).
    max_active : int or None
        Non-root sites eligible for perturbative choices per observable.
    pair_top : int or None
        Evaluate ``<Z_i Z_j>`` only among the ``pair_top`` strongest couplings of each spin (``None`` = all pairs).
        The energy then sums those pairs only, which is what the angle optimiser sees.
    """

    def __init__(self, k: int = 1, max_active: int | None = None, pair_top: int | None = None):
        self.k, self.max_active, self.pair_top = k, max_active, pair_top

    def _pairs(self, inst: IsingInstance):
        from .pauli_prop import candidate_pairs

        return None if self.pair_top is None else candidate_pairs(inst, self.pair_top)

    def energy(self, inst: IsingInstance, theta: Sequence[float], gammas: Sequence[float],
               betas: Sequence[float]) -> float:
        from .pauli_prop import expectations

        return expectations(inst, theta, gammas, betas, k=self.k, max_active=self.max_active,
                            pairs=self._pairs(inst))[0]

    def correlations(self, inst: IsingInstance, theta: Sequence[float], gammas: Sequence[float],
                     betas: Sequence[float], shots: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        from .pauli_prop import expectations

        _, z, zz = expectations(inst, theta, gammas, betas, k=self.k, max_active=self.max_active,
                                pairs=self._pairs(inst))
        return z, zz


def as_correlator(obj: Correlator | Sampler) -> Correlator:
    """Accept a correlator, or wrap a counts sampler in a :class:`StatevectorCorrelator`."""
    if isinstance(obj, Correlator):
        return obj
    if hasattr(obj, "sample"):
        return StatevectorCorrelator(obj)
    raise TypeError(f"{type(obj).__name__} is neither a Correlator nor a Sampler")


# ----------------------------------------------------------------------------------------------------
# Angle optimisation
# ----------------------------------------------------------------------------------------------------
def optimize_angles(inst: IsingInstance, theta: Sequence[float], p: int = 1, seed: int = 0, n_starts: int = 8,
                    maxiter: int = 400, correlator: Correlator | None = None,
                    x0: Sequence[float] | None = None) -> tuple[list[float], list[float], float]:
    """Nelder-Mead over ``(gammas, betas)`` from seeded random starts (plus ``x0``) on the correlator's energy.

    The cost is normalised by :func:`nanoquant.quantum.ising.cost_scale` so the ``gamma`` search range is
    problem-independent; ``x0`` is in those normalised units (as returned by :func:`normalized_angles`) and the
    returned ``gammas`` are in the units of ``inst``.

    Returns
    -------
    tuple
        ``(gammas, betas, expectation)`` of the best start, the expectation in the units of ``inst``.
    """
    correlator = correlator if correlator is not None else StatevectorCorrelator()
    scale = cost_scale(inst)
    norm = inst.scaled(1.0 / scale)
    if hasattr(correlator, "energy_function"):
        value = correlator.energy_function(norm, theta, p)
    else:
        def value(x: np.ndarray) -> float:
            return correlator.energy(norm, theta, list(x[:p]), list(x[p:]))

    rng = np.random.default_rng(seed)
    starts = [np.asarray(x0, dtype=np.float64)] if x0 is not None else []
    starts += [np.concatenate([rng.uniform(0.0, 2 * np.pi, size=p), rng.uniform(0.0, np.pi, size=p)])
               for _ in range(max(0, n_starts))]
    if not starts:
        raise ValueError("optimize_angles needs x0 or n_starts > 0")
    best_x, best_v = None, np.inf
    for s0 in starts:
        res = minimize(value, s0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-7})
        if res.fun < best_v:
            best_x, best_v = res.x, float(res.fun)
    gammas = (best_x[:p] / scale).tolist()
    betas = best_x[p:].tolist()
    return gammas, betas, best_v * scale


def normalized_angles(inst: IsingInstance, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
    """``(gammas * cost_scale(inst), betas)`` - the problem-independent form carried between recursion steps."""
    return np.concatenate([np.asarray(gammas, dtype=np.float64) * cost_scale(inst), np.asarray(betas, dtype=np.float64)])


# ----------------------------------------------------------------------------------------------------
# Recursive QAOA
# ----------------------------------------------------------------------------------------------------
class RQAOAStep(BaseModel):
    """One recursion step: the model size, the angles used, and the variable it fixed."""

    n: int
    gammas: list[float]
    betas: list[float]
    expectation: float | None
    elimination: Elimination
    correlation: float
    seconds: float = 0.0
    reoptimized: bool = True


class RQAOAResult(BaseModel):
    """Final assignment (indexed by the original labels in ``labels`` order) and the recursion trace."""

    labels: list[int]
    spins: list[int]
    energy: float
    steps: list[RQAOAStep]
    final_brute_force_n: int
    seconds: float = 0.0

    def to_json(self, path: str | Path) -> None:
        """Write the result as JSON."""
        Path(path).write_text(json.dumps(self.model_dump(), indent=1))

    @classmethod
    def from_json(cls, path: str | Path) -> RQAOAResult:
        """Read a result written by :meth:`to_json`."""
        return cls.model_validate(json.loads(Path(path).read_text()))


def rqaoa(inst: IsingInstance, theta: Sequence[float], sampler: Correlator | Sampler, shots: int, seed: int = 0,
          n_stop: int = 4, p: int = 1, n_starts: int = 8, reoptimize_every: int = 1,
          fixed_angles: Sequence[float] | None = None, log: Callable[[str], None] | None = None,
          maxiter: int = 400) -> RQAOAResult:
    """Warm-started recursive QAOA.

    Parameters
    ----------
    inst : IsingInstance
        Model to minimise.
    theta : sequence of float
        Warm-start angles, one per spin (see :func:`warm_start_angles`).
    sampler : Correlator or Sampler
        Source of energies and correlations; a counts sampler (statevector, hardware) is wrapped in a
        :class:`StatevectorCorrelator`.
    shots : int
        Shots per recursion step (ignored by exact correlators).
    seed : int
        Seeds the angle multistart and the statevector sampler.
    n_stop : int
        Remaining spins that are brute-forced instead of recursed.
    p : int
        QAOA depth per step.
    n_starts : int
        Random Nelder-Mead starts on the first step; later steps start from the carried angles only.
    reoptimize_every : int
        Re-optimise the angles every this many steps (1 = every step); in between the carried angles are reused.
    fixed_angles : sequence of float, optional
        Normalised ``(gammas, betas)`` (see :func:`normalized_angles`) used at every step without optimisation.
    log : callable, optional
        Receives one line per step.
    maxiter : int
        Nelder-Mead iteration cap per start.

    Returns
    -------
    RQAOAResult
        Spins in the order of ``inst.labels``, their energy under ``inst``, and the per-step trace.
    """
    correlator = as_correlator(sampler)
    rng = np.random.default_rng(seed)
    cur, th = inst, list(theta)
    records: list[Elimination] = []
    steps: list[RQAOAStep] = []
    step = 0
    carried = None if fixed_angles is None else np.asarray(fixed_angles, dtype=np.float64)
    t_all = time.time()
    while cur.n > n_stop:
        t0 = time.time()
        reopt = fixed_angles is None and (step % max(1, reoptimize_every) == 0 or carried is None)
        if reopt:
            gammas, betas, val = optimize_angles(cur, th, p=p, seed=seed + step,
                                                 n_starts=n_starts if carried is None else 0, correlator=correlator,
                                                 x0=None if carried is None else carried.tolist(), maxiter=maxiter)
            carried = normalized_angles(cur, gammas, betas)
        else:
            scale = cost_scale(cur)
            gammas, betas, val = (carried[:p] / scale).tolist(), carried[p:].tolist(), None
        z, zz = correlator.correlations(cur, th, gammas, betas, shots, rng)
        i_s = int(np.argmax(np.abs(z)))
        iu, ju = np.triu_indices(cur.n, 1)
        kk = int(np.argmax(np.abs(zz[iu, ju])))
        i_p, j_p, v_p = int(iu[kk]), int(ju[kk]), float(zz[iu[kk], ju[kk]])
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
        seconds = time.time() - t0
        steps.append(RQAOAStep(n=cur.n + 1, gammas=gammas, betas=betas, expectation=val, elimination=rec,
                               correlation=corr, seconds=seconds, reoptimized=reopt))
        if log is not None:
            log(f"[rqaoa] step {step + 1}: n={cur.n + 1} -> {cur.n}, {rec.kind} {rec.i}"
                f"{'' if rec.j is None else f'~{rec.j}'} sign {rec.sign:+d}, |corr| {abs(corr):.3f}, "
                f"{'reopt' if reopt else 'carried'}, {seconds:.1f}s")
        step += 1
    best_s, _ = cur.brute_force()
    full = back_substitute(records, dict(zip(cur.labels, best_s)))
    spins = [int(full[lab]) for lab in inst.labels]
    return RQAOAResult(labels=list(inst.labels), spins=spins, energy=inst.energy(spins), steps=steps,
                       final_brute_force_n=cur.n, seconds=time.time() - t_all)
