"""Heisenberg-picture emulation of warm-started QAOA on Ising models (exact at p = 1, truncated beyond).

State: ``|psi> = U_M(beta_p) U_C(gamma_p) ... U_M(beta_1) U_C(gamma_1) |psi0>`` with the product warm start
``|psi0> = prod_q RY(theta_q)|0>``, the diagonal cost ``H_C = sum h_q Z_q + sum_{q<r} J_qr Z_q Z_r`` and the
warm-start mixer ``U_M = prod_q exp(-i beta (sin theta_q X_q + cos theta_q Z_q))``.

An observable (``Z_i`` or ``Z_i Z_j``) is propagated backwards as a sum of **product operators**
``coef * prod_q (alpha_q I + x_q X + y_q Y + z_q Z)``:

* a mixer layer conjugates each site separately: the ``(x, y, z)`` vector rotates about the warm-start axis by
  ``-2 beta`` (exact, no growth of the sum);
* a cost layer acts on a product operator through its off-diagonal sites. Writing each site as
  ``d_q + o_q`` (``d_q = alpha I + z Z``, ``o_q = x X + y Y``) and picking the off-diagonal set ``A``,
  ``U_C^dag P U_C = P exp(-2i gamma sum_{q in A} Z_q (h_q + sum_{r notin A} J_qr Z_r))``; resolving the ``Z_q`` on
  ``A`` into eigenvalues ``z_q`` gives, per ``(A, z_A)``, one product operator with
  ``o_q -> (x + i z y) X / 2 + (y - i z x) Y / 2`` on ``A`` (times ``exp(-2i gamma z_q h_q)``) and
  ``d_r -> (a C - i b S) I + (b C - i a S) Z``, ``C = cos 2 gamma phi_r``, ``S = sin 2 gamma phi_r``,
  ``phi_r = sum_{q in A} z_q J_qr`` elsewhere.

The observable's own sites (**roots**) are always enumerated exactly (``3^|roots|`` choices: diagonal, ``z = +1``,
``z = -1``). Every other site only carries off-diagonal content of order ``sin(2 gamma J)`` produced by a previous
layer, so choosing it into ``A`` is a perturbative order. ``k`` bounds the total number of such choices over all
layers: ``k = 0`` is exact at ``p = 1``, the error at ``p > 1`` is O((gamma J)^(k + 1)). ``max_active`` restricts
the eligible non-root sites to the ones most strongly coupled to the roots (a coefficient truncation).

Final expectation in the product state: ``<I> = 1``, ``<X> = sin theta``, ``<Y> = 0``, ``<Z> = cos theta``.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations, product

import numpy as np

from .ising import IsingInstance

Group = tuple[int, np.ndarray, np.ndarray]  # (order, coef (P, T), V (P, T, n, 4))


def mixer_rotations(theta: Sequence[float], beta: float) -> np.ndarray:
    """Per-site rotation of the ``(x, y, z)`` Pauli components under ``U_M^dag . U_M``: axis ``(sin theta, 0,
    cos theta)``, angle ``-2 beta`` (checked against the statevector; ``X -> cos 2b X - sin 2b Y`` at ``theta = 0``)."""
    th = np.asarray(theta, dtype=np.float64)
    ax = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], axis=-1)  # (n, 3)
    phi = -2.0 * beta
    K = np.zeros((len(th), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -ax[:, 2], ax[:, 1]
    K[:, 1, 0], K[:, 1, 2] = ax[:, 2], -ax[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -ax[:, 1], ax[:, 0]
    outer = ax[:, :, None] * ax[:, None, :]
    return np.cos(phi) * np.eye(3)[None] + np.sin(phi) * K + (1.0 - np.cos(phi)) * outer


def candidate_pairs(inst: IsingInstance, per_spin: int) -> list[tuple[int, int]]:
    """Pairs ``(i, j)``, ``i < j``, among the ``per_spin`` strongest couplings of each spin."""
    J = np.abs(inst.J_array())
    n = inst.n
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        for j in np.argsort(-J[i])[:per_spin]:
            j = int(j)
            if j != i and J[i, j] > 0:
                pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def _active_sites(J: np.ndarray, roots: np.ndarray, max_active: int | None) -> np.ndarray:
    """``(P, m)`` non-root sites eligible for perturbative off-diagonal choices, strongest root coupling first."""
    n = J.shape[0]
    strength = np.abs(J[roots]).max(axis=1)  # (P, n)
    np.put_along_axis(strength, roots, -1.0, axis=1)  # never pick a root
    order = np.argsort(-strength, axis=1)
    m = n - roots.shape[1] if max_active is None else min(max_active, n - roots.shape[1])
    return order[:, :m]


def _configs(n: int, roots: np.ndarray, active: np.ndarray, sizes: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Off-diagonal choices as a ``(P, K, n)`` matrix of ``z`` values (0 = diagonal) and the ``(K,)`` perturbative
    size of each choice. Roots take all ``3^R`` options; non-root sites from ``active`` take ``0..max(sizes)``."""
    P, R = roots.shape
    root_opts = np.array(list(product((0, 1, -1), repeat=R)), dtype=np.int8)  # (3^R, R)
    C = len(root_opts)
    m = active.shape[1]
    blocks, block_sizes = [], []
    p_idx = np.arange(P)[:, None]
    for s in sizes:
        if s > m:
            continue
        if s == 0:
            combos, signs = np.zeros((1, 0), dtype=np.int64), np.zeros((1, 0), dtype=np.int8)
        else:
            combos = np.array(list(combinations(range(m), s)), dtype=np.int64)  # (Nc, s)
            signs = np.array(list(product((1, -1), repeat=s)), dtype=np.int8)  # (Ns, s)
        Nc, Ns = len(combos), len(signs)
        Zc = np.zeros((P, C, Nc, Ns, n), dtype=np.int8)
        # roots: option c sets z on every root site
        for c, ro in enumerate(root_opts):
            for r in range(R):
                Zc[p_idx, c, :, :, roots[:, r][:, None]] = ro[r]
        # perturbative sites: combination x sign pattern
        for pos in range(s):
            sites = active[:, combos[:, pos]]  # (P, Nc)
            # advanced indices broadcast to (P, Nc, Ns); the sliced root-option axis lands last
            Zc[p_idx[:, :, None], :, np.arange(Nc)[None, :, None], np.arange(Ns)[None, None, :],
               sites[:, :, None]] = signs[:, pos].reshape(1, 1, Ns, 1)
        blocks.append(Zc.reshape(P, -1, n))
        block_sizes.append(np.full(C * Nc * Ns, s, dtype=np.int64))
    return np.concatenate(blocks, axis=1), np.concatenate(block_sizes)


def _site_factors(V: np.ndarray, Zc: np.ndarray, gamma: float, h: np.ndarray, J: np.ndarray):
    """Per-site transformed components after one cost layer, for every term and choice.

    Returns ``(a', b', x', y', phase)``: diagonal components ``a', b'`` and off-diagonal ``x', y'`` of shape
    ``(P, T, K, n)`` (each site has either ``a', b'`` or ``x', y'`` non-zero) and the ``(P, K)`` coefficient phase.
    """
    zf = Zc.astype(np.float64)
    phi = zf @ J  # (P, K, n)
    C, S = np.cos(2.0 * gamma * phi), np.sin(2.0 * gamma * phi)
    off = Zc != 0
    a, x, y, b = (V[..., c][:, :, None, :] for c in range(4))  # (P, T, 1, n)
    Cb, Sb = C[:, None], S[:, None]
    a2 = np.where(off[:, None], 0.0, a * Cb - 1j * b * Sb)
    b2 = np.where(off[:, None], 0.0, b * Cb - 1j * a * Sb)
    z_ = zf[:, None]
    x2 = np.where(off[:, None], 0.5 * (x + 1j * z_ * y), 0.0)
    y2 = np.where(off[:, None], 0.5 * (y - 1j * z_ * x), 0.0)
    phase = np.exp(-2j * gamma * (zf @ h))  # (P, K)
    return a2, b2, x2, y2, phase


BUDGET = 20_000_000  # complex elements per intermediate (P, T, K, n) array, ~320 MB


def _p_slices(P: int, per_observable: int) -> list[slice]:
    """Slices of the observable axis such that each holds at most ``BUDGET`` intermediate elements."""
    step = max(1, min(P, BUDGET // max(1, per_observable)))
    return [slice(s, min(P, s + step)) for s in range(0, P, step)]


def _count_configs(n_roots: int, m: int, kmax: int) -> int:
    """Number of ``(A, z_A)`` choices: ``3^R`` root options times ``sum_{s <= kmax} C(m, s) 2^s``."""
    from math import comb

    return 3 ** n_roots * sum(comb(m, s) * 2 ** s for s in range(min(kmax, m) + 1))


def _expand(group: Group, Zc: np.ndarray, sizes: np.ndarray, gamma: float, h: np.ndarray, J: np.ndarray,
            k: int) -> list[Group]:
    """Materialise one cost layer for a term group; returns new groups keyed by perturbative order."""
    order, coef, V = group
    keep = sizes <= k - order
    Zc, sizes = Zc[:, keep], sizes[keep]
    P, T, n = V.shape[0], V.shape[1], V.shape[2]
    K = Zc.shape[1]
    V2 = np.empty((P, T, K, n, 4), dtype=np.complex128)
    coef2 = np.empty((P, T, K), dtype=np.complex128)
    for sl in _p_slices(P, T * K * n):
        a2, b2, x2, y2, phase = _site_factors(V[sl], Zc[sl], gamma, h, J)
        V2[sl] = np.stack([a2, x2, y2, b2], axis=-1)
        coef2[sl] = coef[sl, :, None] * phase[:, None, :]
    out: list[Group] = []
    for s in np.unique(sizes):
        sel = sizes == s
        out.append((order + int(s), coef2[:, :, sel].reshape(P, -1), V2[:, :, sel].reshape(P, -1, n, 4)))
    return out


def _evaluate(group: Group, Zc: np.ndarray, sizes: np.ndarray, gamma: float, h: np.ndarray, J: np.ndarray, k: int,
              sin_t: np.ndarray, cos_t: np.ndarray) -> np.ndarray:
    """Expectation of a term group after the last cost layer, without materialising the expansion."""
    order, coef, V = group
    keep = sizes <= k - order
    if not keep.any():
        return np.zeros(coef.shape[0], dtype=np.complex128)
    Zc = Zc[:, keep]
    P, T, n = V.shape[0], V.shape[1], V.shape[2]
    out = np.empty(P, dtype=np.complex128)
    for sl in _p_slices(P, T * Zc.shape[1] * n):
        a2, b2, x2, _, phase = _site_factors(V[sl], Zc[sl], gamma, h, J)
        site = a2 + b2 * cos_t + x2 * sin_t  # <I> = 1, <Z> = cos, <X> = sin, <Y> = 0
        vals = np.prod(site, axis=-1)  # (P, T, K)
        out[sl] = np.einsum("pt,pk,ptk->p", coef[sl], phase, vals)
    return out


def _merge(groups: list[Group]) -> list[Group]:
    by_order: dict[int, list[Group]] = {}
    for g in groups:
        by_order.setdefault(g[0], []).append(g)
    return [(o, np.concatenate([g[1] for g in gs], axis=1), np.concatenate([g[2] for g in gs], axis=1))
            for o, gs in sorted(by_order.items())]


def _propagate(roots: np.ndarray, inst: IsingInstance, theta: Sequence[float], gammas: Sequence[float],
               betas: Sequence[float], k: int, max_active: int | None) -> np.ndarray:
    """``<prod_{r in roots} Z_r>`` for a batch of observables with the same number of roots. Returns ``(P,)`` real."""
    P, R = roots.shape
    n = inst.n
    h, J = inst.h_array(), inst.J_array()
    th = np.asarray(theta, dtype=np.float64)
    sin_t, cos_t = np.sin(th), np.cos(th)
    V = np.zeros((P, 1, n, 4), dtype=np.complex128)
    V[..., 0] = 1.0
    for r in range(R):
        V[np.arange(P), 0, roots[:, r], 0] = 0.0
        V[np.arange(P), 0, roots[:, r], 3] = 1.0
    groups: list[Group] = [(0, np.ones((P, 1), dtype=np.complex128), V)]
    active = _active_sites(J, roots, max_active)
    p = len(gammas)
    total = np.zeros(P, dtype=np.complex128)
    configs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for layer in range(p - 1, -1, -1):
        Rm = mixer_rotations(th, betas[layer])
        for gi, (o, c, Vg) in enumerate(groups):
            Vg = Vg.copy()
            Vg[..., 1:] = np.einsum("nab,ptnb->ptna", Rm, Vg[..., 1:])
            groups[gi] = (o, c, Vg)
        # at the outermost layer only the roots are non-identity, so perturbative choices are exactly zero
        max_size = min(k, active.shape[1]) if layer < p - 1 else 0
        if max_size not in configs:
            configs[max_size] = _configs(n, roots, active, list(range(max_size + 1)))
        Zc, size_of = configs[max_size]
        if layer > 0:
            new: list[Group] = []
            for g in groups:
                new.extend(_expand(g, Zc, size_of, gammas[layer], h, J, k))
            groups = _merge(new)
        else:
            for g in groups:
                total += _evaluate(g, Zc, size_of, gammas[0], h, J, k, sin_t, cos_t)
    return total.real


def expectations(inst: IsingInstance, theta: Sequence[float], gammas: Sequence[float], betas: Sequence[float],
                 k: int = 1, max_active: int | None = None, pairs: Sequence[tuple[int, int]] | None = None,
                 chunk: int = 256) -> tuple[float, np.ndarray, np.ndarray]:
    """``(energy, <Z_i>, <Z_i Z_j>)`` of the warm-started QAOA state by truncated Pauli propagation.

    Parameters
    ----------
    inst : IsingInstance
        Model.
    theta : sequence of float
        Warm-start angles.
    gammas, betas : sequence of float
        QAOA angles, ``p = len(gammas)``.
    k : int
        Total perturbative order kept (non-root off-diagonal choices summed over layers). Exact at ``p = 1`` for any
        ``k``; ``k >= n - 2`` is exact at any depth.
    max_active : int or None
        Non-root sites eligible per observable (strongest coupling to the roots first); ``None`` = all.
    pairs : sequence of (int, int) or None
        Pairs to evaluate (``None`` = all ``i < j``). The energy sums only these pairs.
    chunk : int
        Upper bound on observables per vectorised batch; lowered automatically when the materialised term groups of
        deep circuits would exceed ~1 GB per batch.

    Returns
    -------
    tuple
        ``energy`` (float), ``z`` ``(n,)``, ``zz`` ``(n, n)`` symmetric with zeros on the diagonal and on pairs not
        evaluated.
    """
    if len(gammas) != len(betas) or not gammas:
        raise ValueError("gammas and betas must have the same non-zero length")
    n = inst.n
    if len(theta) != n:
        raise ValueError(f"theta has {len(theta)} entries for {n} spins")
    p = len(gammas)

    def batch(n_roots: int) -> int:
        m = n - n_roots if max_active is None else min(max_active, n - n_roots)
        # terms after the exact outermost layer (3^R) and each truncated middle layer (upper bound: no pruning)
        terms = 3 ** n_roots * _count_configs(n_roots, m, k) ** max(0, p - 2)
        per_observable_bytes = terms * n * 4 * 16
        return int(max(1, min(chunk, 1_000_000_000 // max(1, per_observable_bytes))))

    singles = np.arange(n)[:, None]
    c1 = batch(1)
    z = np.concatenate([_propagate(singles[s:s + c1], inst, theta, gammas, betas, k, max_active)
                        for s in range(0, n, c1)]) if n else np.zeros(0)
    pair_list = [(i, j) for i in range(n) for j in range(i + 1, n)] if pairs is None else [tuple(p) for p in pairs]
    zz = np.zeros((n, n))
    if pair_list:
        roots = np.asarray(pair_list, dtype=np.int64)
        c2 = batch(2)
        vals = np.concatenate([_propagate(roots[s:s + c2], inst, theta, gammas, betas, k, max_active)
                               for s in range(0, len(roots), c2)])
        zz[roots[:, 0], roots[:, 1]] = vals
        zz[roots[:, 1], roots[:, 0]] = vals
    J = inst.J_array()
    energy = inst.const + float(inst.h_array() @ z) + 0.5 * float(np.sum(J * zz))
    return energy, z, zz
