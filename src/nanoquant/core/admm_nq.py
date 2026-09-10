# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .curvature import temper_eigenvalues

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Local registry for rho schedulers
RHO_SCHEDULER_REGISTRY = {}


@torch.no_grad()
def power_iteration(A, num_iters=5):
    """
    Power iteration for top singular triplet (u, sigma, v) of A.
    """
    n = A.shape[1]
    v = torch.randn(n, device=A.device, dtype=A.dtype)
    v = v / torch.norm(v)

    At = A.mT  # view; reuse
    for _ in range(num_iters):
        u = torch.mv(A, v)
        u = u / u.norm()

        v = torch.mv(At, u)
        v = v / v.norm()

    u_unnorm = torch.mv(A, v)
    sigma = torch.norm(u_unnorm)
    u = u_unnorm / sigma
    return u, sigma, v


@torch.no_grad()
def svid(W, inner_iters=5, eps=1e-12):
    """
    Sign-Value-Independent Decomposition (SVID).
    Returns u, v, Sg where Sg is sign matrix of W.
    """
    Sg = W.sign()
    Sg[Sg == 0] = 1
    u, s, v = power_iteration(W.abs(), inner_iters)
    u = u * s
    return u, v, Sg


@torch.no_grad()
def rank1_approx(W, inner_iters=5, eps=1e-12):
    """
    Rank-1 approximation using SVID results.
    """
    u, v, Sg = svid(W, inner_iters, eps)
    apx = torch.outer(u, v)
    return apx * Sg


@torch.no_grad()
def _svid_nonneg(W: torch.Tensor, inner_iters: int, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVID with the rank-1 scale vectors forced to be non-negative (their product is unchanged)."""
    u, v, Sg = svid(W, inner_iters, eps)
    if u.sum() < 0:
        u, v = -u, -v
    return u.abs(), v.abs(), Sg


@torch.no_grad()
def _admm_solve_step(X, Y, Z, U, rho, reg, eps=1e-12):
    """
    Solves one step of ADMM robustly using a stabilized Cholesky decomposition.
    Solve: (X^T X + stabilizer*I) * Factor = X^T Y + rho*(Z-U)
    """
    orig_dtype = X.dtype
    X, Y, Z, U = (t.to(torch.float32) for t in (X, Y, Z, U))

    Xt = X.mT  # view, no materialization
    system_matrix = Xt @ X  # (k,k)
    system_matrix = 0.5 * (system_matrix + system_matrix.mT)

    # stabilizer on diagonal
    diag_mean = system_matrix.diagonal().mean().abs()
    stabilizer = torch.clamp(rho * diag_mean + reg, min=eps)
    system_matrix.diagonal().add_(stabilizer)

    rhs = (Xt @ Y) + rho * (Z - U)

    # Fast path: cholesky_ex gives info instead of exception
    L, info = torch.linalg.cholesky_ex(system_matrix, upper=False)

    if info.item() == 0:
        Factor = torch.cholesky_solve(rhs, L, upper=False)
    else:
        # Rare fallback
        Factor = torch.linalg.solve(system_matrix, rhs)

    return Factor.to(orig_dtype)


@torch.no_grad()
def _sylvester_stabilizer(lam: torch.Tensor, M: torch.Tensor, rho: float, reg: float, eps: float = 1e-12) -> torch.Tensor:
    """Diagonal stabiliser of the generalised Sylvester step.

    ``sigma = rho + reg * mean(lam) * mean(diag(M))``: the ADMM penalty ``rho`` enters exactly as on the
    right-hand side (``rho * (Z - U)``), and the ridge ``reg`` is scaled to the typical eigenvalue of the
    data-term operator ``Sigma kron M``. With ``Sigma = I`` and a unit-diagonal ``M`` (the situation of
    :func:`_admm_solve_step`, whose design matrices have unit-norm columns) this equals ``rho + reg``.

    Parameters
    ----------
    lam : torch.Tensor
        Eigenvalues of the (unit-diagonal) curvature factor ``Sigma``.
    M : torch.Tensor
        Symmetric ``k x k`` Gram matrix of the other factor.
    rho, reg : float
        ADMM penalty and ridge regularisation.
    eps : float
        Lower clamp.
    """
    diag_mean = M.diagonal().mean().abs()
    return torch.clamp(rho + reg * lam.mean() * diag_mean, min=eps)


@dataclass
class SylvesterState:
    """Stale eigenbasis of the ``k x k`` Gram matrix of one ADMM X-update, plus counters of the solver rungs.

    Attributes
    ----------
    mu, Q_M : torch.Tensor or None
        Approximate eigenvalues ``(k,)`` and orthonormal basis ``(k, k)`` of the Gram matrix from the last refresh.
    n_eigh, n_qr, n_pcg, n_calls : int
        Full eigendecompositions, orthogonal-iteration (QR) steps, PCG steps and inexact calls so far.
    """
    mu: torch.Tensor | None = None
    Q_M: torch.Tensor | None = None
    n_eigh: int = 0
    n_qr: int = 0
    n_pcg: int = 0
    n_calls: int = 0

    def summary(self) -> str:
        """One-line rendering of the rung counters."""
        return f"eigh {self.n_eigh} | qr {self.n_qr} | pcg {self.n_pcg} | inexact calls {self.n_calls}"


@torch.no_grad()
def _precond_solve(Q: torch.Tensor, lam: torch.Tensor, Q_M: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
                   X: torch.Tensor) -> torch.Tensor:
    """``Q [ (Q^T X Q_M) / (lam mu^T + sigma) ] Q_M^T``.

    The exact solution of ``Sigma F M + sigma F = X`` when ``(mu, Q_M)`` is the eigendecomposition of ``M``; with a
    stale basis and Rayleigh quotients it is the preconditioner of the inexact solve.
    """
    X_hat = Q.mT @ X @ Q_M
    return Q @ (X_hat / (lam.unsqueeze(1) * mu.unsqueeze(0) + sigma)) @ Q_M.mT


@torch.no_grad()
def _rayleigh_refresh(M: torch.Tensor, Q_M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``Z = M Q_M`` and the Rayleigh quotients ``diag(Q_M^T M Q_M)`` (clamped at zero) of the basis ``Q_M``."""
    Z = M @ Q_M
    return Z, (Z * Q_M).sum(dim=0).clamp(min=0.0)


@torch.no_grad()
def _orthogonal_iteration_step(M: torch.Tensor, Z: torch.Tensor,
                               mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One step of orthogonal (subspace) iteration on a full basis: ``Q_M <- qr(M Q_M)``.

    The columns are ordered by descending Rayleigh quotient first, so the QR sharpens the dominant directions
    (unshifted QR iteration converges dominant-first; from an ascending order the first step would scramble them).
    Mixing between two eigen-directions shrinks by the ratio of their eigenvalues, so one step repairs the
    well-separated spikes and leaves the (harmless) mixing inside clusters alone.

    Parameters
    ----------
    M : torch.Tensor
        Symmetric PSD ``(k, k)`` matrix.
    Z : torch.Tensor
        ``M Q_M`` for the current basis (from :func:`_rayleigh_refresh`).
    mu : torch.Tensor
        Rayleigh quotients of the current basis.

    Returns
    -------
    tuple of torch.Tensor
        New basis ``Q_M``, its ``Z = M Q_M`` and its Rayleigh quotients.
    """
    order = torch.argsort(mu, descending=True)
    Q_M, _ = torch.linalg.qr(Z[:, order])
    Z, mu = _rayleigh_refresh(M, Q_M)
    return Q_M, Z, mu


@torch.no_grad()
def _sylvester_solve_step(Q: torch.Tensor, lam: torch.Tensor, M: torch.Tensor, C: torch.Tensor, rho: float, reg: float,
                          eps: float = 1e-12, eigh_dtype: torch.dtype = torch.float64,
                          Sigma: torch.Tensor | None = None, state: SylvesterState | None = None, tol: float = 0.0,
                          qr_steps: int = 1, max_pcg: int = 3) -> torch.Tensor:
    """Solve ``Sigma F M + sigma F = C`` for ``F`` with ``Sigma = Q diag(lam) Q^T``.

    This is the X-update of the Mahalanobis ADMM: ``Sigma`` is the (fixed, eigendecomposed once per
    layer) unit-diagonal curvature factor on the long side of ``F``, ``M`` the small ``k x k`` Gram matrix
    of the other factor and ``C`` the right-hand side (curvature-weighted data term plus ``rho`` times
    the ADMM target). Rotating into both eigenbases makes the operator diagonal:
    ``F = Q [ (Q^T C Q_M) / (lam lam_M^T + sigma) ] Q_M^T``.

    With ``tol > 0`` and a ``state`` the ``k x k`` eigendecomposition is not recomputed every call. The stale basis
    (with fresh Rayleigh quotients) is used as a preconditioner and the current operator supplies the correction, in
    rungs of increasing cost until the relative residual ``||C - (Sigma F M + sigma F)|| <= tol ||C||``:
    stale basis + Rayleigh refresh; ``qr_steps`` orthogonal-iteration steps; ``max_pcg`` preconditioned
    conjugate-gradient steps; finally a full eigendecomposition (the exact solve). All matmuls run in fp32/TF32,
    which floors the achievable relative residual near 1e-3.

    Parameters
    ----------
    Q, lam : torch.Tensor
        Eigenvectors ``(n, n)`` and eigenvalues ``(n,)`` of ``Sigma``.
    M : torch.Tensor
        Symmetric PSD ``(k, k)`` matrix.
    C : torch.Tensor
        Right-hand side ``(n, k)``.
    rho, reg : float
        ADMM penalty and ridge regularisation (see :func:`_sylvester_stabilizer`).
    eps : float
        Lower clamp for stabiliser / eigenvalues.
    eigh_dtype : torch.dtype
        Precision of the ``k x k`` eigendecomposition (default fp64).
    Sigma : torch.Tensor, optional
        The curvature factor itself ``(n, n)``; required for the inexact rungs (residuals and PCG).
    state : SylvesterState, optional
        Stale basis and counters, updated in place. ``None`` gives the exact legacy solve.
    tol : float
        Relative residual accepted by the inexact rungs; ``<= 0`` gives the exact legacy solve.
    qr_steps, max_pcg : int
        Rung sizes (see above).

    Returns
    -------
    torch.Tensor
        ``F`` of shape ``(n, k)`` in the dtype of ``C``.
    """
    orig_dtype = C.dtype
    Q, lam, M, C = (t.to(torch.float32) for t in (Q, lam, M, C))
    M = 0.5 * (M + M.mT)
    sigma = _sylvester_stabilizer(lam, M, rho, reg, eps)

    def exact() -> torch.Tensor:
        mu, Q_M = torch.linalg.eigh(M.to(eigh_dtype))
        mu = mu.to(torch.float32).clamp(min=0.0)
        Q_M = Q_M.to(torch.float32)
        if state is not None:
            state.mu, state.Q_M = mu, Q_M
            state.n_eigh += 1
        return _precond_solve(Q, lam, Q_M, mu, sigma, C)

    if state is None or tol <= 0 or state.Q_M is None or Sigma is None:
        return exact().to(orig_dtype)

    state.n_calls += 1
    Sigma = Sigma.to(torch.float32)
    tiny = torch.finfo(torch.float32).tiny
    threshold = tol * C.norm()

    def operator(X: torch.Tensor) -> torch.Tensor:
        return Sigma @ X @ M + sigma * X

    def accept(F: torch.Tensor, Q_M: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        state.mu, state.Q_M = mu, Q_M
        return F.to(orig_dtype)

    # rung 1: stale basis with fresh Rayleigh quotients
    Q_M = state.Q_M
    Z, mu = _rayleigh_refresh(M, Q_M)
    F = _precond_solve(Q, lam, Q_M, mu, sigma, C)
    r = C - operator(F)
    if r.norm() <= threshold:
        return accept(F, Q_M, mu)
    # rung 2: orthogonal-iteration steps on the basis
    for _ in range(max(0, int(qr_steps))):
        Q_M, Z, mu = _orthogonal_iteration_step(M, Z, mu)
        state.n_qr += 1
        F = _precond_solve(Q, lam, Q_M, mu, sigma, C)
        r = C - operator(F)
        if r.norm() <= threshold:
            return accept(F, Q_M, mu)
    # rung 3: preconditioned conjugate gradient (operator and preconditioner are SPD in the Frobenius inner product)
    if max_pcg > 0:
        z = _precond_solve(Q, lam, Q_M, mu, sigma, r)
        p = z
        rz = (r * z).sum()
        for _ in range(int(max_pcg)):
            state.n_pcg += 1
            Ap = operator(p)
            alpha = rz / (p * Ap).sum().clamp_min(tiny)
            F = F + alpha * p
            r = r - alpha * Ap
            if r.norm() <= threshold:
                return accept(F, Q_M, mu)
            z = _precond_solve(Q, lam, Q_M, mu, sigma, r)
            rz_new = (r * z).sum()
            p = z + (rz_new / rz.clamp_min(tiny)) * p
            rz = rz_new
    # rung 4: the basis drifted too far, refresh it
    return exact().to(orig_dtype)


class EigCache:
    """Normalised eigendecompositions of curvature factors that several layers share (e.g. the fresh input factor of
    q/k/v or gate/up).

    Only tensors registered with :meth:`register` are cached, so per-layer factors never pile up in memory. Entries
    are keyed by the tensor's storage pointer, shape and the conditioning knobs; the registered tensor is held alive
    so its pointer cannot be recycled while the entry exists.
    """

    def __init__(self) -> None:
        self._shared: dict[int, torch.Tensor] = {}
        self._entries: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def register(self, cov: torch.Tensor) -> None:
        """Mark ``cov`` as shared: its eigendecompositions will be cached."""
        self._shared[cov.data_ptr()] = cov

    def clear(self) -> None:
        """Drop all entries and registrations."""
        self._shared.clear()
        self._entries.clear()

    @staticmethod
    def _key(cov: torch.Tensor, power: float, cond_max: float, spike_rank: int, eigh_dtype: torch.dtype) -> tuple:
        return (cov.data_ptr(), tuple(cov.shape), str(cov.device), float(power), float(cond_max), int(spike_rank),
                str(eigh_dtype))

    def get(self, cov: torch.Tensor, power: float, cond_max: float, spike_rank: int,
            eigh_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Cached ``(Sigma, lam, Q)`` of a registered ``cov`` under these knobs, else ``None``."""
        if cov.data_ptr() not in self._shared:
            return None
        return self._entries.get(self._key(cov, power, cond_max, spike_rank, eigh_dtype))

    def put(self, cov: torch.Tensor, power: float, cond_max: float, spike_rank: int, eigh_dtype: torch.dtype,
            value: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        """Store ``value`` for a registered ``cov``; a no-op for unregistered tensors."""
        if cov.data_ptr() in self._shared:
            self._entries[self._key(cov, power, cond_max, spike_rank, eigh_dtype)] = value


@torch.no_grad()
def _normalized_curvature(cov: torch.Tensor, norm_vec: torch.Tensor, eigh_dtype: torch.dtype, eps: float,
                          power: float = 1.0, cond_max: float = 0.0, spike_rank: int = 0,
                          eig_cache: EigCache | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unit-diagonal curvature factor ``D^-1/2 cov D^-1/2`` (with ``D = norm_vec^2``) and its eigendecomposition.

    With ``power != 1``, ``cond_max > 0`` or ``spike_rank > 0`` the spectrum is projected/tempered
    (:func:`temper_eigenvalues`, trace preserved) and ``Sigma`` is rebuilt from the new eigenvalues, so the data
    term trusts the dominant directions less (tempering) or denoises the bulk (spike-plus-flat projection).

    Parameters
    ----------
    eig_cache : EigCache, optional
        Cache consulted and filled for factors registered as shared (``norm_vec`` must be the diagonal of ``cov``
        for the cached result to be the right one, which holds for the fresh input factor).

    Returns
    -------
    tuple
        ``(Sigma, lam, Q)`` in fp32 with eigenvalues clamped at ``eps``.
    """
    if eig_cache is not None:
        hit = eig_cache.get(cov, power, cond_max, spike_rank, eigh_dtype)
        if hit is not None:
            return hit
    n = norm_vec.reshape(-1).to(torch.float32)
    Sigma = cov.to(torch.float32) / (n.unsqueeze(1) * n.unsqueeze(0))
    Sigma = 0.5 * (Sigma + Sigma.mT)
    lam, Q = torch.linalg.eigh(Sigma.to(eigh_dtype))
    lam = lam.to(torch.float32).clamp(min=eps)
    Q = Q.to(torch.float32)
    if power != 1.0 or cond_max > 0 or spike_rank > 0:
        lam = temper_eigenvalues(lam, power=power, cond_max=cond_max, spike_rank=spike_rank).clamp(min=eps)
        Sigma = (Q * lam) @ Q.mT
        Sigma = 0.5 * (Sigma + Sigma.mT)
    if eig_cache is not None:
        eig_cache.put(cov, power, cond_max, spike_rank, eigh_dtype, (Sigma, lam, Q))
    return Sigma, lam, Q


@torch.no_grad()
def factorize_admm_nanoquant(
    W,
    i_norm,
    o_norm,
    mid_rank,
    outer_iters=400,
    inner_iters=5,
    reg=3e-2,
    is_transpose=False,
    eps=1e-12,
    rho_scheduler='cubic',
    print_admm_steps=False,
    i_cov: torch.Tensor | None = None,
    o_cov: torch.Tensor | None = None,
    eigh_dtype: torch.dtype = torch.float64,
    mid_scale: bool = False,
    curvature_power: float = 1.0,
    curvature_cond_max: float = 0.0,
    curvature_spike_rank: int = 0,
    eig_cache: EigCache | None = None,
    early_stop_patience: int = 0,
    early_stop_tol: float = 1e-4,
    early_stop_min_frac: float = 0.5,
    sylvester_tol: float = 0.0,
    sylvester_qr_steps: int = 1,
    sylvester_max_pcg: int = 3,
):
    """
    Decomposes the weight matrix W into two binary matrices A and B using ADMM.
    Assumes W has the shape (out_features, in_features).

    Post-processing extracts scales and binary-compatible matrices either by mean-magnitude
    extraction (Scale-Binary-Binary-Scale, ``mid_scale=False``) or, with ``mid_scale=True``, by the exact
    SVID triple of each factor (Scale-Binary-Scale-Binary-Scale, i.e. an explicit per-rank middle scale so
    the deployed form equals what ADMM converged to).

    Args:
        W: Weight matrix to decompose
        i_norm: Input norm (diagonal curvature, in_features)
        o_norm: Output norm (diagonal curvature, out_features)
        mid_rank: Middle rank for factorization
        outer_iters: Number of outer iterations
        inner_iters: Number of inner iterations
        reg: Regularization parameter
        is_transpose: Whether to transpose the weight matrix
        eps: Small epsilon value to prevent division by zero and numerical instability
        rho_scheduler: Rho scheduler name. Available:
                       ['cubic', 'linear', 'logistic', 'exp_decay', 'exp_growth']
        print_admm_steps: Whether to print intermediate ADMM steps
        i_cov: Optional dense input-side curvature factor (in, in) whose diagonal is ``i_norm``.
        o_cov: Optional dense output-side curvature factor (out, out) whose diagonal is ``o_norm``.
               When both are given, the data term becomes the Mahalanobis distance
               tr(L (W_n - AB) R (W_n - AB)^T) with the unit-diagonal normalised factors L, R, while the
               rho penalty and the SVID projection stay Euclidean.
        eigh_dtype: Precision of the eigendecompositions used by the Mahalanobis solver.
        mid_scale: Export an explicit per-rank ``scale_mid`` (see above).
        curvature_power, curvature_cond_max, curvature_spike_rank: Spectral tempering / spike-plus-flat projection
               of the unit-diagonal factors L, R (``core.curvature.temper_eigenvalues``); defaults leave them untouched.
        eig_cache: Cache of normalised eigendecompositions for curvature factors shared by several layers
               (see :class:`EigCache`); factors not registered there are never cached.
        early_stop_patience, early_stop_tol, early_stop_min_frac: Stop once the projected binary factors ``A_z``,
               ``B_z`` have shown no sign flip and a relative change below ``early_stop_tol`` for
               ``early_stop_patience`` consecutive iterations, after ``early_stop_min_frac`` of the schedule
               (``patience = 0`` runs all ``outer_iters``; the exported latents depend on the final ``rho``).
        sylvester_tol, sylvester_qr_steps, sylvester_max_pcg: Inexact Mahalanobis X-update
               (see :func:`_sylvester_solve_step`); ``sylvester_tol = 0`` keeps the exact per-iteration
               eigendecomposition.
    """
    if is_transpose:
        results = factorize_admm_nanoquant(W.mT, o_norm, i_norm, mid_rank, outer_iters, inner_iters, reg, False, eps,
                                           rho_scheduler, print_admm_steps, i_cov=o_cov, o_cov=i_cov,
                                           eigh_dtype=eigh_dtype, mid_scale=mid_scale,
                                           curvature_power=curvature_power, curvature_cond_max=curvature_cond_max,
                                           curvature_spike_rank=curvature_spike_rank, eig_cache=eig_cache,
                                           early_stop_patience=early_stop_patience, early_stop_tol=early_stop_tol,
                                           early_stop_min_frac=early_stop_min_frac, sylvester_tol=sylvester_tol,
                                           sylvester_qr_steps=sylvester_qr_steps, sylvester_max_pcg=sylvester_max_pcg)
        swapped = {
            "W_final": results["W_final"].mT,
            "A": results["B"],
            "B": results["A"],
            "A_latent": results["B_latent"],
            "B_latent": results["A_latent"],
            "scale_pre": results["scale_post"],
            "scale_post": results["scale_pre"],
        }
        if "scale_mid" in results:
            swapped["scale_mid"] = results["scale_mid"]
        return swapped

    device = W.device
    out_features, in_features = W.shape

    norm_i = i_norm.sqrt().clamp(eps)
    norm_o = o_norm.sqrt().clamp(eps).unsqueeze(1)
    W_norm = W * norm_i.unsqueeze(0) * norm_o

    # Optional dense curvature -> Mahalanobis data term
    use_maha = i_cov is not None and o_cov is not None
    if use_maha:
        Lt, lam_L, Q_L = _normalized_curvature(o_cov.to(device), norm_o, eigh_dtype, eps, curvature_power,
                                               curvature_cond_max, curvature_spike_rank, eig_cache)  # (out, out)
        Rt, lam_R, Q_R = _normalized_curvature(i_cov.to(device), norm_i, eigh_dtype, eps, curvature_power,
                                               curvature_cond_max, curvature_spike_rank, eig_cache)  # (in, in)
        state_A, state_B = SylvesterState(), SylvesterState()  # stale Gram eigenbases of the two X-updates
        W_norm32 = W_norm.to(torch.float32)
        P = Lt @ W_norm32 @ Rt  # curvature-weighted target, (out, in)
        if print_admm_steps:
            for name, S, lam, Q in (("L", Lt, lam_L, Q_L), ("R", Rt, lam_R, Q_R)):
                rec = (Q * lam) @ Q.mT
                err = ((rec - S).norm() / S.norm().clamp(eps)).item()
                print(f"\t\t[eigh check] {name}: relative reconstruction error {err:.3e} "
                      f"(min/max eig {lam.min().item():.3e}/{lam.max().item():.3e})")
            maha_ref = torch.trace(Lt @ W_norm32 @ Rt @ W_norm32.mT).clamp(eps)

    # we remove SVD-based init, since random init is (1) faster (2) shows on-par or better performance
    A_ls = torch.randn((out_features, mid_rank), device=device, dtype=W.dtype)
    B_ls = torch.randn((mid_rank, in_features), device=device, dtype=W.dtype)

    A_z, B_z = A_ls, B_ls
    if outer_iters > 0:
        A_z = rank1_approx(A_ls, inner_iters, eps)
        B_z = rank1_approx(B_ls, inner_iters, eps)

    if print_admm_steps:
        A_z_old = A_z.clone()
        B_z_old = B_z.clone()

    A_u = A_ls - A_z
    B_u = B_ls - B_z

    rho_scheduler_func = RHO_SCHEDULER_REGISTRY[rho_scheduler]

    # early stopping: the exported factors are A_z / B_z, so once their signs and rank-one magnitudes stop moving
    # the remaining iterations only move the latent
    A_z_prev = B_z_prev = None
    frozen_streak = 0
    min_iters = int(early_stop_min_frac * outer_iters)

    for itt in range(outer_iters):
        rho = rho_scheduler_func(itt / outer_iters)

        # 1) X-update
        mid_norm_b = B_z.norm(dim=1).clamp(eps)
        B_bar = B_z / mid_norm_b.unsqueeze(1)  # (mid, in), unit-norm rows
        if use_maha:
            B_bar32 = B_bar.to(torch.float32)
            M = B_bar32 @ Rt @ B_bar32.mT  # (mid, mid)
            C = P @ B_bar32.mT + rho * (A_z - A_u).to(torch.float32)  # (out, mid)
            A_ls = _sylvester_solve_step(Q_L, lam_L, M, C, rho, reg, eps, eigh_dtype, Sigma=Lt, state=state_A,
                                         tol=sylvester_tol, qr_steps=sylvester_qr_steps,
                                         max_pcg=sylvester_max_pcg).to(W.dtype)
        else:
            # W_norm.T uses view; keep it
            A_ls = _admm_solve_step(B_bar.mT, W_norm.mT, A_z.mT, A_u.mT, rho, reg, eps).mT

        mid_norm_a = A_z.norm(dim=0).clamp(eps)
        A_bar = A_z / mid_norm_a  # (out, mid), unit-norm columns
        if use_maha:
            A_bar32 = A_bar.to(torch.float32)
            N = A_bar32.mT @ Lt @ A_bar32  # (mid, mid)
            C = (A_bar32.mT @ P + rho * (B_z - B_u).to(torch.float32)).mT  # (in, mid)
            B_ls = _sylvester_solve_step(Q_R, lam_R, N, C, rho, reg, eps, eigh_dtype, Sigma=Rt, state=state_B,
                                         tol=sylvester_tol, qr_steps=sylvester_qr_steps,
                                         max_pcg=sylvester_max_pcg).mT.to(W.dtype)
        else:
            B_ls = _admm_solve_step(A_bar, W_norm, B_z, B_u, rho, reg, eps)

        # 2) Z-update
        target_A = A_ls + A_u
        target_B = B_ls + B_u
        A_z = rank1_approx(target_A, inner_iters, eps)
        B_z = rank1_approx(target_B, inner_iters, eps)

        # 3) U-update
        A_u.add_(A_ls - A_z)
        B_u.add_(B_ls - B_z)

        if early_stop_patience > 0:
            if A_z_prev is not None:
                flips = int((torch.sign(A_z) != torch.sign(A_z_prev)).sum()
                            + (torch.sign(B_z) != torch.sign(B_z_prev)).sum())
                change = max((A_z - A_z_prev).norm() / A_z_prev.norm().clamp(eps),
                             (B_z - B_z_prev).norm() / B_z_prev.norm().clamp(eps)).item()
                if itt + 1 >= min_iters and flips == 0 and change < early_stop_tol:
                    frozen_streak += 1
                else:
                    frozen_streak = 0
            A_z_prev, B_z_prev = A_z, B_z  # rank1_approx returns fresh tensors, so references suffice
            if frozen_streak >= early_stop_patience:
                print(f"\t\t[ADMM] early stop at {itt + 1}/{outer_iters} (Z frozen for {frozen_streak} iterations)")
                break

        if print_admm_steps:
            if (itt == 0 or (itt + 1) % 100 == 0 or itt == outer_iters - 1):
                r_A = torch.norm(A_ls - A_z).item()
                r_B = torch.norm(B_ls - B_z).item()
                primal_res = r_A + r_B

                s_A = torch.norm(rho * (A_z - A_z_old)).item()
                s_B = torch.norm(rho * (B_z - B_z_old)).item()
                dual_res = s_A + s_B

                mid = B_z.norm(dim=1).clamp(eps)
                # (A_z / mid) @ B_z  ->  F.linear(A_z / mid, B_z.T)
                pred = F.linear(A_z / mid, B_z.mT)
                curr_loss = (W_norm - pred).norm().item()
                normalized_err = (curr_loss**2) / (W_norm.norm()**2).clamp(eps)

                msg = (f"\t\t[ADMM Step {itt+1:04d}/{outer_iters:04d}] Loss: {normalized_err:.5e} | "
                       f"Primal(r): {primal_res:.5e} | Dual(s): {dual_res:.5e} | Rho: {rho:.4f}")
                if use_maha:
                    E = (W_norm - pred).to(torch.float32)
                    maha = (torch.trace(Lt @ E @ Rt @ E.mT) / maha_ref).item()
                    # the same loss for the un-projected X-variables (A_bar @ B_ls)
                    E_x = (W_norm - F.linear(A_bar, B_ls.mT)).to(torch.float32)
                    maha_x = (torch.trace(Lt @ E_x @ Rt @ E_x.mT) / maha_ref).item()
                    msg += f" | Mahalanobis(Z): {maha:.5e} | Mahalanobis(X): {maha_x:.5e}"
                print(msg)

            A_z_old.copy_(A_z)
            B_z_old.copy_(B_z)

    if use_maha and sylvester_tol > 0:
        print(f"\t\t[ADMM sylvester] A-update: {state_A.summary()} | B-update: {state_B.summary()}")

    # Final export
    A_latent = (A_ls + A_u) / norm_o
    B_latent = (B_ls + B_u) / norm_i

    A_unbalanced = A_z / norm_o
    B_unbalanced = B_z / norm_i

    A_latent_unb = (A_ls + A_u) / norm_o
    B_latent_unb = (B_ls + B_u) / norm_i

    norm_A = A_unbalanced.norm().clamp(eps)
    norm_B = B_unbalanced.norm().clamp(eps)
    balance_factor = (norm_B / norm_A).sqrt()

    A_final = A_unbalanced * balance_factor
    B_final = B_unbalanced / balance_factor
    A_latent = A_latent_unb * balance_factor
    B_latent = B_latent_unb / balance_factor

    # per-rank normaliser compensating the column normalisation of A used in the B-update
    scale_factor = 1.0
    if outer_iters > 0:
        scale_factor = 1.0 / A_z.norm(dim=0).clamp(eps)

    if mid_scale:
        # Exact Scale-Binary-Scale-Binary-Scale export: both factors have rank-1 magnitude by construction
        # (SVID projection), so their SVID triples recover them exactly.
        mid = scale_factor if torch.is_tensor(scale_factor) else torch.ones(mid_rank, device=device)
        u_A, v_A, S_A = _svid_nonneg(A_final.to(torch.float32), inner_iters, eps)  # (out,), (mid,), (out, mid)
        u_B, v_B, S_B = _svid_nonneg(B_final.to(torch.float32), inner_iters, eps)  # (mid,), (in,), (mid, in)

        scale_post = u_A.view(1, -1)
        scale_mid = (v_A * mid.to(torch.float32) * u_B).view(1, -1)
        scale_pre = v_B.view(1, -1)

        # W_final = diag(scale_post) S_A diag(scale_mid) S_B diag(scale_pre), i.e. the deployed form
        W_final = ((S_A * scale_post.view(-1, 1)) @ (S_B * scale_mid.view(-1, 1) * scale_pre)).to(W.dtype)

        return {
            "W_final": W_final,
            "A": S_A.mT.to(W.dtype),  # (mid, out)
            "B": S_B.to(W.dtype),  # (mid, in)
            "A_latent": A_latent.mT,  # (mid, out)
            "B_latent": B_latent,  # (mid, in)
            "scale_pre": scale_pre,
            "scale_mid": scale_mid,
            "scale_post": scale_post,
        }

    A_final = A_final * scale_factor

    scale_pre = B_final.abs().mean(dim=0).view(1, -1)
    scale_post = A_final.abs().mean(dim=1).view(1, -1)

    # W_final = A_final @ B_final  -> F.linear(A_final, B_final.T)
    W_final = F.linear(A_final, B_final.mT)

    return {
        "W_final": W_final,
        "A": A_final.mT,  # (mid, out)
        "B": B_final,  # (mid, in)
        "A_latent": A_latent.mT,  # (mid, out)
        "B_latent": B_latent,  # (mid, in)
        "scale_pre": scale_pre,
        "scale_post": scale_post,
    }


def cubic(x):
    """Cubic rho scheduler with early iterations protection."""
    return min(1.0, x)**3


def linear(x):
    """Linear rho scheduler."""
    return x


def logistic(x, k=5):
    """Logistic rho scheduler."""
    return 1 / (1 + np.exp(-k * (x - 0.5)))


def exp_decay(x, k=5):
    """Exponential decay rho scheduler."""
    return (1 - np.exp(-k * x)) / (1 - np.exp(-k))


def exp_growth(x, k=5):
    """Exponential growth rho scheduler."""
    return (np.exp(k * x) - 1) / (np.exp(k) - 1)


# Register the scheduler functions
RHO_SCHEDULER_REGISTRY.update({
    'cubic': cubic,
    'linear': linear,
    'logistic': logistic,
    'exp_decay': exp_decay,
    'exp_growth': exp_growth,
})
