# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from .curvature import IDENTITY_SPECTRUM, SpectrumSpec

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Local registry for rho schedulers
RHO_SCHEDULER_REGISTRY = {}

# Precision of the k x k eigendecomposition inside every Sylvester X-update (800 per layer). Its eigenvalues are
# clamped and the solution feeds a sign projection, so fp32 is lossless here (the rank probe already runs entirely in
# fp32); the once-per-layer n x n decomposition keeps the configurable ``kron_eigh_dtype``.
SYLVESTER_EIGH_DTYPE = torch.float32


@torch.no_grad()
def power_iteration(A, num_iters=5, v0: torch.Tensor | None = None):
    """
    Power iteration for top singular triplet (u, sigma, v) of A.

    ``v0`` is an optional start vector (``A.shape[1]``); by default one is drawn from the global RNG on ``A``'s
    device. Passing it lets a caller draw the vector elsewhere (another device, a private generator) while
    reproducing the default result exactly.
    """
    n = A.shape[1]
    v = torch.randn(n, device=A.device, dtype=A.dtype) if v0 is None else v0.to(A.device, A.dtype)
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
def svid(W, inner_iters=5, eps=1e-12, v0: torch.Tensor | None = None):
    """
    Sign-Value-Independent Decomposition (SVID).
    Returns u, v, Sg where Sg is sign matrix of W. ``v0``: start vector of the power iteration (see
    :func:`power_iteration`).
    """
    Sg = W.sign()
    Sg[Sg == 0] = 1
    u, s, v = power_iteration(W.abs(), inner_iters, v0=v0)
    u = u * s
    return u, v, Sg


@torch.no_grad()
def rank1_approx(W, inner_iters=5, eps=1e-12, v0: torch.Tensor | None = None):
    """
    Rank-1 approximation using SVID results. ``v0``: start vector of the power iteration (see
    :func:`power_iteration`).
    """
    u, v, Sg = svid(W, inner_iters, eps, v0=v0)
    apx = torch.outer(u, v)
    return apx * Sg


@torch.no_grad()
def _svid_nonneg(W: torch.Tensor, inner_iters: int, eps: float,
                 v0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVID with the rank-1 scale vectors forced to be non-negative (their product is unchanged)."""
    u, v, Sg = svid(W, inner_iters, eps, v0=v0)
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
    return _stabilizer_from_mean(lam.mean(), M, rho, reg, eps)


def _stabilizer_from_mean(lam_mean: torch.Tensor, M: torch.Tensor, rho: float, reg: float,
                          eps: float = 1e-12) -> torch.Tensor:
    """:func:`_sylvester_stabilizer` given the mean eigenvalue of ``Sigma`` directly."""
    diag_mean = M.diagonal().mean().abs()
    return torch.clamp(rho + reg * lam_mean * diag_mean, min=eps)


@torch.no_grad()
def _sylvester_solve_step(Q: torch.Tensor, lam: torch.Tensor, M: torch.Tensor, C: torch.Tensor, rho: float, reg: float,
                          eps: float = 1e-12, eigh_dtype: torch.dtype = SYLVESTER_EIGH_DTYPE) -> torch.Tensor:
    """Solve ``Sigma F M + sigma F = C`` for ``F`` with ``Sigma = Q diag(lam) Q^T``.

    This is the X-update of the Mahalanobis ADMM: ``Sigma`` is the (fixed, eigendecomposed once per
    layer) unit-diagonal curvature factor on the long side of ``F``, ``M`` the small ``k x k`` Gram matrix
    of the other factor and ``C`` the right-hand side (curvature-weighted data term plus ``rho`` times
    the ADMM target). Rotating into both eigenbases makes the operator diagonal:
    ``F = Q [ (Q^T C Q_M) / (lam lam_M^T + sigma) ] Q_M^T``.

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
        Precision of the ``k x k`` eigendecomposition (default :data:`SYLVESTER_EIGH_DTYPE`, fp32).

    Returns
    -------
    torch.Tensor
        ``F`` of shape ``(n, k)`` in the dtype of ``C``.
    """
    orig_dtype = C.dtype
    Q, lam, M, C = (t.to(torch.float32) for t in (Q, lam, M, C))
    M = 0.5 * (M + M.mT)
    mu, Q_M = torch.linalg.eigh(M.to(eigh_dtype))
    mu = mu.to(torch.float32).clamp(min=0.0)
    Q_M = Q_M.to(torch.float32)

    sigma = _sylvester_stabilizer(lam, M, rho, reg, eps)
    C_hat = Q.mT @ C @ Q_M
    F_hat = C_hat / (lam.unsqueeze(1) * mu.unsqueeze(0) + sigma)
    return (Q @ F_hat @ Q_M.mT).to(orig_dtype)


@torch.no_grad()
def _sylvester_solve_structured(U: torch.Tensor, lam_U: torch.Tensor, mu: torch.Tensor, M: torch.Tensor,
                                C: torch.Tensor, rho: float, reg: float, eps: float = 1e-12,
                                eigh_dtype: torch.dtype = SYLVESTER_EIGH_DTYPE) -> torch.Tensor:
    """:func:`_sylvester_solve_step` for ``Sigma = mu I + U diag(lam_U - mu) U^T`` in ``O(n r k + n k^2)``.

    ``Sigma`` is block diagonal in the split ``span(U)`` / its complement, so after rotating the right-hand side
    into the eigenbasis of ``M`` the equation is diagonal on both parts: ``(lam_U mu_M^T + sigma)`` on the ``r``
    kept directions and the scalar ``(mu mu_M + sigma)`` per column on the complement. No ``n x n`` rotation.

    Parameters
    ----------
    U : torch.Tensor
        Orthonormal kept eigenvectors ``(n, r)``.
    lam_U : torch.Tensor
        Their eigenvalues ``(r,)``.
    mu : torch.Tensor
        The shared eigenvalue of the complement (0-d).
    M, C, rho, reg, eps, eigh_dtype
        As in :func:`_sylvester_solve_step`.
    """
    orig_dtype = C.dtype
    U, lam_U, mu, M, C = (t.to(torch.float32) for t in (U, lam_U, mu, M, C))
    M = 0.5 * (M + M.mT)
    mu_M, Q_M = torch.linalg.eigh(M.to(eigh_dtype))
    mu_M = mu_M.to(torch.float32).clamp(min=0.0)
    Q_M = Q_M.to(torch.float32)
    n, r = U.shape
    lam_mean = (lam_U.sum() + mu * (n - r)) / n
    sigma = _stabilizer_from_mean(lam_mean, M, rho, reg, eps)
    C_hat = C @ Q_M  # (n, k)
    C_U = U.mT @ C_hat  # (r, k)
    C_perp = C_hat - U @ C_U
    F_U = C_U / (lam_U.unsqueeze(1) * mu_M.unsqueeze(0) + sigma)
    F_perp = C_perp / (mu * mu_M.unsqueeze(0) + sigma)
    return ((F_perp + U @ F_U) @ Q_M.mT).to(orig_dtype)


class CurvatureFactor:
    """Unit-diagonal curvature factor of the Mahalanobis data term, with a low-rank-plus-identity fast path.

    Holds the dense ``Sigma`` and its eigenpairs. When the spectrum has a flat block (a projected
    :class:`SpectrumSpec` with ``0 < spike_rank + dip_rank < n``) and ``structured`` is set,
    ``Sigma = mu I + U diag(lam_U - mu) U^T`` with ``U`` the ``r`` kept eigenvectors, and every product with
    ``Sigma`` as well as the Sylvester X-update costs ``O(n r k)`` instead of ``O(n^2 k)``. Results are identical
    to the dense path up to floating-point error.

    Parameters
    ----------
    Sigma, lam, Q : torch.Tensor
        Output of :func:`_normalized_curvature` (fp32).
    spectrum : SpectrumSpec
        The conditioning that produced ``lam`` (tells which eigenvalues are exact and which form the flat block).
    structured : bool
        Use the fast path when the spectrum allows it.
    """

    def __init__(self, Sigma: torch.Tensor, lam: torch.Tensor, Q: torch.Tensor, spectrum: SpectrumSpec,
                 structured: bool = True) -> None:
        self.Sigma, self.lam, self.Q = Sigma, lam, Q
        self.U: torch.Tensor | None = None
        n, r = lam.numel(), spectrum.spike_rank + spectrum.dip_rank
        if structured and 0 < r < n:
            order = torch.argsort(lam, descending=True)
            keep = torch.cat([order[:spectrum.spike_rank], order[n - spectrum.dip_rank:]])
            middle = order[spectrum.spike_rank:n - spectrum.dip_rank]
            self.mu = lam[middle].mean()
            self.U = Q[:, keep].contiguous()
            self.lam_U = lam[keep]
            self._delta = (self.lam_U - self.mu).unsqueeze(1)  # (r, 1)

    @property
    def structured(self) -> bool:
        """True when the fast path is active."""
        return self.U is not None

    def to(self, device) -> CurvatureFactor:
        """Copy of the factor with every tensor on ``device`` (the spectrum bookkeeping is shared).

        Parameters
        ----------
        device : str or torch.device
            Target device.

        Returns
        -------
        CurvatureFactor
        """
        other = CurvatureFactor.__new__(CurvatureFactor)
        for k, v in self.__dict__.items():
            setattr(other, k, v.to(device, copy=True) if torch.is_tensor(v) else v)
        return other

    def left(self, X: torch.Tensor) -> torch.Tensor:
        """``Sigma @ X``."""
        if self.U is None:
            return self.Sigma @ X
        return self.mu * X + self.U @ (self._delta * (self.U.mT @ X))

    def right(self, X: torch.Tensor) -> torch.Tensor:
        """``X @ Sigma``."""
        if self.U is None:
            return X @ self.Sigma
        return self.mu * X + ((X @ self.U) * self._delta.mT) @ self.U.mT

    def sylvester(self, M: torch.Tensor, C: torch.Tensor, rho: float, reg: float, eps: float,
                  eigh_dtype: torch.dtype = SYLVESTER_EIGH_DTYPE) -> torch.Tensor:
        """Solve ``Sigma F M + sigma F = C`` (:func:`_sylvester_solve_step`); the ``k x k`` eigh runs in
        ``eigh_dtype`` (fp32 by default, see :data:`SYLVESTER_EIGH_DTYPE`)."""
        if self.U is None:
            return _sylvester_solve_step(self.Q, self.lam, M, C, rho, reg, eps, eigh_dtype)
        return _sylvester_solve_structured(self.U, self.lam_U, self.mu, M, C, rho, reg, eps, eigh_dtype)


class EigCache:
    """Normalised eigendecompositions of curvature factors that several layers share (e.g. the fresh input factor of
    q/k/v or gate/up).

    Only tensors registered with :meth:`register` are cached, so per-layer factors never pile up in memory. Entries
    are keyed by the tensor's storage pointer, shape and the spectrum conditioning (:class:`SpectrumSpec`); the
    registered tensor is held alive so its pointer cannot be recycled while the entry exists.
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
    def _key(cov: torch.Tensor, spectrum: SpectrumSpec, eigh_dtype: torch.dtype) -> tuple:
        return (cov.data_ptr(), tuple(cov.shape), str(cov.device), spectrum, str(eigh_dtype))

    def get(self, cov: torch.Tensor, spectrum: SpectrumSpec,
            eigh_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Cached ``(Sigma, lam, Q)`` of a registered ``cov`` under these knobs, else ``None``."""
        if cov.data_ptr() not in self._shared:
            return None
        return self._entries.get(self._key(cov, spectrum, eigh_dtype))

    def put(self, cov: torch.Tensor, spectrum: SpectrumSpec, eigh_dtype: torch.dtype,
            value: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        """Store ``value`` for a registered ``cov``; a no-op for unregistered tensors."""
        if cov.data_ptr() in self._shared:
            self._entries[self._key(cov, spectrum, eigh_dtype)] = value


@torch.no_grad()
def _normalized_curvature(cov: torch.Tensor, norm_vec: torch.Tensor, eigh_dtype: torch.dtype, eps: float,
                          spectrum: SpectrumSpec = IDENTITY_SPECTRUM, eig_cache: EigCache | None = None,
                          diagnostics: dict | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unit-diagonal curvature factor ``D^-1/2 cov D^-1/2`` (with ``D = norm_vec^2``) and its eigendecomposition.

    Unless ``spectrum`` is the identity, the eigenvalues are projected / tempered (:meth:`SpectrumSpec.apply`, trace
    preserved) and ``Sigma`` is rebuilt from them.

    Parameters
    ----------
    spectrum : SpectrumSpec
        Spectral conditioning (tempering power, two-sided spike-plus-flat projection).
    eig_cache : EigCache, optional
        Cache consulted and filled for factors registered as shared (``norm_vec`` must be the diagonal of ``cov``
        for the cached result to be the right one, which holds for the fresh input factor).
    diagnostics : dict, optional
        Filled with :func:`nanoquant.core.curvature.projection_gaps` of the raw spectrum (left untouched on a cache
        hit).

    Returns
    -------
    tuple
        ``(Sigma, lam, Q)`` in fp32 with eigenvalues clamped at ``eps``.
    """
    if eig_cache is not None:
        hit = eig_cache.get(cov, spectrum, eigh_dtype)
        if hit is not None:
            return hit
    n = norm_vec.reshape(-1).to(torch.float32)
    Sigma = cov.to(torch.float32) / (n.unsqueeze(1) * n.unsqueeze(0))
    Sigma = 0.5 * (Sigma + Sigma.mT)
    lam, Q = torch.linalg.eigh(Sigma.to(eigh_dtype))
    lam = lam.to(torch.float32).clamp(min=eps)
    Q = Q.to(torch.float32)
    if diagnostics is not None:
        diagnostics.update(spectrum.gaps(lam))
    if not spectrum.is_identity:
        lam = spectrum.apply(lam).clamp(min=eps)
        Sigma = (Q * lam) @ Q.mT
        Sigma = 0.5 * (Sigma + Sigma.mT)
    if eig_cache is not None:
        eig_cache.put(cov, spectrum, eigh_dtype, (Sigma, lam, Q))
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
    spectrum: SpectrumSpec = IDENTITY_SPECTRUM,
    eig_cache: EigCache | None = None,
    diagnostics: dict | None = None,
    structured: bool = True,
    side_device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
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
        eigh_dtype: Precision of the once-per-layer ``n x n`` eigendecomposition of each curvature factor (the
               per-iteration ``k x k`` one is always fp32, :data:`SYLVESTER_EIGH_DTYPE`).
        mid_scale: Export an explicit per-rank ``scale_mid`` (see above).
        spectrum: Spectral conditioning of the unit-diagonal factors L, R (tempering power and two-sided
               spike-plus-flat projection, trace preserved; ``core.curvature.SpectrumSpec``); the default leaves
               them untouched.
        eig_cache: Cache of normalised eigendecompositions for curvature factors shared by several layers
               (see :class:`EigCache`); factors not registered there are never cached.
        diagnostics: Optional dict filled with the projection gaps of the raw ``L`` and ``R`` spectra
               (``core.curvature.projection_gaps``), keyed ``"L"`` / ``"R"`` in the orientation of ``W``.
        structured: With a projected ``spectrum``, run the Mahalanobis X-updates through the low-rank-plus-identity
               form of the factors (:class:`CurvatureFactor`, ``O(n r k)`` per product instead of ``O(n^2 k)``);
               ``False`` forces the dense path (same result up to floating-point error).
        side_device: Run the B half of every iteration (its X-, Z- and U-update) on this device in a second
               thread while the A half runs on ``W``'s device; the two halves are independent within an
               iteration and exchange only ``A_z``/``B_z`` at the boundary. ``None`` = serial on one device.
               Same result as the serial path (bitwise on identical hardware).
        generator: RNG for the random initialisation and the power-iteration start vectors (must live on
               ``W``'s device); ``None`` = the global RNG. A fresh generator seeded like ``torch.manual_seed`` gives
               the same result as the global path and does not touch the global state (thread-safe callers).
    """
    if is_transpose:
        results = factorize_admm_nanoquant(W.mT, o_norm, i_norm, mid_rank, outer_iters, inner_iters, reg, False, eps,
                                           rho_scheduler, print_admm_steps, i_cov=o_cov, o_cov=i_cov,
                                           eigh_dtype=eigh_dtype, mid_scale=mid_scale,
                                           spectrum=spectrum, eig_cache=eig_cache, diagnostics=diagnostics,
                                           structured=structured, side_device=side_device, generator=generator)
        if diagnostics is not None and {"L", "R"} <= set(diagnostics):
            diagnostics["L"], diagnostics["R"] = diagnostics["R"], diagnostics["L"]
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
        diag_L = {} if diagnostics is not None else None
        diag_R = {} if diagnostics is not None else None
        Lt, lam_L, Q_L = _normalized_curvature(o_cov.to(device), norm_o, eigh_dtype, eps, spectrum,
                                               eig_cache=eig_cache, diagnostics=diag_L)  # (out, out)
        Rt, lam_R, Q_R = _normalized_curvature(i_cov.to(device), norm_i, eigh_dtype, eps, spectrum,
                                               eig_cache=eig_cache, diagnostics=diag_R)  # (in, in)
        if diagnostics is not None:
            diagnostics["L"], diagnostics["R"] = diag_L, diag_R
        Lf = CurvatureFactor(Lt, lam_L, Q_L, spectrum, structured)
        Rf = CurvatureFactor(Rt, lam_R, Q_R, spectrum, structured)
        W_norm32 = W_norm.to(torch.float32)
        P = Lf.left(Rf.right(W_norm32))  # curvature-weighted target L W R, (out, in)

        def maha_loss(E: torch.Tensor) -> torch.Tensor:
            """``tr(L E R E^T)`` through the factors' products."""
            return (Lf.left(Rf.right(E)) * E).sum()

        if print_admm_steps:
            for name, S, lam, Q in (("L", Lt, lam_L, Q_L), ("R", Rt, lam_R, Q_R)):
                rec = (Q * lam) @ Q.mT
                err = ((rec - S).norm() / S.norm().clamp(eps)).item()
                print(f"\t\t[eigh check] {name}: relative reconstruction error {err:.3e} "
                      f"(min/max eig {lam.min().item():.3e}/{lam.max().item():.3e})")
            print(f"\t\t[curvature] structured fast path: L {Lf.structured} | R {Rf.structured}")
            maha_ref = maha_loss(W_norm32).clamp(eps)

    def _draw(*shape, dtype=None) -> torch.Tensor:
        """Normal draw on the primary device, in the order the serial algorithm consumes the RNG."""
        return torch.randn(*shape, device=device, dtype=W.dtype if dtype is None else dtype, generator=generator)

    # we remove SVD-based init, since random init is (1) faster (2) shows on-par or better performance
    A_ls = _draw(out_features, mid_rank)
    B_ls = _draw(mid_rank, in_features)

    A_z, B_z = A_ls, B_ls
    if outer_iters > 0:
        A_z = rank1_approx(A_ls, inner_iters, eps, v0=_draw(mid_rank))
        B_z = rank1_approx(B_ls, inner_iters, eps, v0=_draw(in_features))

    if print_admm_steps:
        A_z_old = A_z.clone()
        B_z_old = B_z.clone()

    A_u = A_ls - A_z
    B_u = B_ls - B_z

    rho_scheduler_func = RHO_SCHEDULER_REGISTRY[rho_scheduler]

    # --- the two halves of an iteration -------------------------------------------------------------------------
    # The A-update reads (B_z, A_z, A_u) and the B-update reads (A_z, B_z, B_u) of the *previous* iteration (Jacobi
    # sweep), so the halves are independent: with ``side_device`` the B half runs on that device in a second
    # thread and only A_z / B_z cross over at the iteration boundary. The op order inside each half is the serial
    # one, so both paths give the same numbers (bitwise on identical hardware).
    split = side_device is not None
    dev_b = torch.device(side_device) if split else torch.device(device)
    if use_maha:
        Lf_b, Rf_b, P_b = (Lf.to(dev_b), Rf.to(dev_b), P.to(dev_b)) if split else (Lf, Rf, P)
    W_norm_b = W_norm.to(dev_b)
    sa = {"ls": A_ls, "z": A_z, "u": A_u}
    sb = {"ls": B_ls.to(dev_b), "z": B_z.to(dev_b), "u": B_u.to(dev_b)}

    def a_step(rho: float, B_z_here: torch.Tensor, v: torch.Tensor) -> None:
        """X-, Z- and U-update of the A side (primary device); ``B_z_here`` is the previous B_z."""
        with torch.no_grad():
            A_z_prev, A_u_ = sa["z"], sa["u"]
            mid_norm_b = B_z_here.norm(dim=1).clamp(eps)
            B_bar = B_z_here / mid_norm_b.unsqueeze(1)  # (mid, in), unit-norm rows
            if use_maha:
                B_bar32 = B_bar.to(torch.float32)
                M = B_bar32 @ Rf.left(B_bar32.mT)  # B R B^T, (mid, mid)
                C = P @ B_bar32.mT + rho * (A_z_prev - A_u_).to(torch.float32)  # (out, mid)
                A_ls_ = Lf.sylvester(M, C, rho, reg, eps).to(W.dtype)
            else:
                # W_norm.T uses view; keep it
                A_ls_ = _admm_solve_step(B_bar.mT, W_norm.mT, A_z_prev.mT, A_u_.mT, rho, reg, eps).mT
            A_z_ = rank1_approx(A_ls_ + A_u_, inner_iters, eps, v0=v)
            A_u_.add_(A_ls_ - A_z_)
            sa["ls"], sa["z"] = A_ls_, A_z_

    def b_step(rho: float, A_z_here: torch.Tensor, v: torch.Tensor) -> None:
        """X-, Z- and U-update of the B side (``dev_b``); ``A_z_here`` is the previous A_z on that device."""
        with torch.no_grad():
            B_z_prev, B_u_ = sb["z"], sb["u"]
            mid_norm_a = A_z_here.norm(dim=0).clamp(eps)
            A_bar = A_z_here / mid_norm_a  # (out, mid), unit-norm columns
            if use_maha:
                A_bar32 = A_bar.to(torch.float32)
                N = A_bar32.mT @ Lf_b.left(A_bar32)  # A^T L A, (mid, mid)
                C = (A_bar32.mT @ P_b + rho * (B_z_prev - B_u_).to(torch.float32)).mT  # (in, mid)
                B_ls_ = Rf_b.sylvester(N, C, rho, reg, eps).mT.to(W.dtype)
            else:
                B_ls_ = _admm_solve_step(A_bar, W_norm_b, B_z_prev, B_u_, rho, reg, eps)
            B_z_ = rank1_approx(B_ls_ + B_u_, inner_iters, eps, v0=v)
            B_u_.add_(B_ls_ - B_z_)
            sb["ls"], sb["z"] = B_ls_, B_z_

    pool = ThreadPoolExecutor(max_workers=2) if split else None
    A_z_for_b = sa["z"].to(dev_b)  # previous A_z on the B device
    B_z_for_a = sb["z"].to(device)  # previous B_z on the A device
    try:
        for itt in range(outer_iters):
            rho = rho_scheduler_func(itt / outer_iters)
            # start vectors of both rank-1 projections, drawn in the serial order (A then B)
            v_a = _draw(mid_rank)
            v_b = _draw(in_features)

            if split:
                fa = pool.submit(a_step, rho, B_z_for_a, v_a)
                fb = pool.submit(b_step, rho, A_z_for_b, v_b.to(dev_b))
                fa.result()
                fb.result()
                A_z_for_b = sa["z"].to(dev_b, non_blocking=True)
                B_z_for_a = sb["z"].to(device, non_blocking=True)
            else:
                a_step(rho, B_z_for_a, v_a)  # B_z_for_a is still the previous B_z
                b_step(rho, A_z_for_b, v_b)  # A_z_for_b is still the previous A_z
                A_z_for_b, B_z_for_a = sa["z"], sb["z"]

            if print_admm_steps:
                if (itt == 0 or (itt + 1) % 100 == 0 or itt == outer_iters - 1):
                    A_ls, A_z = sa["ls"], sa["z"]
                    B_ls, B_z = sb["ls"].to(device), sb["z"].to(device)
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
                        maha = (maha_loss(E) / maha_ref).item()
                        # the same loss for the un-projected X-variables (A_bar @ B_ls), A_bar from the previous A_z
                        A_bar = A_z_old / A_z_old.norm(dim=0).clamp(eps)
                        E_x = (W_norm - F.linear(A_bar, B_ls.mT)).to(torch.float32)
                        maha_x = (maha_loss(E_x) / maha_ref).item()
                        msg += f" | Mahalanobis(Z): {maha:.5e} | Mahalanobis(X): {maha_x:.5e}"
                    print(msg)

                A_z_old.copy_(sa["z"])
                B_z_old.copy_(sb["z"].to(device))
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    A_ls, A_z, A_u = sa["ls"], sa["z"], sa["u"]
    B_ls, B_z, B_u = (t.to(device) for t in (sb["ls"], sb["z"], sb["u"]))
    del sa, sb, A_z_for_b, B_z_for_a

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
        u_A, v_A, S_A = _svid_nonneg(A_final.to(torch.float32), inner_iters, eps,
                                     v0=_draw(mid_rank, dtype=torch.float32))  # (out,), (mid,), (out, mid)
        u_B, v_B, S_B = _svid_nonneg(B_final.to(torch.float32), inner_iters, eps,
                                     v0=_draw(in_features, dtype=torch.float32))  # (mid,), (in,), (mid, in)

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
