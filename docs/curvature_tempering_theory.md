# Why tempering the ADMM curvature works, and how it relates to KL-Shampoo and Pro-KLShampoo

Working note, 2026-09-08. Companion to `admm_block_tuning_curvature.html` and `results.md`.

## Setting

The Mahalanobis ADMM data term is a metric on the weight error $E = W_n - AB$,

$$J(E) = \operatorname{tr}\!\big(\tilde L\, E\, \tilde R\, E^\top\big) = \operatorname{vec}(E)^\top (\tilde R \otimes \tilde L)\operatorname{vec}(E),$$

with $\tilde L, \tilde R$ the unit-diagonal (correlation) forms of the Kronecker factors $L \otimes R \approx F$, the
layer's empirical Fisher. *Tempering* replaces each factor by $Q\,\lambda^{p} Q^\top$ (eigenvalues raised to the power
$p$, rescaled so the trace is unchanged); $p = 1/2$ turned the Qwen3-1.7B Kron result from 19.21 to 17.46 PPL.

## Three justifications for $p < 1$

1. **Regret-optimal metric under curvature uncertainty.** Newton and natural-gradient methods use $F$ itself. The
   full-matrix AdaGrad analysis shows that when the curvature sequence is uncertain or non-stationary, the metric
   minimising the regret bound is $H \propto \big(\sum_t g_t g_t^\top\big)^{1/2}$, i.e. the square root of the
   accumulated Fisher; Shampoo inherits this with per-factor exponent $1/4$ because there $L \otimes R$ bounds the
   *square* of the accumulator, whereas the nearest-Kronecker-product fit used here targets $F$ directly, giving
   $p = 1/2$ per factor. Our curvature is exactly "uncertain": estimated on the full-precision model, used after a
   quantised prefix and before a quantised suffix.
2. **Misspecification of the quadratic proxy at 1 bpw.** $\Delta\ell \approx \tfrac12 \operatorname{vec}(\Delta W)^\top
   F \operatorname{vec}(\Delta W)$ is a small-perturbation expansion, but the reconstruction error is 40–50 % of
   $\|W\|$. Cross-entropy grows at most linearly in large logit perturbations, so along a direction with eigenvalue
   $\lambda$ the Gauss–Newton curvature decays from $\lambda$ (small residual $r$) toward $\sqrt{\lambda}/|r|$ (large
   residual), as for a Huber loss. The effective spectrum at the perturbation scale of one-bit quantisation lies
   between $\lambda$ and $\sqrt\lambda$: $\lambda^{1/2}$ is the Fisher of a robustified loss whose transition scale
   matches the quantisation error. This is why a smooth power beat a hard condition-number floor in the block-loss
   screen (`results.md`, 4-block screen).
3. **Eigenvalue shrinkage of a heavy-tailed sample covariance.** The factors are sample covariances of strongly
   dependent, heavy-tailed tokens; the Frobenius/NKP ALS weights $x_t^\top R\, x_t$ are quartic in the activation
   magnitude, so a few massive-activation tokens inflate the leading eigenvalues (effective rank 33–86 of 1024 in
   blocks 0–3 at 0.6B). Rotation-equivariant nonlinear shrinkage (Stein, Ledoit–Wolf) keeps the eigenvectors and
   pulls the eigenvalue dispersion in; $\lambda^{1/2}$ is a monotone version of that and exactly undoes the
   quartic-to-quadratic over-weighting. The existing `calib_shrinkage` toward a scaled identity is its linear analogue.

Arguments 1–2 concern how the curvature is **used** and hold for a perfectly estimated $F$; argument 3 concerns how
it is **estimated**.

## KL-Shampoo and Pro-KLShampoo

*KL-Shampoo* fits $L \otimes R$ by minimising $D_{\mathrm{KL}}\big(\mathcal N(0,F)\,\|\,\mathcal N(0, L\otimes R)\big)$
instead of the Frobenius distance. The fixed point is the matrix-normal MLE

$$L \propto \sum_t \big(x_t^\top R^{-1} x_t\big)\,\delta_t\delta_t^\top,\qquad
  R \propto \sum_t \big(\delta_t^\top L^{-1} \delta_t\big)\,x_t x_t^\top,$$

the same alternating update as the NKP fit with the **inverse** of the other factor as the token weight. The weights are
leverage scores, bounded and nearly uniform, so massive tokens stop dominating; Stein's loss weights over- and
under-estimated eigenvalues in relative terms; and the $c L,\ R/c$ scale ambiguity disappears. It is a better
*estimator* of $F$ (argument 3) and is not an approximation of tempering: for a matrix-normal Fisher it is consistent
for $L \otimes R$ and reproduces $p = 1$.

*Pro-KLShampoo* (Sun & Wei, arXiv 2605.06316) observes that KL-Shampoo's factors have a spike-and-flat spectrum and
restricts one factor to $\hat R = U S U^\top + \mu_\perp P_\perp$ ($r$ tracked directions with full spectrum, one
shared eigenvalue on the complement), minimising the KL objective directly over that family (an M-projection). For a
fixed eigenbasis the optimal $U$ is the top eigenspace of the whitened second moment and $\mu_\perp$ the trace-average
of the complement, and the approximation gap is $\tfrac{m(n-r)}{2}\log\frac{\mathrm{AM}}{\mathrm{GM}}$ of the tail
eigenvalues, zero when the tail is flat (exactly so under a rank-$\rho$ signal-plus-noise gradient model). The
orthogonalisation of the complement that gives the optimizer its Muon-like update has no counterpart in a metric,
but the spectral projection transfers directly: keep the top-$r$ eigenpairs, replace the bulk by its mean
(`admm_curvature_spike_rank`). It leaves the well-estimated spikes intact and denoises the bulk, a different
regulariser from tempering, which compresses spikes and bulk alike.

## The grid and what it showed (results.md, 2026-09-08)

Estimator {Frobenius NKP, KL} × structure {full, spike-plus-flat, $r = 64$} × tempering {$p = 1$, $p = 1/2$} on the
4-block screen at 0.6B, then the best arms at 1.7B. The prediction was: if KL or spike-plus-flat at $p = 1$ matches
tempering, the mechanism is estimation (3); if $p = 1/2$ is still needed on top, it is (1)–(2).

- **0.6B**: the KL estimator alone gives the full gain (block-3 PPL 14.78 vs 15.14; its factors have condition
  numbers 39–49 instead of 330–683) and tempering it adds nothing (14.81). Mechanism at this size: estimation (3).
- **1.7B**: KL alone reaches 17.04 (tempered NKP: 17.46; tempered NKP + fresh input factor + refresh: 17.28), and
  tempering the KL factors adds a further −0.32 (16.72). Both mechanisms contribute at this width, consistent with
  (1)–(2) growing with the size of the perturbation and the uncertainty of the curvature.
- **Exponent**: $p = 1/4$ on the KL factors gives 17.37, worse than both $p = 1$ and $p = 1/2$. The optimum sits
  at the square root, as argument (1) predicts for a Kronecker fit of $F$ itself; Shampoo's per-factor ¼ belongs to
  a fit of $F^2$ and over-tempers here.
- **Spike-plus-flat** hurts for both estimators at 0.6B (+0.2 to +0.7): the bulk of these Fisher factors is not
  flat, so the Pro-KLShampoo projection discards structure the ADMM data term uses. The projection remains
  attractive for *cost* at large width (O(nr) instead of O(n²) per factor), but not for accuracy here.
