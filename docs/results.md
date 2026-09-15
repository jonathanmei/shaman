# Results: Kronecker-factored curvature and middle scale (Qwen3, 1.0 bpw)

All runs: obsidian cluster, 1× A100 80 GB, branch `feat/kron-curvature`, configs in `configs/`, hyperparameters
mirroring the NanoQuant paper (arXiv 2602.06694, Appendix C / Sec. 3.2): 128 × 2048 WikiText-2 calibration samples,
seed 0, shrinkage 0.2, 400 linear-schedule ADMM steps, stage learning rates 1e-4 / 1e-5 / 1e-6 with batch sizes
4 / 1 / 1 and 8 epochs each, target 1.0 bpw. Everything the paper leaves unstated is listed in
[`unstated_hyperparameters.md`](unstated_hyperparameters.md).

Curvature: `diag` is the released method (per-feature second moments); `kron` is the iterative nearest-Kronecker-product
estimate of the per-token empirical Fisher (3 alternating passes) with the Mahalanobis ADMM data term. Scales: 2 =
paper-faithful (`scale_pre`, `scale_post`); 3 = explicit per-rank middle scale (`admm_mid_scale=true`).

Bit budget: the rank rule floors ranks to multiples of 32, so "1.0 bpw" is 0.973 (0.6B, 2 scales), 0.977 (0.6B, 3
scales), 0.986 (1.7B and 4B) actual bpw over the factorised layers, by the paper's own definition (Appendix F, Eq. 60).
The actual bit counter of every run matched these predictions.

## Summary (WikiText-2 perplexity, lower is better)

| model | actual bpw | paper (Table 2) | diag, 2 scales (paper-faithful) | kron, 2 scales | diag, 3 scales | kron, 3 scales |
|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 0.973 / 0.977 | 27.56 | 29.21 | **25.82** | 34.18 | 29.29 |
| Qwen3-1.7B-Base | 0.986 | 19.21 | 18.76 | 19.21 (**16.72** with KL-Shampoo factors, tempered ADMM, fresh input factor and curvature refresh, 2026-09-08) | – | – |
| Qwen3-4B-Base | 0.986 | 14.29 | 14.86 | timed out (job 5606588); **14.11** with KL-Shampoo factors, tempered ADMM (p = ½), fresh input factor and curvature refresh (2026-09-09) | – | – |

## Qwen3-0.6B-Base (2026-09-02)

| arm | job | block-27 PPL before KD | KD loss ep1 → ep8 | **WikiText-2 PPL** | zero-shot mean | wall-clock |
|---|---|---|---|---|---|---|
| diag, 2 scales (paper-faithful) | 5606323 | 32.7 | 2.957 → 2.899 | 29.21 | 0.387 | ~1 h 20 |
| **kron, 2 scales** | 5606324 | 27.8 | 2.883 → 2.845 | **25.82** | 0.406 | 1 h 52 |
| diag, 3 scales | 5606092 | 37.8 | 3.008 → 2.941 | 34.18 | 0.408 | 1 h 09 |
| kron, 3 scales | 5606093 | 30.8 | 2.910 → 2.880 | 29.29 | 0.414 | 1 h 52 |

Zero-shot accuracy (lm_eval 0.4.9, 0-shot):

| arm | boolq | piqa | hellaswag | winogrande | arc_e | arc_c | mean |
|---|---|---|---|---|---|---|---|
| diag, 2 scales | 0.535 | 0.528 | 0.279 | 0.500 | 0.273 | 0.206 | 0.387 |
| kron, 2 scales | 0.558 | 0.576 | 0.284 | 0.493 | 0.326 | 0.196 | 0.406 |
| diag, 3 scales | 0.564 | 0.573 | 0.282 | 0.508 | 0.346 | 0.175 | 0.408 |
| kron, 3 scales | 0.578 | 0.574 | 0.284 | 0.521 | 0.346 | 0.183 | 0.414 |

## Qwen3-1.7B-Base (2026-09-02, 2 scales)

| arm | job | block-27 PPL before KD | KD loss ep1 → ep8 | **WikiText-2 PPL** | zero-shot mean | wall-clock |
|---|---|---|---|---|---|---|
| diag | 5606401 | 20.35 | 2.545 → 2.496 | **18.76** | 0.427 | 1 h 18 |
| kron | 5606402 | 20.19 | 2.505 → 2.471 | 19.21 | 0.421 | 3 h 11 (75 min of it CPU-streamed calibration) |

| arm | boolq | piqa | hellaswag | winogrande | arc_e | arc_c | mean |
|---|---|---|---|---|---|---|---|
| diag | 0.614 | 0.589 | 0.307 | 0.508 | 0.352 | 0.189 | 0.427 |
| kron | 0.620 | 0.573 | 0.300 | 0.519 | 0.330 | 0.182 | 0.421 |

## Qwen3-4B-Base (2026-09-02, 2 scales)

| arm | job | block-35 PPL before KD | KD loss ep1 → ep8 | **WikiText-2 PPL** | zero-shot mean | wall-clock |
|---|---|---|---|---|---|---|
| diag | 5606403 | 15.77 | 2.250 → 2.208 | 14.86 | 0.455 | 2 h 30 |
| kron | 5606588 | running (grouped GPU calibration: 3 layer groups of ~44 GiB) | | | | |

| arm | boolq | piqa | hellaswag | winogrande | arc_e | arc_c | mean |
|---|---|---|---|---|---|---|---|
| diag | 0.560 | 0.610 | 0.328 | 0.557 | 0.456 | 0.218 | 0.455 |

## Observations

- **0.6B:** Kronecker curvature lowers perplexity by 12 % against the paper-faithful diagonal baseline at identical
  ranks and bits (25.82 vs 29.21) and by 14 % with 3 scales, despite a *higher* Euclidean ADMM reconstruction error
  (0.40 vs 0.34): the Mahalanobis objective trades Euclidean fit for curvature-weighted fit.
- **1.7B:** the advantage vanishes (kron 19.21 vs diag 18.76, −2 % for diag; zero-shot 0.421 vs 0.427). Kron still has
  the lower pre-KD block perplexity (20.19 vs 20.35) and the lower KD loss (2.471 vs 2.496), yet the higher final
  perplexity, which points to run-to-run noise of the order of ±0.5 PPL at this size rather than a systematic effect.
  A second seed for each arm would settle it.
- **Middle scale** hurts both arms at 0.6B (diag 34.18 vs 29.21; kron 29.29 vs 25.82) despite costing no rank.
- **Reproduction of the paper (diag, 2 scales):** 0.6B +6 % (29.21 vs 27.56), 1.7B −2 % (18.76 vs 19.21), 4B +4 %
  (14.86 vs 14.29). The sign flips across sizes, consistent with seed variance plus the unstated hyperparameters listed in
  `unstated_hyperparameters.md` (top suspects: per-layer vs per-block tuning schedule of Algorithm 1; seeds).
- Zero-shot accuracies at these sizes are noisy; perplexity is the informative metric.

Cost: kron adds the calibration passes (3 ALS iterations; ~30 min for 0.6B) and raises the per-block time by ~40–70 %.
With CPU-streamed accumulation the 1.7B calibration took 75 min and 4B would have taken ~5 h; the grouped on-GPU
accumulation (`kron_gpu_budget_gb`, commit dedd7e5) removes that overhead (4B: 3 layer groups per pass).

## Infrastructure notes

- The KD stage's `ram` teacher mode holds ~0.62 GB of bf16 logits per 2048-token sample (80 GB for 128 samples);
  `model_kd_teacher: online` recomputes them at each step instead (+~30 % KD time). All configs here use `online`.
- Stage cache (`cache/`): calibration statistics, per-layer ADMM solutions (content-addressed), per-block checkpoints, the
  pre-KD model and KD epochs are all reusable/resumable. Drill: job 5606325 restored all 28 blocks after a crash and
  finished. Runs longer than the 4 h partition limit are resubmitted and resume from their block checkpoints. Results of
  every run are appended to `cache/results.jsonl` on the cluster.

## Block-loss screen on the first 4 blocks (2026-09-07, Qwen3-0.6B-Base, 2 scales, kron curvature)

Motivation: in the 2×2 screen of 2026-09-07 (jobs 5615563–5615566) the dense Mahalanobis block loss was worse than
the diagonal one from block 0 on (block 3: 15.96/15.99 vs 15.49/15.29 PPL; block 27: 28.8/28.9 vs 28.0/27.4) while
run-to-run noise at block 3 is only ~0.2. The two `scales_latent` arms crashed at the start of KD (latents were
deleted by `finalize()` after block tuning; fixed on branch `latent-kd-kl-kron`). The 4-block screen
(`max_blocks: 4`, `tune_model: false`, `block_diagnostics: true`, configs `configs/qwen3_0p6b_screen_*.json`,
~15 min per arm) reads out the held-out PPL after block 3 and logs **both** block losses whichever is optimised.

| arm | job | block 0 | block 1 | block 2 | block 3 PPL | block-3 down_proj final losses (diag / dense, ×1e-5) | cond. number of the dense matrix (blocks 0–3) |
|---|---|---|---|---|---|---|---|
| s0 diag (control) | 5615704 | 19.32 | 14.48 | 15.08 | **15.28** | 3.48 / 3.21 | 331 / 217 / 487 / 684 |
| s1 mahalanobis | 5615705 | 20.16 | 14.69 | 15.49 | 15.79 | 4.36 / 3.05 | same |
| s2 maha + `block_loss_cond_max` 10 | 5615706 | 21.52 | 14.57 | 15.24 | 15.49 | | ≤ 10 |
| s2 maha + `block_loss_cond_max` 30 | 5615707 | 20.69 | 14.48 | 15.20 | 15.42 | | ≤ 30 |
| s2 maha + `block_loss_power` 0.5 | 5615708 | 21.67 | 14.36 | 14.99 | 15.30 | 3.70 / 3.44 | 18 / 15 / 22 / 26 |
| s3 diag, `block_loss_source` plain | 5615709 | 20.52 | 14.48 | 15.14 | 15.47 | 4.88 / 4.79 | 66 / 56 / 65 / 63 |
| s3 maha, `block_loss_source` plain | 5615710 | 20.13 | 14.56 | 15.26 | 15.85 | 5.47 / 4.73 | same |
| s4 diag, `admm_input_factor` fresh | 5615711 | 18.18 | 14.21 | 14.80 | 15.36 | 3.15 / 2.81 | (NKP; ADMM uses the fresh factor) |
| s4 maha, `admm_input_factor` fresh | 5615712 | 19.82 | 14.48 | 15.26 | 15.49 | 3.74 / 2.63 | same |

Observations:

- **The dense objective is optimised, and it is the wrong proxy.** The Mahalanobis arm ends block 3 with a lower
  dense loss than the diag arm (3.05 vs 3.21) but a 25 % higher diagonal loss (4.36 vs 3.48) and +0.5 PPL. So the
  gap is not an optimisation failure; weighting block-output errors by the FP gradient covariance trades off the
  wrong directions at 1 bpw (docs/admm_block_tuning_curvature.html §5.3 argues the block objective should be the
  reference; at these error magnitudes the diagonal, near-isotropic form is the more robust one).
- **Conditioning explains most of the size of the gap.** Capping the condition number at 30 halves it, at 10 recovers
  60 %, and spectral tempering with power 0.5 (condition number ~20 instead of ~300–700) closes it entirely
  (15.30 vs 15.28) without beating the diagonal.
- **The plain (unweighted) output-gradient covariance is worse, not better.** It is far better conditioned (~60) and
  has effective rank ~500 of 1024, confirming that the NKP token weighting concentrates the spectrum, yet its
  diagonal is a worse importance than the NKP diagonal (15.47 vs 15.28) and the dense form is the worst arm (15.85).
  The token weighting by MLP-input energy, which up-weights massive-activation tokens, therefore *helps* the
  diagonal block loss; the unweighted Gauss–Newton surrogate is not the right target either.
- **Fresh ADMM input factor (docs/admm_block_tuning_curvature.html §4.3-2).** Measuring the input second moment on
  the quantised prefix right before each layer is binarised is neutral for the diag chain at 4 blocks (better on blocks
  0–2, 15.36 vs 15.28 at block 3, inside the ±0.2 noise) and recovers 0.3 PPL of the Mahalanobis gap. The drift
  diagnostic shows why the effect is not larger: the calibration-time NKP input factor and the plain second moment are
  nearly orthogonal in their top-16 eigenspaces already at block 0 (principal angles 88–90°, relative spectral
  perturbation 11–162), i.e. the difference is the token weighting, not staleness, and ADMM tolerates either. The
  note's eq. 9 holds numerically: for `down_proj` the Gauss–Newton prediction tr(L ΔW R_fresh ΔWᵀ) matches the
  observed dense block-loss increase of the ADMM solution to within 10 % at blocks 0, 1 and 3 (e.g. 3.43e-6
  predicted vs 3.44e-6 observed at block 3) and within 40 % at block 2, where `tune_nonfact` had not converged.
  Staleness is predicted to grow with depth, so the full-model diag+fresh arm remains the open test.
- Cost: the dense loss plus diagnostics raises the per-block time from 132 s to ~220 s at 0.6B.

Decision (stopping rule of the plan): the dense block loss is dropped from further tuning experiments at 0.6B; dense
curvature stays in ADMM, where it earns its gain. Conditioning knobs and the fresh input factor remain available for
the 1.7B check, where the Kron advantage vanished and depth/width make staleness more likely.

## 2×2 rerun with retained latents: block loss × KD mode (2026-09-07/08, Qwen3-0.6B-Base, 2 scales, kron)

Branch `latent-kd-kl-kron`; all arms `retain_latent: true` so that the pre-KD model of each block-loss chain is
built once (jobs 5615715 diag, 5615716 maha; the first 4 blocks were resumed from the screen's checkpoints) and every
KD arm reloads it from the cache (~10–15 min per KD arm). Configs `configs/qwen3_0p6b_kron_2scale*_rl`-paths.

| block loss | pre-KD PPL (block 27) | no KD (control) | KD scales only | KD scales + latent, lr 1e-5 | best latent variant |
|---|---|---|---|---|---|
| diag | 28.05 | 28.05 (5615726) | **26.24** (5615715) | 32.05 (5615722) | 27.75, row-normalised latents @1e-5 (5615725) |
| mahalanobis | 29.01 | 29.01 (5615718) | **26.47** (5615716) | 28.90 (5615717) | 26.61, lr 1e-7 (5615720) |

Latent-KD variants (all on the cached pre-KD models; flip fraction = share of the 418 M binary entries whose sign
changed during KD; PPL after epoch 1 / epoch 8):

| chain | latent lr | normalised | flipped bits | PPL after epoch 1 → 8 | final PPL | zero-shot mean |
|---|---|---|---|---|---|---|
| diag | 1e-5 | no | 2.29 % | – | 32.05 | 0.387 |
| diag | 1e-6 | no | 1.62 % | 929.6 → 30.8 | 30.77 | 0.400 |
| diag | 1e-7 | no | 1.26 % | 48.4 → 29.9 | 29.88 | 0.394 |
| diag | 1e-5 | yes | 0.83 % | 40.2 → 27.8 | 27.75 | 0.421 |
| maha | 1e-5 | no | 1.61 % | – | 28.90 | 0.416 |
| maha | 1e-6 | no | 1.08 % | 33.8 → 30.7 | 30.73 | 0.402 |
| maha | 1e-7 | no | 0.72 % | 32.4 → 26.6 | 26.61 | 0.414 |
| maha | 1e-5 | yes | 0.52 % | 39.5 → 28.0 | 27.96 | 0.416 |

Zero-shot means of the scale-only arms: diag 0.399, maha 0.415 (no-KD controls 0.397 / 0.415); the differences are
within the noise noted for this model size.

Observations:

- **The crash is fixed and the pipeline is consistent.** The no-KD controls reproduce the block-27 perplexity exactly
  from the cached model with retained latents, all arms report the same 0.9729 bpw, and the per-epoch held-out
  perplexity of the STE forward equals the final hardened perplexity (`model_kd_eval_every_epoch`).
- **Scale+latent KD does not beat scale-only KD at 0.6B; the planned lr of 1e-5 is destructive.** Every latent arm is
  far worse than scale-only KD after epoch 1 (PPL 32–930 versus ~27) and recovers only partially over 8 epochs; the
  best variants (maha @1e-7: 26.61; diag normalised: 27.75) end at or below the scale-only result (26.47 / 26.24). The
  mechanism is visible in the diagnostics: the latents are tiny (median |latent| ≈ 2e-3, 23 % below 1e-3), Adam's
  first steps are ±lr regardless of gradient size, so every latent with margin below the lr flips on a single noisy
  gradient, and a binary flip is a full-magnitude weight perturbation whatever the latent's margin was. Even at 1e-7,
  0.7–1.3 % of all bits flip (2.9–5.3 M bits) and the KD then spends its epochs repairing the damage while the
  training KL (2.76–2.85) ends *below* the scale-only value (2.85–2.86): it overfits the 128 calibration samples.
  Row-normalising the latents (uniform flip budget per row) is the least harmful setting but still loses 1.3–1.5 PPL.
- **Block loss after KD.** Scale-only KD narrows the Mahalanobis deficit from 0.96 PPL pre-KD to 0.23 (26.47 vs
  26.24), inside the run-to-run noise, so the dense block loss neither helps nor clearly hurts the final model; the
  4-block screen above explains the pre-KD gap.
- Versus the earlier runs (26.11 / 27.13 for the same two chains without retained latents): diag +0.13, maha −0.66,
  i.e. within the ±0.5 seed/nondeterminism spread; retaining latents does not change the scale-only path.

Next steps if latent KD is pursued: gate flips on gradient-sign consistency over several steps (or use SGD-type
updates whose size scales with the gradient) instead of Adam's sign-like first steps, keep latents frozen for the
first epochs, and enlarge the calibration set; a 1.7B check of the fresh ADMM input factor (`admm_input_factor:
fresh`) remains the open item of docs/admm_block_tuning_curvature.html.

## KD feature distillation on a fixed pre-KD model (2026-09-08, Qwen3-0.6B, diag chain)

All arms reload the same pre-KD checkpoint (`pre_kd_checkpoint`, the diag chain model of job 5615715) and run the
8-epoch scale-only KD with an added residual-stream feature term: the relative squared error of the student's
hidden state after every block against the online teacher's, averaged over blocks (`model_kd_feature_weight`).
Held-out PPL is evaluated after every epoch.

| feature weight | job | final PPL | best epoch (PPL) | final KL | feature term |
|---|---|---|---|---|---|
| 0 (control) | 5615753 | 26.26 | 6 (26.25) | 2.848 | – |
| 0.1 | 5615755 | 26.19 | 6 (26.15) | 2.851 | 0.0277 |
| 1 | 5615754 | **26.07** | 6 (26.06) | 2.849 | 0.0277 |
| 10 | 5615756 | 26.10 | 7 (26.08) | 2.851 | 0.0271 |
| 100 | 5615757 | 26.37 | 7 (26.35) | 2.864 | 0.0267 |

- The KD stage is close to deterministic given the pre-KD model: the control reproduces the original run's 26.24
  to within 0.02, so KD-only comparisons on a fixed checkpoint have a noise floor of a few hundredths, far below
  the ±0.5 of full reconstruction chains. This makes the cached pre-KD model the right test bed for KD changes.
- Feature distillation gives a small, real gain (−0.2 PPL at weights 1–10) and hurts at 100, where it crowds out
  the KL. The feature term itself hardly moves (2.7 % relative residual-stream error throughout) because only the
  392 scale vectors are trainable: it acts as a regulariser on the logit fit, not as a fit of the features. More
  trainable full-precision parameters (RMSNorm / q_norm / k_norm weights) would be the natural next pairing.
- Every arm peaks at epoch 6–7 and drifts up slightly afterwards; early stopping on held-out PPL is worth ~0.02.

## Qwen3-1.7B-Base: ADMM spectral tempering and curvature refresh (2026-09-08, 2 scales, kron)

Motivated by the 0.6B block-loss screen (power-0.5 tempering of the dense block-loss matrix removed its
ill-conditioning) and by docs/admm_block_tuning_curvature.html (input-side staleness grows with depth). Both arms
temper ADMM's unit-diagonal Kronecker factors with `admm_curvature_power: 0.5` (eigenvalues raised to the power
0.5, trace preserved; `core/curvature.py`). The second arm additionally hands ADMM the fresh input second moment of
every layer (`admm_input_factor: fresh`) and re-estimates the Kronecker factors of all remaining layers on the
quantised prefix every 7 blocks (`curvature_refresh_every: 7`, one warm-started ALS pass, 53–104 s each).

| arm | job | block 0 | block 7 | block 14 | block 21 | block 27 (pre-KD) | KD loss ep1 → ep8 | **WikiText-2 PPL** | zero-shot mean | wall-clock |
|---|---|---|---|---|---|---|---|---|---|---|
| diag (2026-09-02) | 5606401 | | | | | 20.35 | 2.545 → 2.496 | 18.76 | 0.426 | 1 h 18 |
| kron (2026-09-02) | 5606402 | | | | | 20.19 | 2.505 → 2.471 | 19.21 | 0.421 | 3 h 11 |
| kron, tempered ADMM | 5615751 | 9.97 | 11.54 | 12.57 | 14.61 | 18.24 | 2.494 → 2.461 | 17.46 | 0.435 | 2 h 22 |
| kron, tempered + fresh R + refresh/7 | 5615752 | 10.03 | 11.44 | 12.46 | 14.49 | **18.02** | 2.489 → 2.457 | **17.28** | 0.446 | 2 h 09 |

Paper (Table 2): 19.21.

Observations:

- **Tempering the ADMM curvature recovers the Kron advantage at 1.7B and beats every previous 1.7B number**: 17.46 vs
  19.21 for untempered Kron (−9 %) and 18.76 for diag (−7 %), 2.1 PPL better pre-KD as well. The earlier 1.7B
  reversal therefore looks like ADMM over-trusting a concentrated dense spectrum rather than seed noise. The
  comparison runs are from 2026-09-02 (older code; the block loss and calibration are the same), so a same-code
  untempered Kron control would make the attribution exact.
- **Fresh input factor plus periodic refresh adds a further −0.18 PPL** (17.28) with a consistently lower trajectory
  from block 7 on (−0.1 to −0.2 at every block boundary), within the ±0.5 noise of single runs but in the predicted
  direction at every checkpoint. The three refreshes cost 4 minutes in total; the arm was faster overall because the
  grouped GPU calibration replaced the 75-minute CPU-streamed one.
- Zero-shot means rose with perplexity (0.435 / 0.446 vs 0.421–0.426).
- Cost: 2 h 10–20 per 1.7B run on the current GPUs (calibration ~10 min, ~4.5 min per block, KD 6 min).

Summary across sizes (WikiText-2 PPL, best arm per size): 0.6B 26.07 (kron, diag block loss, feature-KD weight 1),
1.7B 17.28 (kron, tempered ADMM, fresh R, refresh/7). Open: same-code untempered 1.7B control; tempering at 0.6B and
4B; tempering exponent and refresh period sweeps.

## Estimator × structure × tempering grid for the ADMM curvature (2026-09-08, Qwen3-0.6B, 4-block screen)

Motivation and theory: `curvature_tempering_theory.md`. Arms differ only in how ADMM's dense Kronecker factors are
estimated and conditioned: estimator {Frobenius/NKP, KL-Shampoo (`kron_fit: kl`, inverse-weighted ALS)} × structure
{full, spike-plus-flat with 64 spikes (`admm_curvature_spike_rank`)} × tempering {p = 1, p = 0.5
(`admm_curvature_power`)}. Diag block loss, `max_blocks: 4`, jobs 5615851–5615855 and 5615858–5615860.

| estimator | structure | p | job | block 0 | block 1 | block 2 | **block 3** |
|---|---|---|---|---|---|---|---|
| NKP | full | 1 (control) | 5615851 | 18.39 | 14.25 | 14.85 | 15.14 |
| NKP | full | 0.5 | 5615852 | 16.49 | 14.30 | 14.78 | 15.08 |
| NKP | spike+flat 64 | 1 | 5615853 | 20.65 | 14.54 | 15.05 | 15.86 |
| NKP | spike+flat 64 | 0.5 | 5615854 | 17.61 | 14.44 | 14.84 | 15.58 |
| KL | full | 1 | 5615855 | 17.60 | 14.05 | 14.51 | **14.78** |
| KL | full | 0.5 | 5615858 (repeat 5615863) | 16.61 | 14.05 | 14.51 | 14.81 (14.88) |
| KL | spike+flat 64 | 1 | 5615859 (repeat 5615864) | 17.93 | 14.41 | 14.81 | 15.27 (15.00) |
| KL | spike+flat 64 | 0.5 | 5615860 (repeat 5615865) | 18.98 | 14.21 | 14.66 | 15.00 (15.06) |

Control repeats at block 3 so far: 15.14, 15.28, 15.29, 15.36, 15.49 (noise ≈ 0.2). The three KL arms were run twice
(second values in parentheses; the spike+flat p = 0.5 repeat resumed from the first run's blocks 0–1): the ordering
KL/full < KL/spike+flat < NKP holds in both repeats.

Phase 3b (tempered ADMM p = 0.5 **and** a tempered dense block loss, `block_loss: mahalanobis`, `block_loss_power`
0.5): NKP source 15.03 (5615861), plain source 14.98 (5615862), both within noise of the tempered-ADMM control with
the diagonal block loss (15.08). Tempering both stages does not compound; it does rescue the plain covariance as a
block loss (15.85 untempered → 14.98).

Spectrum of the block-output factor (`mlp.down_proj`, shrunk), blocks 0–3: NKP condition number 330 / 218 / 487 /
683 with effective rank 86 / 154 / 52 / 33; KL condition number 49 / 41 / 44 / 39 with effective rank 692 / 698 /
673 / 680 and a top-50 trace share of 0.15 instead of 0.35–0.70.

Observations:

- **The KL-Shampoo estimator is the best single change at 0.6B**: −0.36 PPL at block 3 against the control, below
  every control repeat, and lower at every block boundary. Its factors are already well conditioned, and tempering
  them adds nothing (14.81 vs 14.78), so at this size the mechanism behind tempering is estimation error of the
  Frobenius fit (argument 3 of the theory note): the leverage-weighted fit removes the massive-token inflation that
  the power law was compensating for.
- **Tempering the NKP factors at 0.6B** is within noise at block 3 (15.08 vs 15.14) but clearly better at block 0
  (16.5 vs 18.4), consistent with the large 1.7B gain being a conditioning effect that grows with width.
- **Spike-plus-flat projection hurts** for both estimators (+0.7 / +0.5 for NKP, +0.5 / +0.2 for KL at p = 1 / 0.5).
  The projected factor is well conditioned, so the loss comes from discarding the bulk eigenvalue structure that ADMM's
  data term uses; the Pro-KLShampoo argument for flattening (the bulk of a rank-ρ signal-plus-noise gradient model is
  exactly flat) does not hold for these calibration-time Fisher factors, whose bulk still carries correlation structure.
  Tempering partially repairs the projection (it re-weights the spikes downward), which is why spike+flat at p = 0.5
  beats spike+flat at p = 1.
- Cost: the KL fit adds one eigendecomposition per layer per ALS pass (fp64); the 0.6B calibration took 25 min
  instead of 10.

### 1.7B follow-up: KL estimator with and without tempering (2026-09-08)

Both on top of the fresh input factor and the refresh every 7 blocks (the periodic refresh uses the configured fit);
configs `qwen3_1p7b_kron_2scale_kl_refresh.json` (p = 1) and `qwen3_1p7b_kron_2scale_kl_temper_refresh.json` (p = 0.5).

| arm | job | block 7 | block 14 | block 21 | block 27 (pre-KD) | KD loss ep1 → ep8 | **WikiText-2 PPL** | zero-shot mean | wall-clock |
|---|---|---|---|---|---|---|---|---|---|
| NKP, tempered + fresh R + refresh (previous best) | 5615752 | 11.44 | 12.46 | 14.49 | 18.02 | 2.489 → 2.457 | 17.28 | 0.446 | 2 h 09 |
| KL, p = 1, fresh R + refresh | 5615872 | 11.25 | 12.16 | 14.11 | 17.78 | 2.494 → 2.462 | 17.04 | 0.436 | 3 h 00 |
| KL, p = 0.5, fresh R + refresh | 5615873 | 11.20 | 12.12 | 14.05 | 17.69 | 2.492 → 2.458 | **16.72** | 0.448 | ~3 h 50 |
| KL, p = 0.25, fresh R + refresh (2026-09-09) | 5616739 | 11.30 | 12.33 | 14.42 | 18.42 | 2.506 → 2.468 | 17.37 | 0.438 | 2 h 25 |

- The KL estimator lowers the trajectory at every checkpoint relative to the tempered NKP factors (−0.2 to −0.4 PPL
  from block 7 on) and ends at 17.04; tempering the KL factors adds another −0.32 (16.72), so at 1.7B both
  mechanisms contribute, whereas at 0.6B tempering was redundant once the estimator was fixed. This matches the
  theory note: estimation error (argument 3) dominates at 0.6B; the use-side arguments (1)–(2) grow with width.
- 16.72 is 13 % below the paper's 19.21 and 11 % below the diag baseline (18.76); zero-shot mean 0.448 vs 0.426.
### 4B (2026-09-09): KL factors, tempered ADMM (p = ½), fresh input factor, refresh every 9 blocks

`configs/qwen3_4b_kron_2scale_kl_temper_refresh.json`, jobs 5616745 (calibration + all 36 blocks, timed out at the
4-hour limit during KD) and 5616747 (reloaded the cached pre-KD model, KD + evaluation, 15 min).

| arm | block 0 | block 9 | block 18 | KD loss ep8 | **WikiText-2 PPL** | zero-shot mean |
|---|---|---|---|---|---|---|
| diag (2026-09-02, job 5606403) | | | | 2.208 | 14.86 | 0.455 |
| paper (Table 2) | | | | | 14.29 | |
| KL, p = ½, fresh R, refresh/9 | 7.98 | 9.02 | 9.85 | 2.186 | **14.11** | 0.463 |

- First 4B result below the paper (−1.3 %) and 5 % below the diag baseline; zero-shot mean 0.463 vs 0.455. The
  earlier untempered Kron 4B run never finished, so there is no same-size untempered comparison.
- Cost: KL calibration with GPU Cholesky inverses ~35 min for 36 × 7 layers in 3 groups; ~6 min per block; refreshes
  at blocks 9 / 18 / 27 took 355 / 271 s (189 / 126 remaining layers). Reconstruction alone is ~3 h 50 at 4B, so a
  single 4-hour job cannot also run KD; the resubmission resumes from the cached pre-KD model in minutes.
- Note: the first 4B attempt (job 5616740) hit a CUDA OOM at its first refresh because `collect_stats` moved the
  model, with ~36 GB of dense factor buffers, to the GPU; the refresh now detaches the factors to the CPU first
  (commit 4e18bb3). Any change to the block-stage source files invalidates the block checkpoints, which is why the
  rerun started from block 0.

- **The tempering exponent is U-shaped with its optimum at the square root**: p = 1 → 17.04, p = 0.5 → 16.72,
  p = 0.25 → 17.37 (worse than no tempering at every block from block 7 on). The literal per-factor Shampoo
  exponent (¼) over-tempers here because the Kronecker fit targets the Fisher itself, not its square, so the
  regret-optimal exponent is ½ per factor (theory note, argument 1). With the Cholesky-damped GPU inverses the
  1.7B KL calibration took 9 minutes instead of 55.
- Cost: the KL calibration at 1.7B took ~55 min instead of 10 because the damped inverses of the 2048–6144-wide
  factors were computed on the CPU; fixed on the branch by inverting on the accumulation device (uncommitted at the
  time of writing, see the pending-state note). The p = 0.5 arm came within minutes of the 4-hour limit for that
  reason.

## KD-stage screen at 4B on the cached pre-KD model (2026-09-09, branch `kd-norms-fact-stab`)

Motivation: cheapest of the five levers ranked after the 4B result (2026-09-09 brainstorm). The KD stage is nearly
deterministic given the pre-KD model (noise ≈ 0.02 at 0.6B), so three cheap KD changes were screened on the cached
4B pre-KD model of job 5616745 (`pre_kd_checkpoint`, `cache/model/4259d0d2…`): residual-stream feature distillation
at weight 1 (−0.2 at 0.6B), training the 145 RMSNorm / q_norm / k_norm weight vectors alongside the 504 scale
vectors (`model_kd_norm_weights`, lr 1e-5; zero bit cost), and best-epoch selection on WikiText-2 *validation*
perplexity (`model_kd_select_best`). Calibration data stays at the paper's 128 sequences. Configs
`configs/qwen3_4b_kl_kd*.json`, ~25 min per arm incl. zero-shot.

| arm | job | KD loss ep8 | test PPL after epochs 1 / 4 / 6 / 8 | **WikiText-2 PPL** | zero-shot mean |
|---|---|---|---|---|---|
| control (scales only) | 5616760 | 2.186 | 14.21 / 14.12 / 14.10 / 14.11 | 14.106 | 0.463 |
| feature-KD w = 1 | 5616761 | | | 14.133 | 0.463 |
| + norm weights | 5616762 | 2.186 | 14.21 / 14.12 / 14.11 / 14.10 | 14.103 | 0.465 |
| best epoch on validation | 5616763 | | validation 14.50 → 14.39 (ep 6), still falling | 14.106 | 0.463 |
| all three | 5616764 | 2.189 (KL) | 14.24 / 14.14 / 14.13 / 14.12 | 14.124 | 0.461 |

- **Null result at 4B.** All arms lie within ±0.03 of the control (14.106, reproducing 14.11). The feature term's
  −0.2 at 0.6B does not transfer; the extra normalisation parameters change nothing; validation perplexity is still
  decreasing at epoch 6, so the last epoch is already the best and early stopping has nothing to restore. Scale-only
  KD at 4B is saturated with respect to these levers.
- Ledger caveat: the `git_commit` of these rows reads 884e1cb because the shared cluster checkout advanced to the
  track-B commit while they ran; the code that ran is eaafbac (the commit hash is read when the ledger line is
  written).

## Factor-tuning stabilisation screen (2026-09-09, Qwen3-0.6B, 4 blocks, KL factors, p = ½)

Motivation: in the 4B logs `tune_fact` recovers almost none of the block-loss jump caused by binarising v/o/down
(and the training block loss sometimes rises during the stage), while 1.4–1.9 % of latent signs flip per pass with
Adam's ±lr first steps on latents of median magnitude ≈ 2e-3, the mechanism that broke latent KD. Arms rescale the
fresh ADMM latents row-wise to unit mean magnitude before tuning (`fact_latent_normalize`, forward unchanged) and/or
change `fact_binary_lr`. Control: `qwen3_0p6b_grid_kl_full_p05.json` (14.78 / 14.81 / 14.88 in three runs).

| arm | job | flips per layer | block 3 PPL |
|---|---|---|---|
| control (lr 1e-5) | 5615855 / 5615858 / 5615863 | 1.4–1.9 % | 14.78 / 14.81 / 14.88 |
| normalised latents, lr 1e-5 | 5616765 | ~1e-4 | 15.04 |
| lr 1e-6 | 5616766 | | 14.89 |
| normalised, lr 1e-6 | 5616767 | ~0 | 14.92 |
| normalised, lr 1e-4 | 5616768 | | 15.07 |

- **Null (slightly negative).** Every stabilised arm is at or above the control range. Suppressing the flips
  (normalised @1e-5, @1e-6) costs 0.1–0.2 PPL, and a larger margin-aware flip budget (normalised @1e-4) costs the
  same, so the flips the current tuner makes are net useful and the rising training loss seen for `o_proj` is not
  hurting the held-out perplexity at this scale. `fact_latent_normalize` stays available but off.

## Non-uniform rank allocation and tail-block logit objective (2026-09-09, Qwen3-0.6B, full runs)

Recipe for all arms: KL factors, ADMM p = ½, fresh input factor, refresh every 7 blocks, diag block loss, scale-only
KD (`configs/qwen3_0p6b_kl_ra_*.json`, `qwen3_0p6b_kl_tail4*.json`). Mechanics and the hypotheses below are written
up in `rank_allocation_note.html`. Rank allocation: per-layer bit multiplier = depth ramp exp(ρ(b/(B−1) − ½)) ×
per-type weight (q 0.85, k 0.9, v 1.1, o 1.0, gate 1.0, up 1.05, down 1.15, from the median block-loss jump each
type's binarisation caused in the 4B logs), renormalised to the uniform rule's total bits ("parity") or to exactly
1.0 bpw ("full"). Tail objective: the last K blocks are reconstructed against the FP suffix's logits (forward KL,
`tail_logit_blocks`) instead of the weighted block MSE, optionally mixed with it (`tail_logit_mix`).

| arm | job | actual bpw | pre-KD PPL (block 27) | **WikiText-2 PPL** |
|---|---|---|---|---|
| control (uniform ranks) | 5616769 | 0.9729 | 27.35 | 25.48 |
| depth ramp ρ = 0.6 | 5616770 | 0.9729 | 25.31 | 24.31 |
| type weights | 5616771 | 0.9729 | 25.55 | 24.12 |
| **ramp + type, parity** | 5616772 | 0.9728 | 24.19 | **23.25** |
| ramp + type, full 1.0 bpw | 5616773 | 1.0000 | 25.08 | 24.15 |
| tail K = 4, pure KL | 5616774 | 0.9729 | 24.79 | 24.57 |
| tail K = 4, mix 0.5 | 5616775 | 0.9729 | 25.36 | 24.55 |

- **Rank allocation is the largest single gain so far**: −2.2 PPL (−8.7 %) against the same-code control at
  identical bits, 16 % below the paper's 27.56 and well below the previous 0.6B best (26.07). Ramp and type weights
  each help alone and stack. The ramp arm is far worse on early blocks by construction (block 0: 24.5 vs 16.3) and
  overtakes the control around block 21, the predicted mechanism (half of the 4B pre-KD damage sat in the last 8 of
  36 blocks).
- **Spending the rounding remainder (full budget) did not add to it** (24.15 vs 23.25). The fill rule is
  sensitivity-blind: it grew 120 layers by one 32-step, skewed to q/k and to blocks 0–11, while the late-block MLP
  layers were already capped at rank min(in, out) = 1024; part of the gap is single-run noise (±0.5).
- **The tail-block logit objective is the second real lever**: −0.9 PPL post-KD (−2.6 pre-KD) for both variants;
  the arms track the control exactly through block 23 and separate over the last four blocks. Pure KL and the
  0.5 mix are indistinguishable.
- Cost: unchanged per block for the ranks; the tail objective adds ~25 % to the last four blocks at 0.6B.
- Follow-ups: rank ceiling lifted to 2 × min(in, out) (`rank_max_ratio`) at ρ = 0.6 and ρ = 1.0, ρ = 1.0
  capped, ranks + tail combined, each at parity and at the full budget (below); 4B with the parity allocation
  (job 5617467, running).

### Follow-ups: rank ceiling, steeper ramp, ranks + tail, parity vs full (2026-09-09, Qwen3-0.6B, full runs)

Same recipe and type weights as above; ρ is the depth ramp, cap is `rank_max_ratio` × min(in, out).

| arm | job | actual bpw | pre-KD PPL (block 27) | **WikiText-2 PPL** |
|---|---|---|---|---|
| ρ = 0.6, cap 1× , parity (reference, above) | 5616772 | 0.9728 | 24.19 | 23.25 |
| ρ = 0.6, cap 2×, parity | 5617472 | 0.9728 | 24.04 | **23.17** |
| ρ = 1.0, cap 2×, parity | 5617473 | 0.9729 | 24.42 | 23.68 |
| ρ = 1.0, cap 1×, parity | 5617474 | 0.9728 | 27.04 | 25.94 |
| ρ = 0.6, cap 1×, parity + tail K = 4 | 5617475 | 0.9728 | 23.95 | 23.82 |
| ρ = 0.6, cap 2×, full 1.0 bpw | 5617476 | 0.9999 | 24.12 | 23.33 |
| ρ = 1.0, cap 2×, full 1.0 bpw | 5617477 | 1.0000 | 23.98 | 23.32 |
| ρ = 1.0, cap 1×, full 1.0 bpw | 5617478 | 0.9999 | 25.30 | 24.35 |
| ρ = 0.6, cap 1×, full + tail K = 4 | 5617479 | 1.0000 | 24.81 | 24.51 |

- **Lifting the rank ceiling is neutral at ρ = 0.6** (23.17 vs 23.25, inside the ±0.5 noise) and necessary at
  ρ = 1.0: with the cap the steeper ramp cannot place its bits (late MLP layers saturate at 1024) and loses 2.7 PPL
  (25.94); with the cap lifted it recovers to 23.68, still no better than ρ = 0.6. The depth ramp is at or past its
  optimum around 0.6 for this model.
- **Parity vs full is noise**: full lost at ρ = 0.6 (23.33 vs 23.17) and won at ρ = 1.0 (23.32 vs 23.68), so the
  earlier 0.9 gap (24.15 vs 23.25) was not systematic. The 2.7 % of extra bits are worth little wherever the current
  fill rule puts them.
- **The tail objective does not stack with the rank allocation** (23.82 vs 23.25) although it still lowers the pre-KD
  perplexity (23.95 vs 24.19): once the last blocks have the extra rank, the logit-level fit buys less and KD recovers
  less on top of it. Single runs, so a 0.6 deficit is at the edge of the noise band; there is no sign of the −0.9 it
  gave on uniform ranks. The full-budget twin (5617479, 24.51) confirms it: both tail arms are the two worst ρ = 0.6
  results.
- Everything at ρ = 0.6 sits at 23.2–23.3 whatever the cap or budget: the hand-set allocation has plateaued, which is
  the motivation for the measured allocation below.

## Measured-sensitivity rank allocation (2026-09-09/10, Qwen3-0.6B, commit da395d6)

Replaces the hand-set depth ramp and type table by a per-layer sensitivity curve measured at calibration time
(`rank_sensitivity: admm`; `core/rank_probe.py`): after the Kronecker statistics are registered, every layer is
solved by a 50-iteration ADMM at 0.5 / 1.0 / 1.5 × its uniform rank (`rank_probe_ranks`, `rank_probe_iters`), the
deployed binary matrix is rebuilt and scored by the Gauss–Newton weight error `J(r) = tr(L ΔW R ΔWᵀ)` with the
layer's own Fisher factors (raw scale, comparable across layers), and a power law `J = exp(a) r^-β` is fitted. The
allocator (`allocate_ranks_measured`) starts every layer at rank 32 and hands out 32-steps by largest predicted loss
decrease per bit until the parity or full bit target is met. The ramp / type multipliers, if set, act as a prior on
the curve level. Probe cost at 0.6B: 707 s for 196 layers (25 s per block), cached under its own artifact key and
shared by all arms of the same calibration.

What the probe measured (0.6B): exponents are nearly uniform, β = 0.75–0.97 for every layer and depth, so the
allocation is driven by the *level* of the curve (Fisher magnitude × reconstruction error), not by its slope. The
resulting allocation at parity: v_proj 843 (uniform 480), up 803, down 779, gate 663, o 533, k 521, q 427; bits per
block relative to uniform 1.31 1.10 1.13 1.25 1.25 1.16 1.02 0.96 0.92 0.92 0.86 0.88 0.78 0.76 0.76 0.81 0.93 0.90
0.92 1.04 0.99 1.02 0.99 0.96 0.94 0.98 1.20 1.25, i.e. U-shaped in depth (blocks 0–5 and 26–27 up, 12–15 down),
unlike the monotone hand ramp. One outlier: `2.mlp.down_proj` has a curve level 100× every other layer (J = 1.85 vs
≈ 0.01) and takes the rank cap; its logged block-loss jump is ordinary, so the Fisher of that layer is inflated
rather than the layer being fragile.

Predictor check (4-block screen, job 5617824, `block_diagnostics`): Spearman correlation between the fitted J at the
chosen rank and the logged diagonal block-loss jump caused by binarising the layer, over the 28 layers of blocks 0–3:
0.77 overall, 0.96 / 0.86 / 0.82 / 0.71 within blocks 0–3 (ranking the seven layer types within a block), not
resolvable across depth within a type (n = 4). The block-3 PPL of the screen (14.01) is below the uniform control
range (14.78–14.88) by construction: the allocation gives blocks 0–3 10–31 % more bits.

| arm | job | actual bpw | pre-KD PPL (block 27) | **WikiText-2 PPL** |
|---|---|---|---|---|
| uniform ranks (control) | 5616769 | 0.9729 | 27.35 | 25.48 |
| hand table: ramp 0.6 + type weights, parity | 5616772 | 0.9728 | 24.19 | **23.25** |
| measured (admm), parity, no multipliers | 5617842 | 0.9729 | 24.53 | 23.75 |
| **measured × ramp 0.6 prior, parity** | 5617843 | 0.9728 | 23.96 | **22.96** |
| measured, full 1.0 bpw | 5617844 → 5627853 | 1.0000 | 24.68 | 23.60 |

(5617844 hit the 4 h limit at block 20, the probe plus the larger late ranks making 640 s blocks; 5627853 resumed from
its block checkpoints and finished in 25 min.)

- **Measured × ramp prior is the new 0.6B best (22.96)**: the Fisher-weighted predictor supplies the within-block
  ranking and the ramp supplies the depth weighting it under-values, and the two combine to −2.5 PPL against uniform
  ranks (−0.3 against the hand table, inside the single-run band but in the predicted direction).

- **The measured allocation recovers most of the hand table's gain with no tuned knob**: −1.7 PPL against uniform
  ranks (23.75 vs 25.48) versus −2.2 for the hand table; the 0.5 gap to the hand table is at the edge of the ±0.5
  single-run band. The trajectories differ in shape: measured is ahead of the hand table at every block boundary
  through block 24 (block 3: 14.10 vs 15.24; block 14: 16.72 vs 17.72; block 21: 19.61 vs 20.17) and is overtaken
  over the last three blocks (24.53 vs 24.19 pre-KD), where the hand ramp holds 35 % more bits and the measured
  allocation about 20–25 % more. The Fisher-weighted predictor therefore under-values the last blocks relative to
  what the end-to-end perplexity rewards, which is exactly what the ramp-prior arm tests.
- **Measured, full budget (23.60) vs parity (23.75)**: within noise, as for the hand table; the 2.7 % of extra bits
  remain worth little.
- Open: a 4B run of measured × ramp; its probe (36 blocks, 9728-wide factors) runs at calibration time and is cached
  under its own key, so the run spans two to three 4 h jobs resuming from the cache.

## Qwen3-4B-Base with the hand-table rank allocation (2026-09-10)

`configs/qwen3_4b_kl_ra_both.json`: the 14.11 recipe (KL factors, ADMM p = ½, fresh input factor, refresh every 9)
with `rank_budget: parity`, depth ramp 0.6 and the type weights above; 241 of 252 layers differ from the uniform rule,
ranks 448–2560. Jobs 5617467 (timed out at block 35/36; its checkpoints were then invalidated by commit da395d6),
5620415 (recomputed from a checkout pinned at ba41b0d, `ob:~/code/shaman-4b`, timed out at block 31) and 5627852
(resumed from block 31, 63 min incl. KD and evaluation).

| arm | actual bpw | block 35 (pre-KD) | KD loss ep8 | **WikiText-2 PPL** | zero-shot mean |
|---|---|---|---|---|---|
| uniform ranks (2026-09-09) | 0.9864 | 15.77 (diag) / – | 2.186 | 14.11 | 0.463 |
| **ramp 0.6 + type weights, parity** | 0.9864 | 13.82 | 2.163 | **13.55** | 0.450 |
| measured × ramp 0.6, parity (`qwen3_4b_best.json`; 2026-09-14, jobs 5663775 → 5666862) | 0.9864 | – | – | 13.80 | 0.463 |
| paper (Table 2) | | | | 14.29 | |

- **Measured sensitivity × ramp 0.6 at 4B: 13.80 / 0.463** (job 5663775 timed out at block 27, resumed as 5666862
  from the pinned checkout `ob:~/code/shaman-4b` at 26d8fd4, 88 min incl. KD and evaluation). +0.25 PPL against the
  hand table while the zero-shot mean returns to the uniform-rank level (0.463 vs 0.450). At 0.6B the measured
  allocator beat the hand table (22.96 vs 23.25); at 4B it does not: the depth prior alone does not reproduce what
  the type weights (down/up/gate over q/k/v) bought there. Single runs both; the seed-1 twin
  `configs/qwen3_4b_best_seed1.json` is the next 4B run.

- **−0.56 PPL (−4.0 %) at identical bits**, 5.2 % below the paper and 8.8 % below the paper-faithful diag baseline
  (14.86). The pre-KD perplexity (13.82) is already below the previous *post*-KD result. Zero-shot mean 0.450 vs
  0.463: a 1.3-point drop, inside the noise noted for these tasks but the first time perplexity and zero-shot move in
  opposite directions; worth a second seed before drawing conclusions.
- Cost: the late blocks with rank up to 2560 take 450–490 s (vs ~360 s uniform), so a full 4B run no longer fits one
  4 h job even without KD; two jobs with block resume are the norm now.
- Ledger caveat as before: job 5627852 ran ba41b0d code (the ledger hash matches because it ran from the pinned
  checkout).

Summary across sizes (WikiText-2 PPL, best arm per size, 1.0 bpw target): 0.6B **22.96** (measured × ramp),
1.7B 16.72 (not yet rerun with rank allocation), 4B **13.55** (hand-table ranks, code at 26d8fd4; the `_best.json`
recipe with measured × ramp ranks gives 13.80). The ADMM fast path
(branch `admm-fast-sylvester`, inexact Sylvester solve with early stopping) reproduces the 1.7B recipe at 16.67
vs 16.72 (job 5627825), i.e. lossless, and is the natural way to buy back the time the larger late ranks cost.

### Config names after the 2026-09-14 merge

The recipe configs were consolidated into `configs/qwen3_{0p6b,1p7b,4b}_best.json` (+ `_best_seed1.json` twins).
Runs logged above under `qwen3_0p6b_kl_ms_ramp_parity`, `qwen3_1p7b_kl_ms_ramp_parity` and
`qwen3_4b_kl_ms_ramp_parity` (job 5663775) used byte-identical settings apart from `qmodel_path`; the 4B hand-table
config `qwen3_4b_kl_ra_both.json` and every other screening config survive only in history at `26d8fd4`.

## Shared fresh input factors; the ADMM compute experiment (2026-09-10 to 2026-09-14, branch fresh-factor-sharing)

Kept from branch `admm-fast-sylvester` (jobs 5627358 / 5627825 / 5627826, full write-up on that branch):

- **Shared fresh input factors.** q, v and k read `input_layernorm`'s output and gate, up read
  `post_attention_layernorm`'s; converted layers are not `nn.Linear`, so `tune_nonfact` cannot change those inputs
  between the group's ADMM calls. The fresh factor is therefore measured once per group (`shared_input_groups`,
  `fresh_input_factor` in `compress_block.py`), validated by a one-sample probe, and its normalised
  eigendecomposition is cached for ADMM (`EigCache`, registered factors only). Saves 3 of 7 fresh-factor
  measurements (128 block forwards each) and 3 small eigendecompositions per block; exact by construction.
- **`kron_eigh_dtype: float32`** in the 1.7B and 4B best-arm configs: the knob only reaches the ADMM
  eigendecompositions, whose eigenvalues are clamped and whose output feeds a sign projection.

Dropped after measuring: an inexact Sylvester X-update (stale k×k eigenbasis as preconditioner with Rayleigh, QR and
PCG rungs) saved ~10 % of ADMM time at 1.7B (14.3 vs 15.9 s per layer; PPL 16.66 vs 16.72) and nothing at 4B
(26.1 vs 26.4 s): at k ≈ 2000, n = 9728 its extra `n²k` matmuls and residual syncs cost what the fp32 `eigh` costs.
Early stopping on a frozen Z is inapplicable: at iteration 400 every layer still flips 30–20 000 signs per iteration
under the linear ρ schedule. ADMM is ~40 % of 4B reconstruction; the tuning stages are the larger lever.

## Spike-plus-flat projection of the factor vs of its inverse (2026-09-14, Qwen3-0.6B, 4-block screen, branch `spectral-projection-screen` @ 34bde42)

Question (`curvature_tempering_theory.md`, "Modelling the inverse instead"): a low-rank-plus-identity model of the
*inverse* factor is the same matrix family as Pro-KLShampoo's model of the factor; it differs in which end of the
spectrum is kept exactly (bottom = the cheap directions) and in the shared value the fit implies (harmonic instead of
arithmetic mean). The two-sided projection (`admm_curvature_spike_rank` top eigenvalues and
`admm_curvature_dip_rank` bottom eigenvalues exact, `admm_curvature_flat_mean` for the middle) contains both as
corners. Base config `qwen3_0p6b_kl_ms_screen.json` (current recipe: KL factors, p = ½, fresh shared input factor,
refresh/7, measured ranks × ramp 0.6, parity), 64 exact eigenvalues per kept side, at the recipe's
`calib_shrinkage` 0.2 and at 0 (on shrunk factors the bottom of the spectrum is a plateau, so the inverse
projection is only meaningful unshrunk). Prediction before the runs: inverse ≤ forward < two-sided ≈ control.

| arm | config | job | spike / dip / mean | shrink | block 0 | block 1 | block 2 | **block 3** |
|---|---|---|---|---|---|---|---|---|
| control | `qwen3_0p6b_proj_ctrl_s02` | 5666720 | – | 0.2 | 13.97 | 13.55 | 13.88 | **14.15** |
| forward (Pro-KLShampoo) | `..._fwd_s02` | 5666721 | 64 / 0 / AM | 0.2 | 14.43 | 13.66 | 14.01 | 14.29 |
| inverse | `..._inv_s02` | 5666722 | 0 / 64 / HM | 0.2 | 14.77 | 13.67 | 14.06 | 14.38 |
| two-sided | `..._two_s02` | 5666723 | 64 / 64 / GM | 0.2 | 14.23 | 13.60 | 13.93 | 14.24 |
| control | `..._ctrl_s0` | 5666724 | – | 0 | 14.30 | 13.44 | 13.78 | **14.14** |
| inverse | `..._inv_s0` | 5666725 | 0 / 64 / HM | 0 | 15.81 | 13.64 | 14.02 | 14.39 |
| two-sided | `..._two_s0` | 5666726 | 64 / 64 / GM | 0 | 14.52 | 13.72 | 14.13 | 14.48 |

All seven completed in 25–27 min. Block-3 single-run noise on this screen is ≈ 0.2 (control repeats in the
2026-09-08 grid).

Spectrum diagnostics (`block_diagnostics` now prints, per layer and factor, the dispersion of the eigenvalues a
projection replaces: `log(AM/GM)` is the KL gap of the forward fit, `log(GM/HM)` that of the inverse fit; with no
projection the whole spectrum is summarised). Block 3, unit-diagonal factors:

- whole spectrum, shrunk 0.2: `log(AM/GM)` 0.13–0.52, `log(GM/HM)` 0.10–0.38; unshrunk: 0.22–1.31 and 0.19–1.16.
  Shrinkage removes two thirds of the dispersion, almost all of it at the bottom.
- middle after removing the top 64 (forward arm, shrunk): 0.08–0.20 / 0.08–0.32 — the remaining dispersion sits at
  the *bottom* of what is left (the inverse-fit gap now exceeds the forward gap), i.e. the middle is not flat.
- middle after removing the bottom 64 (inverse arm, unshrunk): 0.19–1.21 / 0.15–0.81 — the spikes are still in the
  set the harmonic mean replaces, so the inverse projection flattens the dominant directions to a value far below
  them.

Observations:

- **Every projection is worse than its control at every block boundary.** Block-3 penalties: forward +0.14, inverse
  +0.23, two-sided +0.10 (shrunk); inverse +0.25, two-sided +0.34 (unshrunk). The two shrunk arms forward and
  two-sided are inside the noise at block 3 but not at block 0 (+0.3 to +0.5), where every arm is consistently worse.
- **The inverse representation is the worst corner**, as predicted: keeping the cheap directions exactly and
  flattening everything else to the harmonic mean underestimates the curvature of the dominant directions, and the
  fit dumps error there (block 0 unshrunk: 15.81 vs 14.30). Underestimating curvature is the unbounded failure;
  overestimating it (the forward arm's arithmetic mean) is the bounded one.
- **Two-sided does not recover the control.** The diagnostics say why: with 64 eigenvalues removed from each end the
  middle still carries `log(AM/GM)` up to 0.2 (shrunk) and 1.2 (unshrunk) of structure. The bulk of these Fisher
  factors is a smooth slope, not a spike-and-flat spectrum; the KL-Shampoo estimator already makes the spectrum
  well conditioned (2026-09-08 grid), and the remaining slope is what the Mahalanobis data term uses.
- **Shrinkage 0 vs 0.2 is a null result** in the current recipe (14.14 vs 14.15 at block 3; 0.2 is better at
  block 0, 13.97 vs 14.30). The KL estimator and tempering already do what shrinkage was compensating for.

Verdict: closed for accuracy. A structured, eigh-free Sylvester step would only pay if a projection were acceptable;
the cheapest acceptable one (two-sided on shrunk factors, +0.10) is within noise but on the wrong side at every
block, and the ADMM compute saving would be at most the ~40 % ADMM share of block reconstruction. The knobs stay in
the code (defaults off; part of the cache keys) and the projection-gap diagnostic is useful on its own.

### Structured solver at 4B (job 5701940, 2026-09-15): no compute case

The low-rank-plus-identity fast path (`CurvatureFactor`, on whenever the spectrum is projected) was run on the 4B
best recipe with spike 256 / dip 256 / GM. Per-block ADMM totals against the dense run 5663775: block 0 162 vs
175 s, 4 226 vs 247, 7 242 vs 260, 12 198 vs 204, 17 167 vs 173 (5-8 % of ADMM, 1-2 % of block time); block-17
per-layer 16.5 / 12.9 / 28.1 / 12.8 / 23.9 / 36.0 / 36.5 s vs 16.7 / 12.7 / 28.0 / 12.5 / 26.2 / 38.6 / 38.6 s. The
`n^2 k` products the projection removes are not where 4B ADMM time goes; the fp64 `k x k` eigendecomposition run
twice per iteration and the rank-1 projections are. The job was left to time out at block 23 and not resumed; the
projection is not part of any recipe. (At 0.6B, job 5701939, the same arm was within noise, 14.18 vs 14.15, at
identical ADMM time.)

## Efficiency fixes, tuning-budget screen and the KD-only middle scale (2026-09-15, branch `spectral-projection-screen`)

Code (commits 87f0521, e877a52, 815f3d1, ef298c6): fp32 for the per-iteration `k x k` Sylvester eigh
(`SYLVESTER_EIGH_DTYPE`) and `kron_eigh_dtype: float32` in the 1.7B/4B best configs; per-block perplexity evaluation
gated (`block_ppl_every`, screens keep it via `block_diagnostics`) with the tokenised test set cached; `EigCache` in
the rank probe (one decomposition per factor instead of one per candidate rank; the 4B probe went from ~45 to ~13
min); MLP-only tuning forward once the attention half is frozen (`mlp_only_forward`); calibration without
per-sample GPU syncs and with bf16 previous-pass ALS weights on the GPU (4B: 2 layer groups instead of 3, 6 model
passes instead of 9; the first version materialised a whole group in fp32 and OOMed, job 5713189, fixed in
ef298c6); on-device reconstruction error, single weight clone, no per-epoch allocator flush, block targets kept on
the GPU, copy-free tensor hash. Tuning-budget knobs: `tune_epoch_weights` (`type` table from the block-loss jumps,
q/k 0.25, v/o 0.5, gate 0.75, up/down 1; or `measured` from the probe's predicted error at the allocated rank),
`tune_epoch_min_frac`, `tune_plateau_tol`, `nonfact_per_group` (rounds before q, o, gate, down only).

0.6B 4-block screen on the fixed code (`qwen3_0p6b_proj_ctrl_s02.json` = current recipe; jobs 5713182-5713188,
22-25 min each incl. recomputed calibration):

| arm | knob | block 0 | block 1 | block 2 | **block 3** | wall per block |
|---|---|---|---|---|---|---|
| reference (fixes only) | – | 13.72 | 13.55 | 13.87 | **14.15** | 145-150 s |
| plateau | `tune_plateau_tol 0.01` | 13.79 | 13.60 | 13.96 | 14.23 | 109-127 s |
| **type** | `tune_epoch_weights type` | 14.37 | 13.59 | 13.88 | **14.16** | 103-109 s |
| measured | `tune_epoch_weights measured` | 14.44 | 13.60 | 14.04 | 14.30 | 102-110 s |
| group | `nonfact_per_group` | 14.25 | 13.79 | 14.13 | 14.43 | 114-119 s |
| all | type + plateau + group | 14.60 | 13.67 | 14.04 | 14.30 | 102-114 s |

- The fixes are lossless: the reference reproduces the dense control's block-3 PPL (14.15 vs 14.149, job 5666720)
  at 145-150 s per block against 146-169 s (~5 % at 0.6B, where the screen keeps the fp64 eigh and the per-block
  evaluation; the 4B saving is measured by job 5713834).
- **Type-weighted epochs: −28 % block time at +0.01 block-3 PPL** (block 0 is worse by 0.6 and the gap closes by
  block 1: the first block's q/k/v/o get 2-4 epochs and the later blocks absorb it). Adopted into the recipes.
- Plateau stop: −20 % at +0.08, within noise but dominated by `type`; not adopted (can be combined later).
- Measured weights are no better than the type table (+0.15) and cost the same; the table wins on simplicity.
- **Skipping non-factorized rounds hurts** (+0.28, outside noise): every layer's round matters, including the
  intra-group ones. Not adopted.

KD-only middle scale (`model_kd_mid_scale`: a per-rank `scale_mid` initialised to ones is inserted before the
scale-only KD; nothing upstream changes). From the cached pre-KD artifact of job 5617842 (measured ranks, parity, no
ramp; its recorded final PPL 23.75): reference KD rerun **23.78** (job 5713832, reproduces the recorded run to 0.03),
with the middle scale **23.69** (job 5713833): −0.09 PPL for 16 bits per rank (+0.4 % of the factorised bits). KD is
deterministic to ~0.02 on a fixed pre-KD model, so the gain is real but small. Adopted (cheap and positive).

Record of the earlier middle-scale arms (2026-09-02/05): the exact SVID triple through tuning and KD lost 3.5-5 PPL
at 0.6B; the mean-1 rebalanced export (`kron_midbal`) 25.44 vs 25.82 two-scale, within noise; the mid-scale LR-ratio
sweep 26.58 / 26.72 / 26.59 / 27.26 for ratios 0 / 1 / 10 / 100, flat to negative. The degree of freedom only pays
when it enters at the end-to-end stage from the identity.

## Qwen3-8B-Base, best fast 2-scale recipe (2026-09-15, branch `spectral-projection-screen` @ fe181ac)

First 8B run. `configs/qwen3_8b_best.json` = the 4B best config with `tune_epoch_weights: type`,
`kron_gpu_budget_gb: 96` and the fp32 eigh; measured × ramp 0.6 parity ranks (probe 0.6/1.0/1.4), refresh every 9
blocks, 2 scales, no KD middle scale. Job **5721334**: one h200 on partition `lgpus` (no 4 h limit; 3-day request,
400G), from the pinned checkout `ob:~/code/shaman-8b` with `cache/` and `checkpoints/` symlinked to the shared
checkout, log `.objob/logs/nq-8b-best-5721334.out`. Everything (stats, probe, blocks, KD) is computed from scratch.
Readout pending: WikiText-2 PPL, bpw, zero-shot mean, wall time per stage (calibration, probe, per block, KD) for
comparison with the 4B fixes run 5713834 and the paper's 8B point.
