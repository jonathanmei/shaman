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
