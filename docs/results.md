# Results: Kronecker-factored curvature and middle scale (Qwen3, 1.0 bpw)

All runs: obsidian cluster, 1× A100 80 GB, branch `feat/kron-curvature`, configs in `configs/`, hyperparameters
mirroring the NanoQuant paper (arXiv 2602.06694, Appendix C / Sec. 3.2): 128 × 2048 WikiText-2 calibration samples,
seed 0, shrinkage 0.2, 400 linear-schedule ADMM steps, stage learning rates 1e-4 / 1e-5 / 1e-6 with batch sizes
4 / 1 / 1 and 8 epochs each, target 1.0 bpw. Everything the paper leaves unstated is listed in
[`unstated_hyperparameters.md`](unstated_hyperparameters.md).

Curvature: `diag` is the released method (per-feature second moments); `kron` is the iterative nearest-Kronecker-product
estimate of the per-token empirical Fisher (3 alternating passes) with the Mahalanobis ADMM data term. Scales: 2 =
paper-faithful (`scale_pre`, `scale_post`); 3 = explicit per-rank middle scale (`admm_mid_scale=true`), exported either
with SVID triples (`svid`, the original export) or `balanced` (see [Scale magnitude allocation](#scale-magnitude-allocation)).

Bit budget: the rank rule floors ranks to multiples of 32, so "1.0 bpw" is 0.973 (0.6B, 2 scales), 0.977 (0.6B, 3
scales), 0.986 (1.7B and 4B) actual bpw over the factorised layers, by the paper's own definition (Appendix F, Eq. 60).
The actual bit counter of every run matched these predictions.

## Summary (WikiText-2 perplexity, lower is better)

| model | actual bpw | paper (Table 2) | diag, 2 scales (paper-faithful) | kron, 2 scales | diag, 3 scales (svid) | kron, 3 scales (svid) | diag, 3 scales (balanced) | kron, 3 scales (balanced) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 0.973 / 0.977 | 27.56 | 29.21 | 25.82 | 34.18 | 29.29 | 31.90 | **25.44** |
| Qwen3-1.7B-Base | 0.986 | 19.21 | **18.76** | 19.21 | – | – | – | – |
| Qwen3-4B-Base | 0.986 | 14.29 | 14.86 | running (job 5606588) | – | – | – | – |

## Qwen3-0.6B-Base (2026-09-02)

| arm | job | block-27 PPL before KD | KD loss ep1 → ep8 | **WikiText-2 PPL** | zero-shot mean | wall-clock |
|---|---|---|---|---|---|---|
| diag, 2 scales (paper-faithful) | 5606323 | 32.7 | 2.957 → 2.899 | 29.21 | 0.387 | ~1 h 20 |
| kron, 2 scales | 5606324 | 27.8 | 2.883 → 2.845 | 25.82 | 0.406 | 1 h 52 |
| diag, 3 scales (svid) | 5606092 | 37.8 | 3.008 → 2.941 | 34.18 | 0.408 | 1 h 09 |
| kron, 3 scales (svid) | 5606093 | 30.8 | 2.910 → 2.880 | 29.29 | 0.414 | 1 h 52 |
| diag, 3 scales (balanced) | 5606570 | 36.2 | 2.959 → 2.900 | 31.90 | 0.415 | 1 h 13 |
| **kron, 3 scales (balanced)** | 5606569 | 27.4 | 2.886 → 2.847 | **25.44** | 0.410 | 1 h 33 |

Zero-shot accuracy (lm_eval 0.4.9, 0-shot):

| arm | boolq | piqa | hellaswag | winogrande | arc_e | arc_c | mean |
|---|---|---|---|---|---|---|---|
| diag, 2 scales | 0.535 | 0.528 | 0.279 | 0.500 | 0.273 | 0.206 | 0.387 |
| kron, 2 scales | 0.558 | 0.576 | 0.284 | 0.493 | 0.326 | 0.196 | 0.406 |
| diag, 3 scales (svid) | 0.564 | 0.573 | 0.282 | 0.508 | 0.346 | 0.175 | 0.408 |
| kron, 3 scales (svid) | 0.578 | 0.574 | 0.284 | 0.521 | 0.346 | 0.183 | 0.414 |
| diag, 3 scales (balanced) | 0.598 | 0.567 | 0.277 | 0.517 | 0.339 | 0.193 | 0.415 |
| kron, 3 scales (balanced) | 0.565 | 0.570 | 0.283 | 0.522 | 0.328 | 0.195 | 0.410 |

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
- **Middle scale:** with the original SVID export it hurts both arms at 0.6B (diag 34.18 vs 29.21; kron 29.29 vs 25.82)
  despite costing no rank. With the `balanced` export the kron deficit disappears entirely (25.44 vs 25.82, within seed
  noise; pre-KD 27.4 vs 27.8 and KD loss 2.847 vs 2.845 are indistinguishable) and the diag deficit halves (31.90 vs
  29.21; pre-KD 36.2 vs 32.7 while the KD loss matches, 2.900 vs 2.899). The SVID penalty was therefore the magnitude
  allocation, not the middle scale itself; the residual diag gap sits in the block stage and needs a second seed to be
  called real (diag pre-KD perplexities have varied by ~2 between identical runs).
- **Reproduction of the paper (diag, 2 scales):** 0.6B +6 % (29.21 vs 27.56), 1.7B −2 % (18.76 vs 19.21), 4B +4 %
  (14.86 vs 14.29). The sign flips across sizes, consistent with seed variance plus the unstated hyperparameters listed in
  `unstated_hyperparameters.md` (top suspects: per-layer vs per-block tuning schedule of Algorithm 1; seeds).
- Zero-shot accuracies at these sizes are noisy; perplexity is the informative metric.

Cost: kron adds the calibration passes (3 ALS iterations; ~30 min for 0.6B) and raises the per-block time by ~40–70 %.
With CPU-streamed accumulation the 1.7B calibration took 75 min and 4B would have taken ~5 h; the grouped on-GPU
accumulation (`kron_gpu_budget_gb`, commit dedd7e5) removes that overhead (4B: 3 layer groups per pass).

## Scale magnitude allocation (why the 3-scale export is not a fair comparison)

The 3-scale deficit is fully present before KD (block-27 PPL 37.8 vs 32.7 diag, 30.8 vs 27.8 kron), i.e. it arises in
the block-level STE stage. The ADMM factors have exactly rank-1 magnitude, $|A| = a \otimes \alpha$ and
$|B| = \beta \otimes b$, so both exports agree on the signs and on the deployed weight and differ only in how the
shared magnitude is split between the scale vectors (verified on a synthetic layer: $\beta$ had a CV of 2 % and the two
deployed reconstructions matched to four digits):

| export | `scale_pre` | `scale_mid` | `scale_post` |
|---|---|---|---|
| mean-magnitude (2 scales) | $\bar\beta\, q$ | – | $p/\|a\|$ (lossless: all columns of $|A|$ are identical after the per-rank normaliser) |
| SVID (`admm_mid_scale_export: svid`) | $q/\|q\|$ (unit norm) | $\propto \beta$ | $\|\alpha\|\, p$ (carries the singular value of $|A|$) |
| balanced (`admm_mid_scale_export: balanced`) | $\bar\beta\, q$ (= 2-scale) | $\beta/\bar\beta$ (mean 1) | $p/\|a\|$ (= 2-scale) |

Adam's per-entry step is ≈ lr in absolute terms, so the relative step of a scale entry is lr/|entry|. Under the SVID
allocation `scale_post` was ~26× larger than in the 2-scale export on the synthetic layer (hence ~26× smaller relative
steps at the shared `fact_scale_lr`), the middle scale moved as fast as `scale_pre`, and the split also drifts with the
arbitrary magnitude of the calibration statistics. The `balanced` export removes the confounder: `scale_pre`/`scale_post`
are bit-for-bit the 2-scale values, and the middle scale's relative step is simply its own learning rate
(`fact_mid_scale_lr`, `model_kd_mid_scale_lr`; default = the scale learning rate, i.e. ~10× smaller relative steps than
the outer scales, whose entries are ~0.1).

Outcome on Qwen3-0.6B (jobs 5606569 / 5606570, mid-scale LRs at their defaults): kron 29.29 → **25.44** (2-scale: 25.82),
diag 34.18 → 31.90 (2-scale: 29.21). See the 0.6B tables above.

## Infrastructure notes

- The KD stage's `ram` teacher mode holds ~0.62 GB of bf16 logits per 2048-token sample (80 GB for 128 samples);
  `model_kd_teacher: online` recomputes them at each step instead (+~30 % KD time). All configs here use `online`.
- Stage cache (`cache/`): calibration statistics, per-layer ADMM solutions (content-addressed), per-block checkpoints, the
  pre-KD model and KD epochs are all reusable/resumable. Drill: job 5606325 restored all 28 blocks after a crash and
  finished. Runs longer than the 4 h partition limit are resubmitted and resume from their block checkpoints. Results of
  every run are appended to `cache/results.jsonl` on the cluster.

## Pending

- Second seeds for the 0.6B diag arms (2 scales vs 3 scales balanced) to decide whether the residual 2.7-PPL diag gap is
  real; a middle-scale learning-rate sweep with the `balanced` export.
