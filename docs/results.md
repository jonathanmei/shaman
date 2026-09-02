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
| Qwen3-1.7B-Base | 0.986 | 19.21 | **18.76** | 19.21 | – | – |
| Qwen3-4B-Base | 0.986 | 14.29 | 14.86 | running (job 5606588) | – | – |

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
