# Results: Kronecker-factored curvature and middle scale (Qwen3, 1.0 bpw)

All runs: obsidian cluster, 1× A100 80 GB, branch `feat/kron-curvature`, configs in `configs/`, hyperparameters
mirroring the NanoQuant paper (arXiv 2602.06694, Appendix C / Sec. 3.2): 128 × 2048 WikiText-2 calibration samples,
seed 0, shrinkage 0.2, 400 linear-schedule ADMM steps, stage learning rates 1e-4 / 1e-5 / 1e-6 with batch sizes
4 / 1 / 1 and 8 epochs each, target 1.0 bpw. Everything the paper leaves unstated is listed in
[`unstated_hyperparameters.md`](unstated_hyperparameters.md).

Curvature: `diag` is the released method (per-feature second moments); `kron` is the iterative nearest-Kronecker-product
estimate of the per-token empirical Fisher (3 alternating passes) with the Mahalanobis ADMM data term. Scales: 2 =
paper-faithful (`scale_pre`, `scale_post`); 3 = explicit per-rank middle scale (`admm_mid_scale=true`).

## Qwen3-0.6B-Base (2026-09-02)

Bit budget: the rank rule floors ranks to multiples of 32, giving identical ranks for 2 and 3 scales
(q/o 640, k/v 480, gate/up/down 736) and therefore **0.9729 bpw (2 scales) / 0.9774 bpw (3 scales)** over the
factorised layers, by the paper's own definition (Appendix F, Eq. 60). The actual bit counter of every run matched these.

| arm | job | actual bpw | block-27 PPL before KD | KD loss ep1 → ep8 | **WikiText-2 PPL** | zero-shot mean | wall-clock |
|---|---|---|---|---|---|---|---|
| paper, NanoQuant 1.00 bit (Table 2) | – | ~0.973 | – | – | 27.56 | – | – |
| diag, 2 scales (paper-faithful) | 5606323 | 0.9729 | 32.7 | 2.957 → 2.899 | 29.21 | 0.387 | ~1 h 20 |
| **kron, 2 scales** | 5606324 | 0.9729 | 27.8 | 2.883 → 2.845 | **25.82** | 0.406 | 1 h 52 |
| diag, 3 scales | 5606092 | 0.9774 | 37.8 | 3.008 → 2.941 | 34.18 | 0.408 | 1 h 09 |
| kron, 3 scales | 5606093 | 0.9774 | 30.8 | 2.910 → 2.880 | 29.29 | 0.414 | 1 h 52 |

Zero-shot accuracy (lm_eval 0.4.9, 0-shot):

| arm | boolq | piqa | hellaswag | winogrande | arc_e | arc_c | mean |
|---|---|---|---|---|---|---|---|
| diag, 2 scales | 0.535 | 0.528 | 0.279 | 0.500 | 0.273 | 0.206 | 0.387 |
| kron, 2 scales | 0.558 | 0.576 | 0.284 | 0.493 | 0.326 | 0.196 | 0.406 |
| diag, 3 scales | 0.564 | 0.573 | 0.282 | 0.508 | 0.346 | 0.175 | 0.408 |
| kron, 3 scales | 0.578 | 0.574 | 0.284 | 0.521 | 0.346 | 0.183 | 0.414 |

Observations:

- Kronecker curvature lowers perplexity by 12 % against the paper-faithful diagonal baseline at identical ranks and bits
  (25.82 vs 29.21) and by 14 % with 3 scales (29.29 vs 34.18). Its Euclidean ADMM reconstruction error is *higher*
  (0.40 vs 0.34 normalised): the Mahalanobis objective trades Euclidean fit for curvature-weighted fit, which is what
  transfers to perplexity.
- The explicit middle scale hurts both arms (diag 34.18 vs 29.21; kron 29.29 vs 25.82) despite costing no rank.
- Our diagonal reproduction is 6 % above the paper's 27.56 at the same budget. Candidate causes are listed in
  `unstated_hyperparameters.md`; the top suspects are the per-layer vs per-block tuning schedule of Algorithm 1 and seed
  variance (two identical diag runs differed by ~2 PPL before KD).
- 0.6B zero-shot accuracies are near chance and do not separate the arms; perplexity is the informative metric.

Cost: kron adds ~30 min of 3-pass calibration and raises the per-block time from ~110–140 s to ~150–190 s on 0.6B.

## Infrastructure notes

- The KD stage's `ram` teacher mode holds ~0.62 GB of bf16 logits per 2048-token sample (80 GB for 128 samples);
  `model_kd_teacher: online` recomputes them at each step instead (+~30 % KD time) and lets 0.6B run in 128 GB host RAM.
- Stage cache (`cache/`): calibration statistics, per-layer ADMM solutions (content-addressed), per-block checkpoints, the
  pre-KD model and KD epochs are all reusable/resumable. Drill: job 5606325 restored all 28 blocks after a crash and
  finished. Results of every run are appended to `cache/results.jsonl` on the cluster.

## Pending

- Qwen3-1.7B-Base and Qwen3-4B-Base, 2 scales, diag vs kron (`configs/qwen3_{1p7b,4b}_{diag,kron}_2scale.json`).
