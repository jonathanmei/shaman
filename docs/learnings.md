# Learnings: 1-bit NanoQuant on Qwen3, September 2026

`main` (merge of `rank-allocation`, `measured-rank-alloc-cleanup` and `fresh-factor-sharing`, 2026-09-14) is the
distillation of the `latent-kd-kl-kron` and `kd-norms-fact-stab` experiments. It keeps only the recipe: KL Kronecker
factors, tempered ADMM (p = ½), the fresh input factor (shared across q/k/v and gate/up), periodic curvature refresh,
the measured rank allocator (ADMM probe, depth ramp as the prior, parity/full budgets) and scale-only KD, and ships
one "best recipe" config per model size with a seed-1 twin. Everything that did not move perplexity was removed from
the code: the hand-multiplier allocator (type weights, rank ceiling), the tail-block logit objective, latent KD, the
dense block loss and the spike/cond curvature knobs. Their last implementation is commit `26d8fd4`
(`kd-norms-fact-stab`), which the ledger entries below reference. Full run tables are in `results.md`; the allocator
is derived in `rank_allocation_note.html`.

## Best recipes (WikiText-2 PPL, 1.0 bpw target)

| model | config | recipe | actual bpw | PPL | paper |
|---|---|---|---|---|---|
| Qwen3-0.6B-Base | `configs/qwen3_0p6b_best.json` | KL factors, ADMM p = ½, fresh input factor, refresh/7, **measured ranks × ramp 0.6, parity** | 0.9728 | **22.96** | 27.56 |
| Qwen3-1.7B-Base | `configs/qwen3_1p7b_best.json` | same recipe with measured × ramp ranks (not yet run; 16.72 with uniform ranks) | 0.9863 | – | 19.21 |
| Qwen3-4B-Base | `configs/qwen3_4b_best.json` | same recipe, refresh/9, measured × ramp 0.6 ranks (probe 0.6/1.0/1.4; jobs 5663775 → 5666862 from `26d8fd4`); zero-shot mean 0.463 | 0.9864 | 13.80 | 14.29 |
| Qwen3-4B-Base | commit `26d8fd4`, `configs/qwen3_4b_kl_ra_both.json` | KL factors, p = ½, fresh R, refresh/9, **hand-table ranks (ramp 0.6 + type weights), parity**; recorded best, code removed | 0.9864 | **13.55** | 14.29 |
| Qwen3-8B-Base | `configs/qwen3_8b_best.json` | same recipe as 4B (refresh/9, probe 0.6/1.0/1.4) plus the efficiency fixes and `tune_epoch_weights: type`; 2 scales, no KD middle scale (job 5721334, h200 on `lgpus`, running 2026-09-15) | – | pending | – |
| Qwen3-14B-Base | `configs/qwen3_14b_best.json` | same as 8B with refresh/10 (40 blocks) and an 80 GB GPU factor budget (job id in `results.md`, second h200 on `lgpus`, launched 2026-09-15) | – | pending | – |

Every recipe keeps the paper's protocol: 128 × 2048 WikiText-2 calibration samples, seed 0, 2 scales, 8/8/8 epochs,
scale-only KD, and the same total bits as the uniform rank rule (parity).

Recipe additions of 2026-09-15 (branch `spectral-projection-screen`, `results.md` "Efficiency fixes, tuning-budget
screen and the KD-only middle scale"): `tune_epoch_weights: type` (q/k 2 epochs, v/o 4, gate 6, up/down 8 per
stage: −28 % block time at +0.01 block-3 PPL on the 0.6B screen) and `model_kd_mid_scale: true` (a per-rank
middle scale initialised to ones and trained only by KD: −0.09 PPL for +0.4 % bits). Plus the lossless efficiency
fixes (fp32 Sylvester eigh, gated per-block eval, probe eigen cache, MLP-only tuning forward, leaner calibration).
Not adopted after measurement: the spike-plus-flat / structured ADMM (5-8 % of 4B ADMM time, 1-2 % of block time),
plateau stopping (dominated by the type table), measured epoch weights (no better than the table), non-factorized
retuning per input group (+0.28 PPL: every round matters).

## Three things to keep

### 1. Where the bits go matters more than how well each layer is fit

Uniform rank per layer was the single largest inefficiency in the pipeline. Moving bits toward the late blocks and
toward the MLP down/up projections, at identical total bits, gave −2.2 PPL at 0.6B (25.48 → 23.25) and −0.56 at 4B
(14.11 → 13.55), more than the curvature work delivered on top of the paper. The evidence was already in the logs:

- the per-block held-out perplexity trajectory (at 4B, half of the pre-KD damage came from the last 8 of 36 blocks);
- the relative block-loss jump each layer's binarisation causes (median over 27 blocks: down +44 %, up +32 %,
  gate +23 %, o/v +10 %, k/q ≈ +2 %).

Judge any future change against the allocated ranks, not against uniform ranks.

### 2. Measure sensitivity, but keep a depth prior

The calibration-time probe (`rank_sensitivity: admm`: short ADMM at 3 candidate ranks per layer, Gauss–Newton weight
error with the layer's Fisher factors, power-law fit, bits spent by marginal predicted loss per bit) reproduces the
within-block ranking of layers well (Spearman 0.8–0.96 against the logged block-loss jumps) with no tuned knob. On
its own it under-values the last blocks, because a Fisher-weighted weight error does not see how little downstream
capacity is left to absorb late errors: measured alone 23.75, hand table 23.25, **measured × ramp 0.6 = 22.96**.
Local curvature proxies rank layers of the same depth well and trade off across depth poorly; the depth dimension
needs an end-to-end signal or a prior. At 4B the order flips: measured × ramp 0.6 gives 13.80 against the hand
table's 13.55 (zero-shot mean 0.463 vs 0.450), so the depth prior alone does not recover what the hand table's type
weights (down/up/gate over q/k/v) bought at that width; both are single runs (seed-1 twins pending).

### 3. Beyond that, single runs are noise: spend runs on seeds, not knobs

Single-run spread at 0.6B is about ±0.5 PPL. Everything below landed inside it and is **not** worth re-running:

| tried | result | where |
|---|---|---|
| full 1.0 bpw budget instead of parity (spend the 32-multiple rounding remainder) | ±0.2, sign flips between settings | results.md, follow-ups table |
| rank ceiling 2 × min(in, out) at ramp 0.6 | 23.17 vs 23.25 | same |
| steeper ramp (1.0) | worse capped (25.94), equal uncapped (23.68) | same |
| tail-block logit objective on top of allocated ranks | 23.82 vs 23.25 (−0.9 on uniform ranks, gone once late blocks have rank) | same |
| KD: feature distillation, trainable norm weights, best-epoch selection (4B) | all within ±0.03 of 14.106 | KD-stage screen |
| factor tuning: latent row normalisation, lower/higher binary lr | 14.89–15.07 vs 14.78–14.88 control | factor-tuning screen |
| dense (Mahalanobis) block loss, spike-plus-flat factors, 3 scales, latent KD | earlier branches, all null or negative | results.md |
| spike-plus-flat projection of the factor *or of its inverse* (top-64 / bottom-64 / both exact, AM / HM / GM middle), shrunk and unshrunk | every projection worse at every block boundary: block 3 +0.10 … +0.34 vs control 14.15 (inverse and unshrunk two-sided outside noise); `calib_shrinkage` 0 vs 0.2 identical (14.14 vs 14.15) | results.md, 2026-09-14 projection screen |

The KD stage on a fixed pre-KD model is deterministic to ~0.02, so it is the one place cheap screening is trustworthy.
The seed-1 twins (`configs/*_best_seed1.json`) exist to turn the two single-run bests into two-seed numbers; the 4B
zero-shot mean dipped to 0.450 from 0.463 while perplexity improved, which is the kind of thing only a second seed
settles.

## Operational notes

- A full 4B run with allocated ranks no longer fits one 4 h job (late blocks with rank up to 2560 take 450–490 s);
  it resumes from per-block checkpoints, but **any commit touching the `blocks` source group between jobs invalidates
  them**. Freeze the cluster checkout for the duration, or pin a separate checkout (`git worktree add --detach`,
  symlink `cache/` and `checkpoints/`, `uv sync`, sbatch a copy of `.objob/job.sh`).
- The measured probe at 4B runs at calibration time and is cached under its own key, so a 4B measured run spans two
  to three jobs.
- The ADMM fast path (branch `admm-fast-sylvester`, inexact Sylvester solve) reproduced the 1.7B recipe losslessly
  (16.67 vs 16.72) and is the natural way to buy back the time the larger late ranks cost.
