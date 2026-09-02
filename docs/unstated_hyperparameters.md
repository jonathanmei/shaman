# NanoQuant: hyperparameters not stated in the paper

Reference: arXiv 2602.06694 (ICML 2026), Sec. 3, Algorithm 1, Appendix C ("Implementation Details"), Appendix F.
Purpose: enumerate every knob that influences the paper-faithful pipeline (`configs/qwen3_0p6b_diag_2scale.json`)
but whose value is **not given in the paper**, so that reproduction gaps (e.g. Qwen3-0.6B: paper 27.56 PPL vs. our
29.21 at the same 0.973 bpw) can be attributed. "Current value" is what the released code (and our runs) use.

Algorithm 1 declares the parameter groups explicitly: robust-diagonal parameters $(\tau, \gamma)$, ADMM parameters
$(K, \rho, \lambda, \epsilon)$, optimisation steps $(T_{\mathrm{pre}}, T_{\mathrm{post}}, T_{\mathrm{glob}})$. Of these the
paper gives $\gamma$ (0.2 for Llama/Qwen, 0.6 for Gemma/Rnj-1), $K = 400$ with a linear $\rho$ schedule, and the three
stage epoch counts (8/8/8); $\tau$, $\lambda$, $\epsilon$ and the $\rho$ range are **not** given.

## Stated in the paper (for reference)

| item | paper value |
|---|---|
| calibration data | 128 samples × 2048 tokens, WikiText-2 train, seed 0 |
| shrinkage $\gamma$ | 0.2 (Llama, Qwen), 0.6 (Gemma, Rnj-1) |
| ADMM | linear penalty schedule, 400 steps per weight matrix |
| pre-factorised tuning | lr 1e-4, batch 4, 8 epochs, cosine schedule |
| factorised tuning (latent binaries + scales) | lr 1e-5, batch 1, 8 epochs, cosine |
| global scale reconstruction (KD) | lr 1e-6, batch 1, 8 epochs, cosine |
| scale extraction | magnitude balancing $\eta=\sqrt{\|\hat V\|_F/\|\hat U\|_F}$, then mean absolute value per channel (Eq. 7–8) |
| effective bpw | bits of decoder linear layers / their weight count (Appendix F, Eq. 60); scales 16-bit |
| software | torch 2.6.0, transformers 4.51.3, datasets 4.0.0, lm_eval 0.4.9, CUDA 12.4; gradient checkpointing; cut-cross-entropy loss |
| evaluation | WikiText-2 perplexity; zero-shot BoolQ, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c (lm_eval) |

## Unstated: calibration / curvature

| # | knob | current value (code) | notes |
|---|---|---|---|
| 1 | robust clipping percentile $\tau$ | `PERCENTILE = 0.999` of per-token L2 norms (`core/importance.py`) | Lemma 1 only says entries are clipped at $\tau_{\max}$ |
| 2 | clipping strategy | `calib_strategy="online"`: running robust max with retroactive rescaling of already-accumulated stats; alternatives `two_phase`, `dbf` (none) | which variant produced the paper numbers is not stated |
| 3 | gradient source for $z_{\text{out}}$ | next-token CE on the calibration text (`use_truefisher=False`); true-Fisher sampling exists but is off | paper: "activation- and gradient-based" |
| 4 | statistic normalisation | `mean` of squares over tokens, summed over samples, `/ n_samples`; gradients pre-scaled by `GRAD_SCALE_FACTOR = 1e6` | scale-invariant for ADMM, but sets the magnitude of the importance weights (item 13) |
| 5 | forward-hook double counting | with re-entrant gradient checkpointing the forward hook fires twice per sample, so `i_norm` is 2× `o_norm`'s normalisation | harmless for ADMM (diagonal rescaling), implementation quirk |
| 6 | calibration slicing | random windows of the `"\n\n"`-joined WikiText-2 train split, `random.randint` after `set_seed(0)` | only "128 samples, 2048, seed 0" is stated |

## Unstated: ADMM (LB-ADMM)

| # | knob | current value (code) | notes |
|---|---|---|---|
| 7 | ridge $\lambda$ | `reg = 3e-2`, added to the diagonal **relative to** the mean diagonal of $V^\top V$ (`_admm_solve_step`) | Eq. 5 has $(\rho+\lambda)I$ with absolute $\lambda$; value unstated. `admm_reg` config field is currently not threaded into the call (function default used) |
| 8 | $\epsilon$ | `eps = 1e-12` (clamps on norms / stabiliser) | unstated |
| 9 | $\rho$ range | `linear`: $\rho_t = t/K$, i.e. 0 → 0.9975, never reaching 1; stabiliser uses $\rho \cdot \overline{\mathrm{diag}} + \lambda$ | only "linear schedule" stated |
| 10 | SVID power iterations | `inner_iters = 5` random-start power iterations per projection | unstated |
| 11 | factor initialisation | `torch.randn` for both factors (SVD init removed), seeded by `set_seed(0)` | paper does not state init |
| 12 | mid-dimension normalisation | X-updates use $V$ with unit-norm rows / $U$ with unit-norm columns; final $1/\|U_{:,r}\|$ folded into $U$ | implementation detail, not in Eq. 5 |

## Unstated: rank budget / bit accounting

| # | knob | current value (code) | notes |
|---|---|---|---|
| 13 | rank rounding | rank floored to a multiple of 32, minimum 32 (`calculate_ranks`) | makes "1.0 bpw" ≈ 0.973 bpw for Qwen3-0.6B (q/o 640, k/v 480, gate/up/down 736) |
| 14 | rank formula | `Rank = bits·a·b/(a+b) − 16` (2 scales) | consistent with Eq. 60 but the rounding is not stated |

## Unstated: block reconstruction (TUNEFP / TUNELATENTSTE)

| # | knob | current value (code) | notes |
|---|---|---|---|
| 15 | **schedule inside a block** | code: for each of the 7 layers in order q, v, o, k, gate, up, down: tune remaining FP layers → ADMM this layer → tune latent binaries + scales (7 × 3 stages per block) | **Algorithm 1 runs TUNEFP once, then ADMM for all layers, then TUNELATENTSTE once per block.** The largest structural ambiguity; also changes the effective number of tuning epochs per block (7× the stated 8) |
| 16 | weighted-MSE importance vector | `o_norm` of `mlp.down_proj` (per hidden dimension), unnormalised | paper: "weighted MSE function utilised in previous quantisation works" |
| 17 | optimiser | vendored `optimi.AdamW`, weight decay 0, default betas/eps | paper names only the cosine schedule |
| 18 | cosine floor | `eta_min = 1e-4 × lr` for the block stages, 0 for KD | unstated |
| 19 | update granularity | one sample per step, gradient accumulation to the stated batch (4 or 1); data reshuffled per epoch (`randperm`) | unstated |
| 20 | which FP weights are tuned in TUNEFP | all still-full-precision `nn.Linear` weights of the block (norms, biases untouched) | unstated |
| 21 | which parameters are tuned in TUNELATENTSTE | latent $U, V$ (STE), all scales, biases of every already-factorised layer in the block | unstated |
| 22 | STE | `sign(x)` forward, identity backward; `sign(0) := +1` | standard, unstated |
| 23 | block inputs | student sees activations of the quantised prefix; targets are FP-block outputs on FP-prefix activations | matches "error propagation mitigation" but the exact pairing is not spelled out |

## Unstated: global scale reconstruction (KD)

| # | knob | current value (code) | notes |
|---|---|---|---|
| 24 | loss | forward KL(teacher ‖ student) on logits, temperature 1.0, pad-token mask | direction and temperature unstated |
| 25 | tuned parameters | every parameter with `scale` in its name (`scale_pre`, `scale_mid`, `scale_post`); binaries frozen | consistent with "scale reconstruction" |
| 26 | KD data | the same 128 calibration sequences, one per step, reshuffled per epoch | unstated whether a separate set is used |
| 27 | teacher | full-precision model logits (our `model_kd_teacher` modes are equivalent) | — |

## Unstated: evaluation and seeds

| # | knob | current value (code) | notes |
|---|---|---|---|
| 28 | WikiText-2 perplexity protocol | `"\n\n".join(test)`; non-overlapping 2048-token windows; `nsamples = numel // 2048` | GPTQ-style, standard but unstated |
| 29 | zero-shot | lm_eval 0.4.9, `batch_size="auto"`, `num_fewshot=0`, full task sets | stated except batch size |
| 30 | precision | model in bf16 with SDPA attention; ADMM in fp32 on bf16 weights | partially stated |
| 31 | seeds | `set_seed(0)` before every stage (data, ADMM init, shuffles); CUDA nondeterminism not disabled | only the data seed is stated; block-27 PPL differed by ~2 points between two identical diag runs (35.8 vs 37.8) |

## Not in the paper at all (our extensions, default off)

`curvature=kron` (+ `kron_nkp_iters`, `kron_stats_device`, `kron_eigh_dtype`), `admm_mid_scale`, `cache_dir`,
`checkpoint_every_blocks`, `model_kd_teacher`.

## Most likely contributors to the 27.56 → 29.21 gap (ordered by prior)

1. Item 15 (per-layer vs per-block tuning schedule) if the paper's runs used the Algorithm 1 schedule.
2. Item 31 (seed / GPU nondeterminism): two identical diag runs already differed by ~2 PPL before KD.
3. Items 7–10 ($\lambda$, $\epsilon$, $\rho$ range, inner iterations).
4. Items 1–2 (clipping percentile and strategy).
