# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

def NanoQuantConfig(
    # model id
    model_id: str = "meta-llama/Llama-2-7b-hf",
    # quant precision
    bits: float = 1.0,
    # rank allocation: "uniform" (legacy per-layer rule), "parity" (non-uniform, same total bits as uniform) or
    # "full" (non-uniform, exactly `bits` per weight incl. the rounding remainder); multipliers on the per-layer bit
    # budget: log-ratio last/first block (depth ramp) and per-layer-type weights ("v_proj:1.2,down_proj:1.15")
    rank_budget: str = "uniform",
    rank_depth_ramp: float = 0.0,
    rank_type_weights: str = "",
    # rank ceiling as a multiple of min(in, out) (1.0 = legacy cap; binary factors stay meaningful above it)
    rank_max_ratio: float = 1.0,
    # calib
    seed: int = 0,
    num_calib_samples: int = 128,
    calib_dataset: str = "wikitext2",
    calib_shrinkage: float = 0.4,
    calib_strategy: str = "online",
    block_loss: str = "diag",
    # block-loss curvature conditioning (dense block loss only): condition-number cap by eigenvalue flooring
    # (0 = off), spectral power tempering (1 = off) and mixing with the diagonal (1 = fully dense)
    block_loss_cond_max: float = 0.0,
    block_loss_power: float = 1.0,
    block_loss_mix: float = 1.0,
    # block-loss curvature source: "nkp" (output factor of the Kronecker fit of mlp.down_proj) or "plain"
    # (unweighted, clipped covariance of the block-output gradient collected alongside it)
    block_loss_source: str = "nkp",
    # curvature estimate: "diag" (legacy per-feature second moments) or "kron" (nearest Kronecker product)
    curvature: str = "diag",
    # Kronecker fit: "frobenius" (nearest Kronecker product, Shampoo-like) or "kl" (matrix-normal MLE, KL-Shampoo)
    kron_fit: str = "frobenius",
    kron_nkp_iters: int = 3,
    kron_stats_device: str = "cpu",
    kron_eigh_dtype: str = "float64",
    # >0: accumulate the dense factors on the GPU for groups of layers fitting this budget (one pass per group)
    kron_gpu_budget_gb: float = 0.0,
    seqlen: int = 2048,
    device_map: str = "cpu",
    # stage-level artifact cache / resume ("" disables)
    cache_dir: str = "cache",
    checkpoint_every_blocks: int = 1,
    # >0: reconstruct only the first N decoder blocks (screening); KD and the pre-KD model artifact are skipped
    max_blocks: int = 0,
    # tune_nonfact
    tune_nonfact: bool = True,
    nonfact_lr: float = 1e-4,
    nonfact_batch_size: int = 4,
    nonfact_epochs: int = 8,
    # fact (admm)
    admm_type: str = "nanoquant",
    admm_outer_iters: int = 400,
    admm_inner_iters: int = 5,
    admm_reg: float = 3e-2,
    admm_penalty_scheduler: str = "linear",
    admm_print_steps: bool = False,
    admm_mid_scale: bool = False,
    # input-side curvature handed to ADMM: "calib" (calibration-time factor of the full-precision model) or "fresh"
    # (plain second moment of the inputs actually reaching the layer, measured right before it is binarised)
    admm_input_factor: str = "calib",
    # spectral tempering of ADMM's unit-diagonal dense factors (power < 1 and/or condition-number floor)
    admm_curvature_power: float = 1.0,
    admm_curvature_cond_max: float = 0.0,
    # >0: spike-plus-flat projection of ADMM's dense factors (keep this many eigenpairs, flatten the tail; Pro-KLShampoo)
    admm_curvature_spike_rank: int = 0,
    # >0: every N blocks re-estimate the curvature of the remaining layers on the quantised prefix (kron only)
    curvature_refresh_every: int = 0,
    curvature_refresh_iters: int = 1,
    # log input-factor drift, Mahalanobis weight errors and the block-loss change of every ADMM solution
    block_diagnostics: bool = False,
    # >0: reconstruct the last N blocks against the logits of the FP suffix (forward KL) instead of the block loss;
    # tail_logit_mix < 1 mixes the two (each normalised by its first-step value)
    tail_logit_blocks: int = 0,
    tail_logit_mix: float = 1.0,
    # tune_fact
    tune_fact: bool = True,
    fact_binary_lr: float = 1e-5,
    fact_scale_lr: float = 1e-5,
    fact_bias_lr: float = 1e-5,
    fact_batch_size: int = 1,
    fact_epochs: int = 8,
    # rescale each latent row of the freshly factorised layer to unit mean magnitude before factor tuning
    # (sign-preserving), so that fact_binary_lr means the same flip budget in every layer and row
    fact_latent_normalize: bool = False,
    # keep the continuous latent factors (frozen) after block tuning; required by model_kd_mode="scales_latent"
    retain_latent: bool = False,
    # tune_model
    tune_model: bool = True,
    model_kd_lr: float = 1e-5,
    model_kd_latent_lr: float = 1e-6,
    model_kd_batch_size: int = 1,
    model_kd_epochs: int = 8,
    model_kd_mode: str = "scales",
    # rescale each latent row to unit mean magnitude before KD (sign-preserving) so one lr means one flip budget
    model_kd_latent_normalize: bool = False,
    # evaluate held-out perplexity after every KD epoch
    model_kd_eval_every_epoch: bool = False,
    # weight of the residual-stream feature-distillation term added to the logit KL (0 = off; online teacher)
    model_kd_feature_weight: float = 0.0,
    # also train the weights of every normalisation layer (RMSNorm / LayerNorm) during KD, at model_kd_norm_lr
    model_kd_norm_weights: bool = False,
    model_kd_norm_lr: float = 1e-5,
    # evaluate WikiText-2 *validation* perplexity after every KD epoch and keep the best epoch's parameters
    model_kd_select_best: bool = False,
    # explicit pre-KD checkpoint to load instead of the keyed cache artifact ("" = use the cache)
    pre_kd_checkpoint: str = "",
    # teacher logits for KD: "ram" (legacy host cache), "disk" (memmap in cache_dir), "online" (recompute)
    model_kd_teacher: str = "ram",
) -> dict:
    return {
        # model id
        "model_id": model_id,
        # quant precision
        "bits": bits,
        "rank_budget": rank_budget,
        "rank_depth_ramp": rank_depth_ramp,
        "rank_type_weights": rank_type_weights,
        "rank_max_ratio": rank_max_ratio,
        # calibration
        "seed": seed,
        "num_calib_samples": num_calib_samples,
        "calib_dataset": calib_dataset,
        "calib_shrinkage": calib_shrinkage,
        "calib_strategy": calib_strategy,
        "block_loss": block_loss,
        "block_loss_cond_max": block_loss_cond_max,
        "block_loss_power": block_loss_power,
        "block_loss_mix": block_loss_mix,
        "block_loss_source": block_loss_source,
        "curvature": curvature,
        "kron_fit": kron_fit,
        "kron_nkp_iters": kron_nkp_iters,
        "kron_stats_device": kron_stats_device,
        "kron_eigh_dtype": kron_eigh_dtype,
        "kron_gpu_budget_gb": kron_gpu_budget_gb,
        "seqlen": seqlen,
        "device_map": device_map,
        # cache / resume
        "cache_dir": cache_dir,
        "checkpoint_every_blocks": checkpoint_every_blocks,
        "max_blocks": max_blocks,
        # tune_nonfact
        "tune_nonfact": tune_nonfact,
        "nonfact_lr": nonfact_lr,
        "nonfact_batch_size": nonfact_batch_size,
        "nonfact_epochs": nonfact_epochs,
        # fact (admm)
        "admm_type": admm_type,
        "admm_outer_iters": admm_outer_iters,
        "admm_inner_iters": admm_inner_iters,
        "admm_reg": admm_reg,
        "admm_penalty_scheduler": admm_penalty_scheduler,
        'admm_print_steps': admm_print_steps,
        "admm_mid_scale": admm_mid_scale,
        "admm_input_factor": admm_input_factor,
        "admm_curvature_power": admm_curvature_power,
        "admm_curvature_cond_max": admm_curvature_cond_max,
        "admm_curvature_spike_rank": admm_curvature_spike_rank,
        "curvature_refresh_every": curvature_refresh_every,
        "curvature_refresh_iters": curvature_refresh_iters,
        "block_diagnostics": block_diagnostics,
        "tail_logit_blocks": tail_logit_blocks,
        "tail_logit_mix": tail_logit_mix,
        # tune_fact
        "tune_fact": tune_fact,
        "fact_binary_lr": fact_binary_lr,
        "fact_scale_lr": fact_scale_lr,
        "fact_bias_lr": fact_bias_lr,
        "fact_batch_size": fact_batch_size,
        "fact_epochs": fact_epochs,
        "fact_latent_normalize": fact_latent_normalize,
        "retain_latent": retain_latent,
        # tune_model
        "tune_model": tune_model,
        "model_kd_lr": model_kd_lr,
        "model_kd_latent_lr": model_kd_latent_lr,
        "model_kd_batch_size": model_kd_batch_size,
        "model_kd_epochs": model_kd_epochs,
        "model_kd_mode": model_kd_mode,
        "model_kd_latent_normalize": model_kd_latent_normalize,
        "model_kd_eval_every_epoch": model_kd_eval_every_epoch,
        "model_kd_feature_weight": model_kd_feature_weight,
        "model_kd_norm_weights": model_kd_norm_weights,
        "model_kd_norm_lr": model_kd_norm_lr,
        "model_kd_select_best": model_kd_select_best,
        "pre_kd_checkpoint": pre_kd_checkpoint,
        "model_kd_teacher": model_kd_teacher,
    }
