# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

def NanoQuantConfig(
    # model id
    model_id: str = "meta-llama/Llama-2-7b-hf",
    # quant precision
    bits: float = 1.0,
    # rank allocation: "uniform" (the paper's per-layer rule) or, with a measured sensitivity, "parity" (same total
    # bits as uniform) / "full" (exactly `bits` per weight incl. the rounding remainder)
    rank_budget: str = "uniform",
    # measured per-layer sensitivity driving the non-uniform allocation: "admm" (short ADMM solves at
    # rank_probe_ranks x uniform rank, rank_probe_iters outer iterations each, curvature-weighted weight error) or
    # "svd" (tail energy of the whitened spectrum); "none" = uniform rule only
    rank_sensitivity: str = "none",
    rank_probe_ranks: str = "0.5,1.0,1.5",
    rank_probe_iters: int = 50,
    # depth prior on the measured curves: log-ratio of the last block's multiplier to the first block's (0 = flat)
    rank_depth_ramp: float = 0.0,
    # per-layer-type prior on the measured curves ("q_proj:0.85,down_proj:1.15"; multiplies the ramp; "" = none)
    rank_type_weights: str = "",
    # calib
    seed: int = 0,
    num_calib_samples: int = 128,
    # calibration samples for the curvature statistics and refreshes only (0 = num_calib_samples)
    num_stats_samples: int = 0,
    calib_dataset: str = "wikitext2",
    calib_shrinkage: float = 0.4,
    calib_strategy: str = "online",
    # curvature estimate: "diag" (legacy per-feature second moments) or "kron" (nearest Kronecker product)
    curvature: str = "diag",
    # Kronecker fit: "frobenius" (nearest Kronecker product, Shampoo-like) or "kl" (matrix-normal MLE, KL-Shampoo)
    kron_fit: str = "frobenius",
    kron_nkp_iters: int = 3,
    kron_stats_device: str = "cpu",
    kron_eigh_dtype: str = "float64",
    # >0: accumulate the dense factors on the GPU for groups of layers fitting this budget (one pass per group)
    kron_gpu_budget_gb: float = 0.0,
    # GPUs for the layer-sharded stages (rank probe, calibration layer groups): 0 = every visible one, 1 = serial
    parallel_devices: int = 0,
    # run the B half of every ADMM iteration on a second GPU (same result as the serial path; needs >= 2 GPUs)
    admm_parallel_sides: bool = False,
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
    # one non-factorized retuning round per shared-input group (q/k/v, gate/up) instead of before every layer
    nonfact_per_group: bool = False,
    # tuning budget per layer: "none" | "type" (fixed table by layer type) | "measured" (rank-probe error at the
    # allocated rank, normalised per block); floor of the multiplier; plateau stop tolerance (0 = off)
    tune_epoch_weights: str = "none",
    tune_epoch_min_frac: float = 0.25,
    tune_plateau_tol: float = 0.0,
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
    # spectral tempering of ADMM's unit-diagonal dense factors: eigenvalues raised to this power (1 = off; 0.5 = the
    # square-root metric, see docs/curvature_tempering_theory.md)
    admm_curvature_power: float = 1.0,
    # two-sided spike-plus-flat projection of the same factors: keep the `spike_rank` largest and `dip_rank` smallest
    # eigenvalues exactly, replace the middle by its arithmetic / geometric / harmonic mean ("am" | "gm" | "hm");
    # spike-only = Pro-KLShampoo's low-rank + identity model of the factor, dip-only + "hm" = the same model of its
    # inverse (both 0 = off)
    admm_curvature_spike_rank: int = 0,
    admm_curvature_dip_rank: int = 0,
    admm_curvature_flat_mean: str = "am",
    # >0: every N blocks re-estimate the curvature of the remaining layers on the quantised prefix (kron only)
    curvature_refresh_every: int = 0,
    curvature_refresh_iters: int = 1,
    # log input-factor drift, Mahalanobis weight errors and the block-loss change of every ADMM solution
    block_diagnostics: bool = False,
    # >0: evaluate WikiText-2 perplexity every N blocks (and after the last); 0 = only when block_diagnostics is on
    block_ppl_every: int = 0,
    # KD only: give every factorised layer without a middle scale a per-rank scale_mid initialised to ones and
    # let the scale-only KD train it (the block stage is untouched)
    model_kd_mid_scale: bool = False,
    # tune_fact
    tune_fact: bool = True,
    fact_binary_lr: float = 1e-5,
    fact_scale_lr: float = 1e-5,
    fact_bias_lr: float = 1e-5,
    fact_batch_size: int = 1,
    fact_epochs: int = 8,
    # keep the best factorized-tuning state (eval after every epoch; ADMM init is a candidate)
    fact_keep_best: bool = False,
    # tune_model (scale-only KD)
    tune_model: bool = True,
    model_kd_lr: float = 1e-5,
    model_kd_batch_size: int = 1,
    model_kd_epochs: int = 8,
    # evaluate held-out perplexity after every KD epoch
    model_kd_eval_every_epoch: bool = False,
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
        "rank_sensitivity": rank_sensitivity,
        "rank_probe_ranks": rank_probe_ranks,
        "rank_probe_iters": rank_probe_iters,
        "rank_depth_ramp": rank_depth_ramp,
        "rank_type_weights": rank_type_weights,
        # calibration
        "seed": seed,
        "num_calib_samples": num_calib_samples,
        "num_stats_samples": num_stats_samples,
        "calib_dataset": calib_dataset,
        "calib_shrinkage": calib_shrinkage,
        "calib_strategy": calib_strategy,
        "curvature": curvature,
        "kron_fit": kron_fit,
        "kron_nkp_iters": kron_nkp_iters,
        "kron_stats_device": kron_stats_device,
        "kron_eigh_dtype": kron_eigh_dtype,
        "kron_gpu_budget_gb": kron_gpu_budget_gb,
        "parallel_devices": parallel_devices,
        "admm_parallel_sides": admm_parallel_sides,
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
        "nonfact_per_group": nonfact_per_group,
        "tune_epoch_weights": tune_epoch_weights,
        "tune_epoch_min_frac": tune_epoch_min_frac,
        "tune_plateau_tol": tune_plateau_tol,
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
        "admm_curvature_spike_rank": admm_curvature_spike_rank,
        "admm_curvature_dip_rank": admm_curvature_dip_rank,
        "admm_curvature_flat_mean": admm_curvature_flat_mean,
        "curvature_refresh_every": curvature_refresh_every,
        "curvature_refresh_iters": curvature_refresh_iters,
        "block_diagnostics": block_diagnostics,
        "block_ppl_every": block_ppl_every,
        "model_kd_mid_scale": model_kd_mid_scale,
        # tune_fact
        "tune_fact": tune_fact,
        "fact_binary_lr": fact_binary_lr,
        "fact_scale_lr": fact_scale_lr,
        "fact_bias_lr": fact_bias_lr,
        "fact_batch_size": fact_batch_size,
        "fact_epochs": fact_epochs,
        "fact_keep_best": fact_keep_best,
        # tune_model
        "tune_model": tune_model,
        "model_kd_lr": model_kd_lr,
        "model_kd_batch_size": model_kd_batch_size,
        "model_kd_epochs": model_kd_epochs,
        "model_kd_eval_every_epoch": model_kd_eval_every_epoch,
        "pre_kd_checkpoint": pre_kd_checkpoint,
        "model_kd_teacher": model_kd_teacher,
    }
