# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Single entry point of the quantisation pipeline (calibration → block reconstruction → KD).

Shared by ``modules.hub.NanoQuantModel.quantize_model`` and ``modules.auto_model.AutoNQModel``; wires the
stage-level artifact cache so that repeated or interrupted runs reuse whatever is still valid.
"""

from __future__ import annotations

import torch

from ..utils.bits import format_accounting, model_accounting, static_accounting
from ..utils.cache import ArtifactCache, atomic_save, chain_keys, stats_key
from ..utils.data_utils import get_calib_loader, prepare_dataset
from ..utils.load_utils import load_compressed_model, load_model, load_tokenizer
from ..utils.utils import cleanup_memory, get_decoder_layers, get_layers_to_factorize, has_mid_scale
from .compress_model import compress_block_recon, compress_model_recon
from .importance import CURVATURE_TYPES, collect_stats, get_shrunk_stats, register_stats
from .latent import drop_latents
from .resume import compressed_state_dict

PRE_KD_KIND = "model"
BLOCK_LOSSES = ("diag", "mahalanobis")
BLOCK_LOSS_SOURCES = ("nkp", "plain")
KD_MODES = ("scales", "scales_latent")
ADMM_INPUT_FACTORS = ("calib", "fresh")


def validate_config(quant_config: dict) -> None:
    """Reject inconsistent configurations before any (expensive) stage runs.

    Parameters
    ----------
    quant_config : dict
        Quantisation configuration.

    Raises
    ------
    ValueError
        For unknown enum values, a dense block loss without Kronecker curvature, latent KD without
        retained latents, or a negative ``max_blocks``.
    """
    curvature = quant_config.get("curvature", "diag")
    if curvature not in CURVATURE_TYPES:
        raise ValueError(f"Unknown curvature: {curvature}")
    block_loss = quant_config.get("block_loss", "diag")
    if block_loss not in BLOCK_LOSSES:
        raise ValueError(f"Unknown block_loss: {block_loss}")
    if block_loss == "mahalanobis" and curvature != "kron":
        raise ValueError("block_loss='mahalanobis' requires curvature='kron' (dense output-side curvature)")
    source = quant_config.get("block_loss_source", "nkp")
    if source not in BLOCK_LOSS_SOURCES:
        raise ValueError(f"Unknown block_loss_source: {source}")
    if source == "plain" and curvature != "kron":
        raise ValueError("block_loss_source='plain' requires curvature='kron' (the plain output-gradient covariance "
                         "is collected during the Kronecker calibration passes)")
    if quant_config.get("admm_input_factor", "calib") not in ADMM_INPUT_FACTORS:
        raise ValueError(f"Unknown admm_input_factor: {quant_config.get('admm_input_factor')}")
    kd_mode = quant_config.get("model_kd_mode", "scales")
    if kd_mode not in KD_MODES:
        raise ValueError(f"Unknown model_kd_mode: {kd_mode}")
    if quant_config.get("tune_model", True) and kd_mode == "scales_latent" \
            and not quant_config.get("retain_latent", False):
        raise ValueError("model_kd_mode='scales_latent' requires retain_latent=true: the latent factors are "
                         "dropped when each layer is hardened after block tuning otherwise")
    if int(quant_config.get("max_blocks", 0) or 0) < 0:
        raise ValueError("max_blocks must be >= 0")


def collect_stats_kwargs(quant_config: dict) -> dict:
    """Keyword arguments for :func:`collect_stats` derived from the quantisation config.

    Parameters
    ----------
    quant_config : dict
        Quantisation configuration (``calib_strategy``, ``curvature``, ``kron_*`` keys; missing keys
        fall back to the legacy diagonal behaviour).

    Returns
    -------
    dict
    """
    curvature = quant_config.get('curvature', 'diag')
    return {
        'strategy': quant_config['calib_strategy'],
        'curvature': curvature,
        'nkp_iters': quant_config.get('kron_nkp_iters', 3),
        'stats_device': quant_config.get('kron_stats_device', 'cpu') if curvature == 'kron' else None,
        'gpu_budget_gb': float(quant_config.get('kron_gpu_budget_gb', 0.0) or 0.0),
    }


def run_quantization_pipeline(model_id: str, quant_config: dict, dev: str = "cuda") -> torch.nn.Module:
    """Quantise ``model_id`` according to ``quant_config`` and return the quantised model.

    Stages and their cache behaviour (all governed by ``quant_config['cache_dir']``; empty disables):

    1. calibration statistics – ``stats`` artifact keyed by the calibration settings only;
    2. block-wise reconstruction – per-block checkpoints and per-layer ADMM memo (see
       :mod:`nanoquant.core.resume` and :func:`nanoquant.core.compress_block.factorize_and_replace`);
       the fully reconstructed pre-KD model is stored as a ``model`` artifact;
    3. model-level KD – per-epoch checkpoints; skipped when ``tune_model`` is false.

    With ``max_blocks > 0`` (screening) only the first blocks are reconstructed: the pre-KD artifact is
    not written (it would masquerade as a full model under the chain's key), KD is skipped and the
    per-block checkpoints stay reusable by a later full run of the same chain.

    The predicted (rank-budget) and actual bits-per-weight accounting are printed.

    Parameters
    ----------
    model_id : str
        Hugging Face model id or local path.
    quant_config : dict
        Quantisation configuration.
    dev : str
        Compute device.

    Returns
    -------
    torch.nn.Module
        The quantised model (on CPU/GPU as left by the last stage).
    """
    validate_config(quant_config)
    cache = ArtifactCache(quant_config.get("cache_dir", ""))
    device_map = quant_config.get('device_map', 'cpu')

    fp_model = load_model(model_id, quant_config['seqlen'], device_map=device_map)
    data = prepare_dataset(model_id, quant_config)
    tokenizer = load_tokenizer(model_id)
    dataloader = get_calib_loader(data, tokenizer, quant_config['num_calib_samples'], quant_config['seed'],
                                  quant_config['seqlen'])
    n_blocks = len(get_decoder_layers(fp_model))
    print(format_accounting(static_accounting(fp_model, get_layers_to_factorize(fp_model.config.model_type),
                                              quant_config), title="bpw budget"))

    max_blocks = int(quant_config.get("max_blocks", 0) or 0)
    truncated = 0 < max_blocks < n_blocks
    pre_kd_key = chain_keys(quant_config, n_blocks)[-1]
    if not truncated and cache.exists(PRE_KD_KIND, pre_kd_key):
        # Every block-level input is unchanged: reload the reconstructed model and go straight to KD.
        print(f"[cache] hit  {PRE_KD_KIND} {pre_kd_key[:12]} (skipping calibration and block reconstruction)")
        model = load_compressed_model(model_name_or_path=model_id,
                                      checkpoint_path=str(cache.path(PRE_KD_KIND, pre_kd_key)),
                                      seqlen=quant_config['seqlen'], device="cpu",
                                      has_mid_scale=has_mid_scale(quant_config), dtype=torch.bfloat16)
    else:
        model = load_model(model_id, quant_config['seqlen'], device_map=device_map)

        # 1) calibration statistics (diagonal or Kronecker-factored curvature)
        raw_stats = cache.load_or_compute(
            "stats", stats_key(quant_config),
            lambda: collect_stats(model, dataloader, dev, **collect_stats_kwargs(quant_config)))
        shrunk_stats = get_shrunk_stats(raw_stats, shrinkage=quant_config['calib_shrinkage'])
        model = register_stats(model, shrunk_stats)
        del raw_stats, shrunk_stats
        cleanup_memory()

        # 2) block-wise reconstruction (resumable)
        model = compress_block_recon(model, fp_model, dataloader, quant_config, cache=cache)
        if truncated:
            print(f"[screen] reconstructed the first {max_blocks}/{n_blocks} blocks only: "
                  f"pre-KD model not cached, KD skipped")
        elif cache.enabled:
            atomic_save(compressed_state_dict(model), cache.path(PRE_KD_KIND, pre_kd_key))
            print(f"[cache] saved {PRE_KD_KIND} {pre_kd_key[:12]}")

    # 3) model-level KD (scale-only or scale + latent reconstruction)
    if quant_config.get('tune_model', True) and not truncated:
        model = compress_model_recon(model, fp_model, dataloader, quant_config, dev=dev, cache=cache)
    else:
        drop_latents(model)

    print(format_accounting(model_accounting(model), title="bpw actual"))
    return model
