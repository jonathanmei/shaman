# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Single entry point of the quantisation pipeline (calibration → block reconstruction → KD).

Shared by ``modules.hub.NanoQuantModel.quantize_model`` and ``modules.auto_model.AutoNQModel``; wires the
stage-level artifact cache so that repeated or interrupted runs reuse whatever is still valid.
"""

from __future__ import annotations

import torch

from ..utils.cache import ArtifactCache, chain_keys, stats_key
from ..utils.data_utils import get_calib_loader, prepare_dataset
from ..utils.load_utils import load_compressed_model, load_model, load_tokenizer
from ..utils.utils import cleanup_memory, get_decoder_layers, has_mid_scale
from .compress_model import compress_block_recon, compress_model_recon
from .importance import collect_stats, get_shrunk_stats, register_stats
from .resume import compressed_state_dict

PRE_KD_KIND = "model"


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
    }


def run_quantization_pipeline(model_id: str, quant_config: dict, dev: str = "cuda") -> torch.nn.Module:
    """Quantise ``model_id`` according to ``quant_config`` and return the quantised model.

    Stages and their cache behaviour (all governed by ``quant_config['cache_dir']``; empty disables):

    1. calibration statistics – ``stats`` artifact keyed by the calibration settings only;
    2. block-wise reconstruction – per-block checkpoints and per-layer ADMM memo (see
       :mod:`nanoquant.core.resume` and :func:`nanoquant.core.compress_block.factorize_and_replace`);
       the fully reconstructed pre-KD model is stored as a ``model`` artifact;
    3. model-level KD – per-epoch checkpoints; skipped when ``tune_model`` is false.

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
    cache = ArtifactCache(quant_config.get("cache_dir", ""))
    device_map = quant_config.get('device_map', 'cpu')

    fp_model = load_model(model_id, quant_config['seqlen'], device_map=device_map)
    data = prepare_dataset(model_id, quant_config)
    tokenizer = load_tokenizer(model_id)
    dataloader = get_calib_loader(data, tokenizer, quant_config['num_calib_samples'], quant_config['seed'],
                                  quant_config['seqlen'])
    n_blocks = len(get_decoder_layers(fp_model))

    pre_kd_key = chain_keys(quant_config, n_blocks)[-1]
    if cache.exists(PRE_KD_KIND, pre_kd_key):
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
        if cache.enabled:
            torch.save(compressed_state_dict(model), cache.path(PRE_KD_KIND, pre_kd_key))
            print(f"[cache] saved {PRE_KD_KIND} {pre_kd_key[:12]}")

    # 3) model-level KD (scale-only reconstruction)
    if quant_config.get('tune_model', True):
        model = compress_model_recon(model, fp_model, dataloader, quant_config, dev=dev, cache=cache)
    return model
