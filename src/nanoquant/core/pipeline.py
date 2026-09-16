# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Single entry point of the quantisation pipeline (calibration → rank probe → block reconstruction → KD).

Shared by ``modules.hub.NanoQuantModel.quantize_model`` and ``modules.auto_model.AutoNQModel``; wires the
stage-level artifact cache so that repeated or interrupted runs reuse whatever is still valid.
"""

from __future__ import annotations

import os

import torch

from ..utils.bits import format_accounting, model_accounting, static_accounting
from ..utils.cache import ArtifactCache, atomic_save, chain_keys, probe_key, stats_key
from ..utils.data_utils import get_calib_loader, prepare_dataset
from ..utils.load_utils import load_compressed_model, load_model, load_tokenizer
from ..utils.utils import (
    RANK_BUDGETS,
    RANK_SENSITIVITIES,
    cleanup_memory,
    get_decoder_layers,
    get_layers_to_factorize,
    has_mid_scale,
    parse_probe_ranks,
    parse_type_weights,
    stage_devices,
)
from .compress_block import TUNE_EPOCH_WEIGHT_MODES
from .compress_model import compress_block_recon, compress_model_recon
from .curvature import SpectrumSpec
from .importance import (
    CURVATURE_TYPES,
    KRON_FITS,
    collect_stats,
    get_shrunk_stats,
    register_stats,
)
from .kd_mid_scale import insert_unit_mid_scales
from .rank_probe import PROBE_KIND, measure_sensitivity
from .resume import compressed_state_dict

PRE_KD_KIND = "model"
ADMM_INPUT_FACTORS = ("calib", "fresh")


def stats_sample_count(quant_config: dict) -> int:
    """Number of calibration sequences the curvature statistics (and refreshes) are collected on.

    ``num_stats_samples`` (0 or absent = same as ``num_calib_samples``) may raise the statistics' sample count
    without touching the block-reconstruction and KD stages, whose cost is linear in samples × epochs and whose
    activations live on the GPU. It is never lower than ``num_calib_samples``.

    Parameters
    ----------
    quant_config : dict
        Quantisation configuration.

    Returns
    -------
    int
        Sample count of the statistics loader.
    """
    n_calib = int(quant_config["num_calib_samples"])
    n_stats = int(quant_config.get("num_stats_samples", 0) or 0)
    return max(n_calib, n_stats)


def build_stats_loader(model_id: str, tokenizer, quant_config: dict, dataloader: torch.Tensor) -> torch.Tensor:
    """Calibration loader for the curvature statistics.

    Returns ``dataloader`` itself unless ``num_stats_samples`` exceeds ``num_calib_samples``; then a separate pool
    of that many sequences is generated (same seed and dataset) and every sequence is used exactly once. The
    block/KD loader is left untouched, so those stages see the same tokens as a run without the knob.

    Parameters
    ----------
    model_id : str
        Model id (tokenizer of the dataset preparation).
    tokenizer : PreTrainedTokenizer
        Tokenizer (padding id).
    quant_config : dict
        Quantisation configuration.
    dataloader : torch.Tensor
        The ``(num_calib_samples, seqlen)`` loader of the block and KD stages.

    Returns
    -------
    torch.Tensor
        ``(n, seqlen)`` token ids with ``n = stats_sample_count(quant_config)``.
    """
    n_stats = stats_sample_count(quant_config)
    if n_stats == len(dataloader):
        return dataloader
    cfg = dict(quant_config)
    cfg["num_calib_samples"] = n_stats
    pool = prepare_dataset(model_id, cfg)
    return get_calib_loader(pool, tokenizer, n_stats, quant_config["seed"], quant_config["seqlen"], replace=False)


def validate_config(quant_config: dict) -> None:
    """Reject inconsistent configurations before any (expensive) stage runs.

    Parameters
    ----------
    quant_config : dict
        Quantisation configuration.

    Raises
    ------
    ValueError
        For unknown enum values, a non-uniform rank budget without a measured sensitivity, a curvature refresh
        without Kronecker curvature, or a negative ``max_blocks``.
    """
    curvature = quant_config.get("curvature", "diag")
    if curvature not in CURVATURE_TYPES:
        raise ValueError(f"Unknown curvature: {curvature}")
    budget = quant_config.get("rank_budget", "uniform") or "uniform"
    if budget not in RANK_BUDGETS:
        raise ValueError(f"Unknown rank_budget: {budget}")
    sensitivity = quant_config.get("rank_sensitivity", "none") or "none"
    if sensitivity not in RANK_SENSITIVITIES:
        raise ValueError(f"Unknown rank_sensitivity: {sensitivity}")
    ramp = float(quant_config.get("rank_depth_ramp", 0.0) or 0.0)
    type_weights = parse_type_weights(quant_config.get("rank_type_weights", "") or "")  # raises when malformed
    if budget == "uniform" and (ramp or type_weights or sensitivity != "none"):
        raise ValueError("rank_depth_ramp / rank_type_weights / rank_sensitivity require rank_budget='parity' or "
                         "'full'")
    if int(quant_config.get("num_stats_samples", 0) or 0) < 0:
        raise ValueError("num_stats_samples must be >= 0")
    if budget != "uniform" and sensitivity == "none":
        raise ValueError("rank_budget='parity' / 'full' requires a measured rank_sensitivity ('admm' or 'svd')")
    if sensitivity != "none":
        parse_probe_ranks(quant_config.get("rank_probe_ranks", "0.5,1.0,1.5") or "")  # raises on malformed entries
        if int(quant_config.get("rank_probe_iters", 50) or 0) < 1:
            raise ValueError("rank_probe_iters must be >= 1")
        if sensitivity == "admm" and quant_config.get("admm_type", "nanoquant") != "nanoquant":
            raise ValueError("rank_sensitivity='admm' requires admm_type='nanoquant'")
    if quant_config.get("kron_fit", "frobenius") not in KRON_FITS:
        raise ValueError(f"Unknown kron_fit: {quant_config.get('kron_fit')}")
    if quant_config.get("admm_input_factor", "calib") not in ADMM_INPUT_FACTORS:
        raise ValueError(f"Unknown admm_input_factor: {quant_config.get('admm_input_factor')}")
    SpectrumSpec.from_config(quant_config)  # raises on negative ranks, an unknown flat mean or a non-positive power
    if (quant_config.get("tune_epoch_weights", "none") or "none") not in TUNE_EPOCH_WEIGHT_MODES:
        raise ValueError(f"Unknown tune_epoch_weights: {quant_config.get('tune_epoch_weights')}")
    if not 0.0 < float(quant_config.get("tune_epoch_min_frac", 0.25)) <= 1.0:
        raise ValueError("tune_epoch_min_frac must be in (0, 1]")
    if float(quant_config.get("tune_plateau_tol", 0.0) or 0.0) < 0.0:
        raise ValueError("tune_plateau_tol must be >= 0")
    if int(quant_config.get("max_blocks", 0) or 0) < 0:
        raise ValueError("max_blocks must be >= 0")
    if int(quant_config.get("curvature_refresh_every", 0) or 0) < 0:
        raise ValueError("curvature_refresh_every must be >= 0")
    if int(quant_config.get("curvature_refresh_every", 0) or 0) > 0 and curvature != "kron":
        raise ValueError("curvature_refresh_every > 0 requires curvature='kron'")
    override = quant_config.get("pre_kd_checkpoint") or ""
    if override and not os.path.isfile(override):
        raise ValueError(f"pre_kd_checkpoint does not exist: {override}")


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
        'fit': quant_config.get('kron_fit', 'frobenius'),
        'nkp_iters': quant_config.get('kron_nkp_iters', 3),
        'stats_device': quant_config.get('kron_stats_device', 'cpu') if curvature == 'kron' else None,
        'gpu_budget_gb': float(quant_config.get('kron_gpu_budget_gb', 0.0) or 0.0),
    }


def run_quantization_pipeline(model_id: str, quant_config: dict, dev: str = "cuda") -> torch.nn.Module:
    """Quantise ``model_id`` according to ``quant_config`` and return the quantised model.

    Stages and their cache behaviour (all governed by ``quant_config['cache_dir']``; empty disables):

    1. calibration statistics – ``stats`` artifact keyed by the calibration settings only;
    1b. measured rank sensitivity (``rank_sensitivity != "none"``) – ``rank_probe`` artifact keyed by the
        calibration key and the probe's ADMM inputs;
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
    stats_loader = build_stats_loader(model_id, tokenizer, quant_config, dataloader)
    n_blocks = len(get_decoder_layers(fp_model))
    layers = get_layers_to_factorize(fp_model.config.model_type)
    print(format_accounting(static_accounting(fp_model, layers, quant_config), title="bpw budget"))

    max_blocks = int(quant_config.get("max_blocks", 0) or 0)
    truncated = 0 < max_blocks < n_blocks
    pre_kd_key = chain_keys(quant_config, n_blocks)[-1]
    override = quant_config.get("pre_kd_checkpoint") or ""
    if not truncated and (override or cache.exists(PRE_KD_KIND, pre_kd_key)):
        # Every block-level input is unchanged (or an explicit pre-KD checkpoint was given): reload the
        # reconstructed model and go straight to KD.
        if override:
            print(f"[pre-kd] loading {override} (explicit override; chain key {pre_kd_key[:12]} not checked)")
            path = override
        else:
            print(f"[cache] hit  {PRE_KD_KIND} {pre_kd_key[:12]} (skipping calibration and block reconstruction)")
            path = str(cache.path(PRE_KD_KIND, pre_kd_key))
        model = load_compressed_model(model_name_or_path=model_id, checkpoint_path=path,
                                      seqlen=quant_config['seqlen'], device="cpu",
                                      has_mid_scale=has_mid_scale(quant_config), dtype=torch.bfloat16)
    else:
        model = load_model(model_id, quant_config['seqlen'], device_map=device_map)

        # 1) calibration statistics (diagonal or Kronecker-factored curvature)
        print(f"stats: {len(stats_loader)} calibration samples (block reconstruction / KD: {len(dataloader)})")
        raw_stats = cache.load_or_compute(
            "stats", stats_key(quant_config),
            lambda: collect_stats(model, stats_loader, dev, devices=stage_devices(quant_config, dev),
                                  **collect_stats_kwargs(quant_config)))
        shrunk_stats = get_shrunk_stats(raw_stats, shrinkage=quant_config['calib_shrinkage'])
        model = register_stats(model, shrunk_stats)
        del raw_stats, shrunk_stats
        cleanup_memory()

        # 1b) measured rank sensitivity (short ADMM solves at candidate ranks against the registered curvature)
        sensitivity = None
        if (quant_config.get("rank_sensitivity", "none") or "none") != "none":
            sensitivity = cache.load_or_compute(
                PROBE_KIND, probe_key(quant_config),
                lambda: measure_sensitivity(model, layers, quant_config, dev))
            cleanup_memory()
            print(format_accounting(static_accounting(fp_model, layers, quant_config, sensitivity=sensitivity),
                                    title="bpw budget, measured"))

        # 2) block-wise reconstruction (resumable)
        model = compress_block_recon(model, fp_model, dataloader, quant_config, cache=cache, sensitivity=sensitivity,
                                     stats_dataloader=stats_loader)
        if truncated:
            print(f"[screen] reconstructed the first {max_blocks}/{n_blocks} blocks only: "
                  f"pre-KD model not cached, KD skipped")
        elif cache.enabled:
            atomic_save(compressed_state_dict(model), cache.path(PRE_KD_KIND, pre_kd_key))
            print(f"[cache] saved {PRE_KD_KIND} {pre_kd_key[:12]}")

    # 3) model-level KD (scale reconstruction)
    if quant_config.get('tune_model', True) and not truncated:
        if quant_config.get('model_kd_mid_scale', False):
            # per-rank middle scales, identity-initialised, trained by the scale-only KD (block stage untouched)
            n_mid = insert_unit_mid_scales(model)
            print(f"[kd] inserted a unit middle scale into {n_mid} factorised layers")
        model = compress_model_recon(model, fp_model, dataloader, quant_config, dev=dev, cache=cache)

    print(format_accounting(model_accounting(model), title="bpw actual"))
    return model
