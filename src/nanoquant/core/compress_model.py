# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
import time

import torch
from tqdm import trange

from ..core.compress_block import (
    block_importance,
    evaluate_block_loss,
    factor_drift,
    factorize_and_replace,
    format_drift,
    fresh_input_factor,
    shared_input_groups,
    tune_fact,
    tune_nonfact,
)
from ..modules.linear import NanoQuantLinear
from ..optimi import AdamW
from ..utils.cache import ArtifactCache, chain_keys, chain_root, kd_key, teacher_key
from ..utils.eval_utils import evaluate_ppl_after_block
from ..utils.load_utils import cache_inputs_and_kwargs, load_tokenizer
from ..utils.utils import (
    calculate_ranks,
    cleanup_memory,
    find_layers,
    get_decoder_layers,
    get_layers_to_factorize,
    set_seed,
)
from .admm_nq import EigCache
from .importance import (
    collect_stats,
    get_shrunk_stats,
    register_stats,
)
from .kd_loss import kd_kl_loss
from .resume import restore_prefix, save_block_checkpoint, save_progress
from .teacher import TeacherLogits

KD_KIND = "kd"


def refresh_block_curvature(model, dataloader, dev: str, quant_config: dict) -> int:
    """Re-estimate the Kronecker curvature of every not-yet-factorised layer on the *current* model.

    The remaining ``nn.Linear`` layers (the quantised ones are ``NanoQuantLinear`` and are skipped automatically)
    get new ``i_cov``/``o_cov``/``i_norm``/``o_norm`` buffers from ``curvature_refresh_iters`` calibration passes
    warm-started from their present factors, shrunk with ``calib_shrinkage``. The forward and backward passes run
    through the quantised prefix, so the statistics see the activations and gradients that later blocks actually
    receive (docs/admm_block_tuning_curvature.html, section 5).

    Returns
    -------
    int
        Number of layers refreshed.
    """
    layers = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "lm_head" not in n}
    if not layers:
        return 0
    # Detach the current dense factors to the CPU and drop them from the modules: collect_stats moves the whole
    # model to the device, and at 4B the remaining layers' factors alone are ~36 GB. The collector moves the
    # warm-start factors to the device one layer group at a time and register_stats re-attaches the new ones.
    init: dict[str, dict[str, torch.Tensor]] = {"i_cov": {}, "o_cov": {}}
    for n, m in model.named_modules():
        for key in ("i_cov", "o_cov"):
            if hasattr(m, key):
                if n in layers:
                    init[key][n] = getattr(m, key).detach().to("cpu")
                delattr(m, key)
    with torch.enable_grad():
        raw = collect_stats(model, dataloader, dev, strategy=quant_config['calib_strategy'], curvature='kron',
                            fit=quant_config.get('kron_fit', 'frobenius'),
                            nkp_iters=int(quant_config.get('curvature_refresh_iters', 1) or 1),
                            stats_device=quant_config.get('kron_stats_device', 'cpu'),
                            gpu_budget_gb=float(quant_config.get('kron_gpu_budget_gb', 0.0) or 0.0),
                            init_factors=init)
    register_stats(model, get_shrunk_stats(raw, shrinkage=quant_config['calib_shrinkage']))
    del raw, init
    # collect_stats leaves the model on the device in train mode with gradient checkpointing: undo that
    model.cpu()
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = False
    cleanup_memory()
    return len(layers)


@torch.no_grad()
def compress_block_recon(model, fp_model, dataloader, quant_config, cache: ArtifactCache | None = None,
                         sensitivity: dict | None = None):
    """
    Compresses a model using a functional, sequential tune-then-factorize approach.

    ``sensitivity`` is the calibration-time rank-probe artifact (``core.rank_probe.measure_sensitivity``) that the
    measured rank allocation consumes; ``None`` with ``rank_sensitivity != "none"`` falls back to the uniform rule.

    With an enabled ``cache`` the loop is resumable: after every ``checkpoint_every_blocks`` blocks the
    reconstructed blocks and the activations entering the next block are stored under the run's chain
    keys (see :mod:`nanoquant.core.resume`), and a later run with the same chain restores the completed
    prefix and continues. Per-layer ADMM solutions are memoised by :func:`factorize_and_replace`.

    ``quant_config["max_blocks"] > 0`` stops after that many blocks (screening); the checkpoints written
    are those of the full chain, so a later full run resumes from them.
    """
    # set seed
    set_seed(quant_config['seed'])
    # get device
    dev = "cuda"
    # adjust model configs
    model.cpu()
    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = False
    # adjust fp model configs
    fp_model.gradient_checkpointing_disable()
    fp_model.eval()
    fp_model.config.use_cache = False
    # get relevant blocks/layers
    q_blocks = get_decoder_layers(model)
    fp_blocks = get_decoder_layers(fp_model)
    layers_to_factorize = get_layers_to_factorize(model.config.model_type)
    # get admm ranks
    admm_ranks = calculate_ranks(model, layers_to_factorize, quant_config, sensitivity=sensitivity)
    # get kwargs
    original_inputs, kwargs = cache_inputs_and_kwargs(fp_model, dataloader, dev)
    kwargs = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    kwargs['use_cache'] = False
    if 'past_key_value' in kwargs:
        kwargs['past_key_value'] = None
    # get inputs
    compressed_inputs = original_inputs.clone().detach().cpu()

    # resume from the longest checkpointed prefix of this chain
    n_blocks = len(q_blocks)
    max_blocks = int(quant_config.get('max_blocks', 0) or 0)
    stop = min(n_blocks, max_blocks) if max_blocks > 0 else n_blocks
    keys = root = None
    start = 0
    if cache is not None and cache.enabled:
        keys = chain_keys(quant_config, n_blocks)
        root = chain_root(quant_config)
        start, ci, oi = restore_prefix(cache, root, keys, q_blocks)
        if start > 0:
            compressed_inputs = ci.cpu()
            original_inputs = oi.cpu()
            print(f"[resume] restored blocks 0..{start - 1} from cache; continuing at block {start}")
    every = max(1, int(quant_config.get('checkpoint_every_blocks', 1)))
    last_saved = start - 1
    fresh_input = quant_config.get('admm_input_factor', 'calib') == 'fresh'
    diagnostics = bool(quant_config.get('block_diagnostics', False))
    num_samples = quant_config['num_calib_samples']
    refresh_every = int(quant_config.get('curvature_refresh_every', 0) or 0)

    # block reconstruction loop
    for i in trange(start, stop, initial=start, total=stop, desc="Compressing Layers"):
        cleanup_memory()
        if refresh_every > 0 and i > 0 and (i % refresh_every == 0 or i == start):
            # periodic refresh (also right after a resume, whose prefix may have skipped one)
            t_ref = time.time()
            n_ref = refresh_block_curvature(model, dataloader, dev, quant_config)
            print(f"\t\t[refresh] block {i}: re-estimated curvature of {n_ref} remaining layers on the quantised "
                  f"prefix ({time.time() - t_ref:.0f}s)")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_block = time.time()
        # move qblock and fp_block to gpu
        q_block = q_blocks[i].to(dev)
        fp_block = fp_blocks[i].to(dev)
        # Calculate target outputs in batches to minimize CPU-GPU transfers
        with torch.no_grad():
            target_outputs = torch.zeros_like(original_inputs)
            for j in range(quant_config['num_calib_samples']):
                batch_input = original_inputs[j:j + 1].to(dev)
                batch_output = fp_block(batch_input, **kwargs)[0]
                target_outputs[j:j + 1] = batch_output.cpu().detach()
        # get qblock inputs
        tuning_inputs = compressed_inputs.clone().detach()
        # get all linear layers
        sublayers = find_layers(q_block)
        # output-side importance of the block (weights of the reconstruction loss)
        importance = block_importance(sublayers, model.config.hidden_size, dev)
        # move data to GPU
        tuning_inputs = tuning_inputs.to(dev)
        target_outputs = target_outputs.to(dev)
        # layers reading the same activation (q/k/v, gate/up) share one fresh input factor and its eigendecomposition
        fresh_cache: dict = {}
        eig_cache = EigCache() if fresh_input else None
        groups = (shared_input_groups(q_block, sublayers, layers_to_factorize, tuning_inputs[:1], kwargs)
                  if (fresh_input or diagnostics) else {})
        # compress each linear layer
        memo_hits = 0
        for name in layers_to_factorize:
            if name not in sublayers: continue
            # 1/3) tune non-factorized, full-precision weights to absorb quant error
            if quant_config['tune_nonfact']:
                print(f"\t(1/3) Block {i+1}/{n_blocks}, {name} | Tuning Non-Factorized Weights...")
                tune_nonfact(q_block, tuning_inputs, target_outputs, importance, kwargs, quant_config)
                cleanup_memory()
            # 2/3) ADMM to factorize/initialize low-rank binary matrices and scales
            print(f"\t(2/3) Block {i+1}/{n_blocks}, {name} | Initialization via ADMM...")
            curr_rank = admm_ranks.get(f"{i}.{name}")
            layer = sublayers[name]
            input_factor = R_fresh = R_fresh_shrunk = None
            if fresh_input or diagnostics:
                # plain second moment of the inputs that actually reach the layer (quantised prefix, tuned block),
                # reused from an earlier layer of the same shared-input group when its input is unchanged
                R_fresh, R_fresh_shrunk, reused = fresh_input_factor(
                    q_block, layer, name, groups, tuning_inputs, kwargs, num_samples, quant_config['calib_shrinkage'],
                    fresh_cache, eig_cache)
                if reused:
                    print(f"\t\t[fresh R] {name}: reusing the input factor measured for {groups[name]}")
                if diagnostics:
                    # the calibration-time factor is shrunk; compare it with the equally shrunk fresh one
                    stale = getattr(layer, 'i_cov', None)
                    stale = layer.i_norm.diag() if stale is None else stale
                    print("\t\t" + format_drift(factor_drift(stale.to(dev), R_fresh_shrunk),
                                                 title=f"{name} input factor drift"))
                if fresh_input:
                    input_factor = R_fresh_shrunk
            if diagnostics:
                loss_before = evaluate_block_loss(q_block, tuning_inputs, target_outputs, importance, kwargs,
                                                  num_samples)
            nano_linear, final_factor_results = factorize_and_replace(q_block, name, curr_rank, quant_config,
                                                                      cache=cache, input_factor=input_factor,
                                                                      eig_cache=eig_cache)
            memo_hits += int(getattr(final_factor_results, "cache_hit", False))
            if diagnostics:
                loss_after = evaluate_block_loss(q_block, tuning_inputs, target_outputs, importance, kwargs,
                                                 num_samples)
                print(f"\t\tblock loss before -> after ADMM: diag {loss_before:.4e} -> {loss_after:.4e}")
            del final_factor_results, input_factor, R_fresh, R_fresh_shrunk
            cleanup_memory()
            # 3/3) tune low-rank binary and scales
            if quant_config['tune_fact']:
                print(f"\t(3/3) Block {i+1}/{n_blocks}, {name} | Tuning Factorized Weights...")
                tune_fact(q_block, nano_linear, tuning_inputs, target_outputs, importance, kwargs, quant_config)
                cleanup_memory()
            cleanup_memory()
        fresh_cache.clear()
        if eig_cache is not None:
            eig_cache.clear()
        if cache is not None and cache.enabled:
            print(f"\t\t[cache] block {i}: ADMM memo hits {memo_hits}/{len(sublayers)}")

        # move fp_blocks[i] to cpu
        fp_blocks[i] = fp_block.cpu()
        # fp_blocks[i+1] input = fp_blocks[i] output
        original_inputs = target_outputs.clone().detach().cpu()

        # use qblock[i] outputs for qblocks[i+1] inputs
        with torch.no_grad():
            for j in range(quant_config['num_calib_samples']):
                batch_input = compressed_inputs[j:j + 1].to(dev)
                batch_output = q_block(batch_input, **kwargs)[0]
                compressed_inputs[j:j + 1] = batch_output.cpu().detach()
        q_blocks[i] = q_block.cpu()

        del q_block, fp_block, target_outputs, importance
        cleanup_memory()

        # checkpoint the reconstructed blocks and the activations entering block i+1
        if keys is not None and ((i + 1) % every == 0 or i == stop - 1):
            for j in range(last_saved + 1, i + 1):
                save_block_checkpoint(cache, keys[j], q_blocks[j])
            save_progress(cache, root, i, compressed_inputs, original_inputs)
            last_saved = i
            print(f"\t\t[resume] checkpointed blocks {start}..{i}")

        if torch.cuda.is_available():
            print(f"\t\tBlock {i}: {time.time() - t_block:.0f}s, peak CUDA memory "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB allocated / "
                  f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved")
        test_ppl = evaluate_ppl_after_block(model, model_name=quant_config['model_id'], dev=dev)
        print(f"\t\tBlock {i}: Test Data PPL        = {test_ppl:.3f}")

    return model


def _kd_parameters(model) -> list:
    """Put every :class:`NanoQuantLinear` into its deployed forward (hardened ``U``/``V``) and collect its scales.

    Returns
    -------
    list
        The trainable scale parameters (``scale_pre``, ``scale_mid``, ``scale_post``).
    """
    scale_params = []
    for module in model.modules():
        if not isinstance(module, NanoQuantLinear):
            continue
        module.do_train = False
        module._binarized = True
        for name, param in module.named_parameters():
            if 'scale' in name:
                param.requires_grad = True
                scale_params.append(param)
    return scale_params


def _finish_kd(model) -> None:
    """Leave KD with every factorised layer in its deployed forward mode."""
    for module in model.modules():
        if isinstance(module, NanoQuantLinear):
            module.do_train = False
            module._binarized = True


def compress_model_recon(model, fp_model, dataloader, quant_config, dev="cuda", cache: ArtifactCache | None = None):
    """
    Use knowledge distillation to globally tune the scales of the factorised layers (binaries frozen).

    Teacher logits come from :class:`TeacherLogits` in the mode selected by ``model_kd_teacher``
    (``"ram"`` legacy host cache, ``"disk"`` memmap in the artifact cache, ``"online"`` recompute).
    With an enabled ``cache`` the tuned parameters, optimizer/scheduler state and RNG states are
    checkpointed after every epoch and restored on the next run with the same KD key.
    ``model_kd_eval_every_epoch`` evaluates held-out perplexity after every epoch.
    """
    eval_every_epoch = bool(quant_config.get("model_kd_eval_every_epoch", False))

    # set seed
    set_seed(quant_config['seed'])
    # load tokenizer
    tokenizer = load_tokenizer(quant_config['model_id'])

    model.cpu()

    # Prepare data indices and pre-load
    data_indices = list(range(len(dataloader)))
    dataloader = dataloader.to(device=dev, non_blocking=True)
    samples = [dataloader[idx].unsqueeze(0) for idx in data_indices]

    # teacher logits (ram / disk / online)
    teacher_mode = quant_config.get('model_kd_teacher', 'ram')
    use_cache = cache is not None and cache.enabled
    fp_model.eval()
    teacher = TeacherLogits(teacher_mode, fp_model, samples, dev, cache=cache if use_cache else None,
                            key=teacher_key(quant_config) if use_cache else None)
    if teacher_mode != "online":
        # the teacher is not needed on the GPU any more
        fp_model.cpu()
        cleanup_memory(verbose=True)

    # Identify Pad Token for Masking
    pad_token_id = -100
    if hasattr(model, "config"):
        model.config.use_cache = False  # Disable KV cache for training
        if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is not None:
            pad_token_id = model.config.pad_token_id

    model.train()
    # Enable Gradient Checkpointing to save VRAM
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "model") and hasattr(model.model, "gradient_checkpointing_enable"):
        model.model.gradient_checkpointing_enable()
    model.cuda()

    params_to_tune = _kd_parameters(model)
    print(f"Total number of scale parameters to tune: {len(params_to_tune)}")
    if not params_to_tune:
        print("No scales found to tune. Returning original model.")
        _finish_kd(model)
        model.eval()
        return model

    optimizer = AdamW([{'params': params_to_tune, 'lr': quant_config['model_kd_lr']}], weight_decay=0)
    epochs = quant_config["model_kd_epochs"]
    total_steps = epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # KD checkpoint / resume
    ck_key = kd_key(quant_config, len(get_decoder_layers(model))) if use_cache else None
    start_epoch = 1
    if ck_key is not None:
        ck = cache.load(KD_KIND, ck_key)
        if ck is not None and ck["epoch"] < epochs:
            for p, s in zip(params_to_tune, ck["params"]):
                p.data.copy_(s.to(p.device))
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            torch.set_rng_state(ck["torch_rng"])
            random.setstate(ck["py_rng"])
            start_epoch = ck["epoch"] + 1
            print(f"[resume] KD restored after epoch {ck['epoch']}; continuing at epoch {start_epoch}")

    # -------------------------------------------
    # 3) KD-tuning loop (student model)
    # -------------------------------------------
    with torch.enable_grad():
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            random.shuffle(data_indices)
            total_train_loss = torch.zeros(1, device=dev)
            t_epoch = time.time()

            for idx in data_indices:
                batch = samples[idx]

                # Mask Generation
                if pad_token_id != -100:
                    mask = (batch != pad_token_id).int().to(dev)
                else:
                    mask = torch.ones_like(batch).int().to(dev)

                # KD Loss (forward KL on the logits)
                student_outputs = model(batch)
                teacher_logits = teacher.get(idx, batch)
                student_logits = student_outputs.logits if hasattr(student_outputs, "logits") else student_outputs
                loss = kd_kl_loss(student_logits, teacher_logits, mask)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.detach()

            avg_train = total_train_loss / len(dataloader)
            print(f"Epoch {epoch} - Loss: {avg_train.item():.4f} ({time.time() - t_epoch:.0f}s)")
            if eval_every_epoch:
                model.eval()
                with torch.no_grad():
                    ppl = evaluate_ppl_after_block(model, model_name=quant_config['model_id'], dev=dev)
                print(f"Epoch {epoch} - Test Data PPL = {ppl:.3f}")
                model.train()

            if ck_key is not None:
                cache.save(KD_KIND, ck_key, {
                    "epoch": epoch,
                    "params": [p.detach().cpu().clone() for p in params_to_tune],
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "py_rng": random.getstate(),
                })

    # -------------------------------------------
    # 4) Cleanup
    # -------------------------------------------
    del params_to_tune, optimizer, scheduler, dataloader, samples, teacher, tokenizer
    cleanup_memory(verbose=True)
    _finish_kd(model)

    model.eval()
    return model
