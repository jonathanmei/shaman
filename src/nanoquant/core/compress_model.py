# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from tqdm import trange

from ..core.compress_block import (factorize_and_replace, tune_fact, tune_nonfact)
from ..modules.linear import NanoQuantLinear
from ..optimi import AdamW
from ..utils.cache import ArtifactCache, chain_keys, chain_root, kd_key, teacher_key
from ..utils.eval_utils import evaluate_ppl_after_block
from ..utils.load_utils import cache_inputs_and_kwargs, load_tokenizer
from ..utils.utils import (calculate_ranks, cleanup_memory, find_layers, get_decoder_layers, get_layers_to_factorize,
                           set_seed)
from .resume import restore_prefix, save_block_checkpoint, save_progress
from .teacher import TeacherLogits

KD_KIND = "kd"


@torch.no_grad()
def compress_block_recon(model, fp_model, dataloader, quant_config, cache: ArtifactCache | None = None):
    """
    Compresses a model using a functional, sequential tune-then-factorize approach.

    With an enabled ``cache`` the loop is resumable: after every ``checkpoint_every_blocks`` blocks the
    reconstructed blocks and the activations entering the next block are stored under the run's chain
    keys (see :mod:`nanoquant.core.resume`), and a later run with the same chain restores the completed
    prefix and continues. Per-layer ADMM solutions are memoised by :func:`factorize_and_replace`.
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
    admm_ranks = calculate_ranks(model, layers_to_factorize, quant_config)
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

    # block reconstruction loop
    for i in trange(start, n_blocks, initial=start, total=n_blocks, desc="Compressing Layers"):
        cleanup_memory()
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
        block_loss = quant_config.get("block_loss", "diag")
        if block_loss not in ("diag", "mahalanobis"):
            raise ValueError(f"Unknown block_loss: {block_loss}")
        # get importance. For a dense block loss, the output-side curvature of
        # mlp.down_proj is the appropriate factor; the sampled block errors
        # already include the input-activation distribution.
        # Try to get importance from common layer names, fall back to uniform
        importance_layer = sublayers.get('mlp.down_proj', sublayers.get('fc2', None))
        importance_cov = None
        if importance_layer is None:
            # Fallback to uniform importance if expected layer not found
            importance = torch.ones(model.config.hidden_size, device=dev)
        elif not hasattr(importance_layer, 'o_norm'):
            # Fallback if o_norm attribute missing
            importance = torch.ones(model.config.hidden_size, device=dev)
        else:
            importance = importance_layer.o_norm.to(dev)
            if block_loss == "mahalanobis":
                if not hasattr(importance_layer, 'o_cov'):
                    raise ValueError("block_loss='mahalanobis' requires dense Kron curvature statistics")
                importance_cov = importance_layer.o_cov.to(dev)
                importance_cov = 0.5 * (importance_cov + importance_cov.mT)
        if block_loss == "mahalanobis" and importance_cov is None:
            raise ValueError("block_loss='mahalanobis' requires dense Kron curvature statistics")
        # move data to GPU
        tuning_inputs = tuning_inputs.to(dev)
        target_outputs = target_outputs.to(dev)
        # compress each linear layer
        memo_hits = 0
        for name in layers_to_factorize:
            if name not in sublayers: continue
            # 1/3) tune non-factorized, full-precision weights to absorb quant error
            if quant_config['tune_nonfact']:
                print(f"\t(1/3) Block {i+1}/{n_blocks}, {name} | Tuning Non-Factorized Weights...")
                tune_nonfact(q_block, tuning_inputs, target_outputs, importance, kwargs, quant_config,
                             importance_cov=importance_cov)
                cleanup_memory()
            # 2/3) ADMM to factorize/initialize low-rank binary matrices and scales
            print(f"\t(2/3) Block {i+1}/{n_blocks}, {name} | Initialization via ADMM...")
            curr_rank = admm_ranks.get(f"{i}.{name}")
            nano_linear, final_factor_results = factorize_and_replace(q_block, name, curr_rank, quant_config,
                                                                      cache=cache)
            memo_hits += int(getattr(final_factor_results, "cache_hit", False))
            del final_factor_results
            cleanup_memory()
            # 3/3) tune low-rank binary and scales
            if quant_config['tune_fact']:
                print(f"\t(3/3) Block {i+1}/{n_blocks}, {name} | Tuning Factorized Weights...")
                tune_fact(q_block, nano_linear, tuning_inputs, target_outputs, importance, kwargs, quant_config,
                          importance_cov=importance_cov)
                cleanup_memory()
            cleanup_memory()
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

        del q_block, fp_block, target_outputs
        del importance_cov
        cleanup_memory()

        # checkpoint the reconstructed blocks and the activations entering block i+1
        if keys is not None and ((i + 1) % every == 0 or i == n_blocks - 1):
            for j in range(last_saved + 1, i + 1):
                save_block_checkpoint(cache, keys[j], q_blocks[j])
            save_progress(cache, root, i, compressed_inputs, original_inputs)
            last_saved = i
            print(f"\t\t[resume] checkpointed blocks {start}..{i}")

        test_ppl = evaluate_ppl_after_block(model, model_name=quant_config['model_id'], dev=dev)
        print(f"\t\tBlock {i}: Test Data PPL        = {test_ppl:.3f}")

    return model


def compress_model_recon(model, fp_model, dataloader, quant_config, dev="cuda", cache: ArtifactCache | None = None):
    """
    Use knowledge distillation to globally tune scales.

    Teacher logits come from :class:`TeacherLogits` in the mode selected by ``model_kd_teacher``
    (``"ram"`` legacy host cache, ``"disk"`` memmap in the artifact cache, ``"online"`` recompute).
    With an enabled ``cache`` the scale parameters, optimizer/scheduler state and RNG states are
    checkpointed after every epoch and restored on the next run with the same KD key.
    """
    def kl_loss_fn(student_logits, teacher_logits, mask, temperature: float = 1.0) -> torch.Tensor:
        """
        Standard Forward KL (FKL): KL(Teacher || Student)

        Description:
            - The standard objective for Knowledge Distillation.
            - Has a 'Mean-seeking' property, forcing the student to cover the entire teacher distribution.
            - Can lead to overestimation of low-probability regions (tail), potentially causing hallucinations in LLMs.

        Reference:
            Hinton et al. (2015). Distilling the Knowledge in a Neural Network.
        """
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_logprobs = F.log_softmax(student_logits / temperature, dim=-1)

        inf_mask = torch.isinf(student_logits)
        prod = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
        x = torch.sum(prod, dim=-1).view(-1)

        # Minimize -x (which is CE)
        loss = -torch.sum(x * mask.view(-1), dim=0) / (torch.sum(mask.view(-1), dim=0) + 1e-8)
        return (temperature**2) * loss

    kd_mode = quant_config.get("model_kd_mode", "scales")
    if kd_mode not in ("scales", "scales_latent"):
        raise ValueError(f"Unknown model_kd_mode: {kd_mode}")

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

    params_to_tune = []
    scale_params = []
    latent_params = []
    for module in model.modules():
        if isinstance(module, NanoQuantLinear):
            # Scale-only KD must evaluate the same hardened U/V weights that
            # will be used after KD. Latent KD explicitly enables STE mode.
            module.do_train = kd_mode == "scales_latent"
            if kd_mode == "scales_latent":
                if not hasattr(module, "U_latent") or not hasattr(module, "V_latent"):
                    raise ValueError("model_kd_mode='scales_latent' requires retained latent factors")
                module._binarized = False
            else:
                module._binarized = True
            for name, param in module.named_parameters():
                if 'scale' in name:
                    param.requires_grad = True
                    params_to_tune.append(param)
                    scale_params.append(param)
                elif kd_mode == "scales_latent" and "latent" in name:
                    param.requires_grad = True
                    params_to_tune.append(param)
                    latent_params.append(param)

    print(f"Total number of scale parameters to tune: {len(params_to_tune)}")
    if not params_to_tune:
        print("No scales found to tune. Returning original model.")
        model.eval()
        return model

    if kd_mode == "scales_latent":
        optimizer = AdamW([
            {'params': scale_params, 'lr': quant_config['model_kd_lr']},
            {'params': latent_params, 'lr': quant_config.get('model_kd_latent_lr', 1e-6)},
        ])
    else:
        optimizer = AdamW(params_to_tune, lr=quant_config['model_kd_lr'])
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
        step = 0
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            random.shuffle(data_indices)
            total_train_loss = torch.zeros(1, device=dev)

            for idx in data_indices:
                batch = samples[idx]

                # Mask Generation
                if pad_token_id != -100:
                    mask = (batch != pad_token_id).int().to(dev)
                else:
                    mask = torch.ones_like(batch).int().to(dev)

                # KD Loss
                student_outputs = model(batch)
                student_logits = student_outputs.logits if hasattr(student_outputs, "logits") else student_outputs
                teacher_logits = teacher.get(idx, batch)

                # Pass logits + mask to KD functions
                loss = kl_loss_fn(student_logits, teacher_logits, mask)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1

                total_train_loss += loss.detach()

            avg_train = total_train_loss / len(dataloader)
            print(f"Epoch {epoch} - Loss: {avg_train.item():.4f}")

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
    del params_to_tune, optimizer, scheduler, dataloader, samples, teacher
    cleanup_memory(verbose=True)

    if kd_mode == "scales_latent":
        for module in model.modules():
            if isinstance(module, NanoQuantLinear):
                module.finalize()

    for module in model.modules():
        if isinstance(module, NanoQuantLinear):
            module.do_train = False
            module._binarized = True

    model.eval()
    return model
