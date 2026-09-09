# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
import time

import torch
import torch.nn.functional as F
from tqdm import trange

from ..core.compress_block import (
    block_curvature,
    evaluate_block_loss,
    factor_drift,
    factorize_and_replace,
    format_drift,
    input_second_moment,
    mahalanobis_weight_error,
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
from .curvature import format_spectrum
from .importance import (
    PLAIN_COV_KEY,
    collect_stats,
    get_shrunk_stats,
    register_stats,
    shrink_toward_identity,
)
from .latent import (
    format_flip_stats,
    format_margin_stats,
    latent_flip_stats,
    latent_margin_stats,
    normalize_latents,
)
from .resume import restore_prefix, save_block_checkpoint, save_progress
from .teacher import TeacherLogits

KD_KIND = "kd"


def refresh_block_curvature(model, dataloader, dev: str, quant_config: dict) -> int:
    """Re-estimate the Kronecker curvature of every not-yet-factorised layer on the *current* model.

    The remaining ``nn.Linear`` layers (the quantised ones are ``NanoQuantLinear`` and are skipped automatically)
    get new ``i_cov``/``o_cov``/``i_norm``/``o_norm``/``o_cov_plain`` buffers from ``curvature_refresh_iters``
    calibration passes warm-started from their present factors, shrunk with ``calib_shrinkage``. The forward
    and backward passes run through the quantised prefix, so the statistics see the activations and gradients
    that later blocks actually receive (docs/admm_block_tuning_curvature.html, section 5).

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
        if hasattr(m, PLAIN_COV_KEY):
            delattr(m, PLAIN_COV_KEY)
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


def feature_loss(student_hidden, teacher_hidden, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Relative squared error of the residual stream after every block, averaged over blocks.

    Parameters
    ----------
    student_hidden, teacher_hidden : sequence of torch.Tensor
        ``output_hidden_states`` tuples ``(embedding output, block 1, ..., block N)``; the embedding entry is
        skipped because it is identical for student and teacher.
    mask : torch.Tensor
        ``(1, seqlen)`` token mask.

    Returns
    -------
    torch.Tensor
        ``mean_b sum_t ||s_bt - t_bt||^2 / sum_t ||t_bt||^2`` over masked tokens.
    """
    m = mask.to(torch.float32).unsqueeze(-1)
    total = None
    n = 0
    for s, t in zip(student_hidden[1:], teacher_hidden[1:]):
        t32 = t.float()
        num = ((s.float() - t32).square() * m).sum()
        den = (t32.square() * m).sum().clamp_min(eps)
        total = num / den if total is None else total + num / den
        n += 1
    return total / max(n, 1)
# layers whose output is added straight to the residual stream (their weight error maps 1:1 onto the block error)
BLOCK_OUTPUT_LAYERS = ("mlp.down_proj", "fc2")


@torch.no_grad()
def compress_block_recon(model, fp_model, dataloader, quant_config, cache: ArtifactCache | None = None):
    """
    Compresses a model using a functional, sequential tune-then-factorize approach.

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
        # output-side curvature of the block (weights of the reconstruction losses)
        curvature = block_curvature(sublayers, model.config.hidden_size, quant_config, dev)
        if curvature.summary:
            print("\t\t" + format_spectrum(curvature.summary, title=f"block {i} curvature"))
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
                tune_nonfact(q_block, tuning_inputs, target_outputs, curvature, kwargs, quant_config)
                cleanup_memory()
            # 2/3) ADMM to factorize/initialize low-rank binary matrices and scales
            print(f"\t(2/3) Block {i+1}/{n_blocks}, {name} | Initialization via ADMM...")
            curr_rank = admm_ranks.get(f"{i}.{name}")
            layer = sublayers[name]
            input_factor = R_fresh = R_fresh_shrunk = None
            if fresh_input or diagnostics:
                # plain second moment of the inputs that actually reach the layer (quantised prefix, tuned block)
                R_fresh = input_second_moment(q_block, layer, tuning_inputs, kwargs, num_samples)
                R_fresh_shrunk = shrink_toward_identity(R_fresh, quant_config['calib_shrinkage'])
                if diagnostics:
                    # the calibration-time factor is shrunk; compare it with the equally shrunk fresh one
                    stale = getattr(layer, 'i_cov', None)
                    stale = layer.i_norm.diag() if stale is None else stale
                    print("\t\t" + format_drift(factor_drift(stale.to(dev), R_fresh_shrunk),
                                                 title=f"{name} input factor drift"))
                if fresh_input:
                    input_factor = R_fresh_shrunk
            if diagnostics:
                w_before = layer.weight.detach().clone()
                loss_before = evaluate_block_loss(q_block, tuning_inputs, target_outputs, curvature, kwargs, num_samples)
            nano_linear, final_factor_results = factorize_and_replace(q_block, name, curr_rank, quant_config,
                                                                      cache=cache, input_factor=input_factor)
            memo_hits += int(getattr(final_factor_results, "cache_hit", False))
            if diagnostics:
                # eq. (9) of the design note: for down_proj the dense block-loss increase of the ADMM solution equals
                # tr(L_blk dW R_fresh dW^T) with the *summed* fresh second moment (exact if tune_nonfact converged)
                loss_after = evaluate_block_loss(q_block, tuning_inputs, target_outputs, curvature, kwargs, num_samples)
                W_final = final_factor_results.W_final.to(w_before.device)
                numel = target_outputs.numel()
                tokens = tuning_inputs.shape[0] * tuning_inputs.shape[1]
                msg = (f"\t\tblock loss before -> after ADMM: diag {loss_before[0]:.4e} -> {loss_after[0]:.4e}")
                if loss_before[1] is not None:
                    msg += (f" | dense {loss_before[1]:.4e} -> {loss_after[1]:.4e} "
                            f"(delta {loss_after[1] - loss_before[1]:.4e}")
                    if name in BLOCK_OUTPUT_LAYERS and w_before.shape[0] == curvature.dense.shape[0]:
                        # only the block-output layer writes straight to the residual stream (eq. 9 exact)
                        gn = mahalanobis_weight_error(w_before, W_final, curvature.dense, R_fresh) * tokens / numel
                        msg += f", Gauss-Newton prediction with fresh R {gn:.4e}"
                    msg += ")"
                print(msg)
                del w_before, W_final
            del final_factor_results, input_factor, R_fresh, R_fresh_shrunk
            cleanup_memory()
            # 3/3) tune low-rank binary and scales
            if quant_config['tune_fact']:
                print(f"\t(3/3) Block {i+1}/{n_blocks}, {name} | Tuning Factorized Weights...")
                tune_fact(q_block, nano_linear, tuning_inputs, target_outputs, curvature, kwargs, quant_config)
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

        del q_block, fp_block, target_outputs, curvature
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


def _kd_parameters(model, kd_mode: str) -> tuple[list, list]:
    """Put every :class:`NanoQuantLinear` into the forward mode of ``kd_mode`` and collect its trainable parameters.

    ``"scales"`` evaluates the hardened ``U``/``V`` (the deployed model) and trains the scales only;
    ``"scales_latent"`` switches to the STE forward on the latents and trains scales and latents.

    Returns
    -------
    tuple of list
        ``(scale_params, latent_params)``.
    """
    scale_params, latent_params = [], []
    for module in model.modules():
        if not isinstance(module, NanoQuantLinear):
            continue
        module.do_train = kd_mode == "scales_latent"
        module._binarized = kd_mode != "scales_latent"
        if kd_mode == "scales_latent" and not module.has_latent:
            raise ValueError("model_kd_mode='scales_latent' requires retained latent factors (retain_latent=true)")
        for name, param in module.named_parameters():
            if 'scale' in name:
                param.requires_grad = True
                scale_params.append(param)
            elif kd_mode == "scales_latent" and "latent" in name:
                param.requires_grad = True
                latent_params.append(param)
    return scale_params, latent_params


def _finish_kd(model, kd_mode: str) -> None:
    """Leave KD: harden latents (``scales_latent``) or drop retained ones (``scales``); deployed forward for all."""
    for module in model.modules():
        if isinstance(module, NanoQuantLinear):
            if kd_mode == "scales_latent":
                module.finalize()
            else:
                module.drop_latent()
            module.do_train = False
            module._binarized = True


def compress_model_recon(model, fp_model, dataloader, quant_config, dev="cuda", cache: ArtifactCache | None = None):
    """
    Use knowledge distillation to globally tune scales (and, with ``model_kd_mode="scales_latent"``, the
    latent binary factors through the straight-through estimator).

    Teacher logits come from :class:`TeacherLogits` in the mode selected by ``model_kd_teacher``
    (``"ram"`` legacy host cache, ``"disk"`` memmap in the artifact cache, ``"online"`` recompute).
    With an enabled ``cache`` the tuned parameters, optimizer/scheduler state and RNG states are
    checkpointed after every epoch and restored on the next run with the same KD key.

    Latent mode logs the sign-margin distribution of the latents at the start and the fraction of flipped
    bits after every epoch; ``model_kd_eval_every_epoch`` additionally evaluates held-out perplexity per
    epoch. The returned model is always hardened (no latents).
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
    latent_mode = kd_mode == "scales_latent"
    eval_every_epoch = bool(quant_config.get("model_kd_eval_every_epoch", False))
    feat_w = float(quant_config.get("model_kd_feature_weight", 0.0) or 0.0)

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
    if feat_w > 0 and teacher_mode != "online":
        raise ValueError("model_kd_feature_weight > 0 requires model_kd_teacher='online' (teacher hidden states)")
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

    scale_params, latent_params = _kd_parameters(model, kd_mode)
    params_to_tune = scale_params + latent_params
    print(f"Total number of scale parameters to tune: {len(scale_params)}"
          + (f", latent parameters: {len(latent_params)}" if latent_mode else ""))
    if not params_to_tune:
        print("No scales found to tune. Returning original model.")
        _finish_kd(model, kd_mode)
        model.eval()
        return model

    if latent_mode:
        if quant_config.get("model_kd_latent_normalize", False):
            normalize_latents(model)
            print("[latent] rows rescaled to unit mean magnitude (forward unchanged)")
        print(format_margin_stats(latent_margin_stats(model), title="latent margins at KD start"))
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
            total_kl = torch.zeros(1, device=dev)
            total_feat = torch.zeros(1, device=dev)
            t_epoch = time.time()

            for idx in data_indices:
                batch = samples[idx]

                # Mask Generation
                if pad_token_id != -100:
                    mask = (batch != pad_token_id).int().to(dev)
                else:
                    mask = torch.ones_like(batch).int().to(dev)

                # KD Loss (logit KL, optionally plus residual-stream feature distillation)
                if feat_w > 0:
                    student_outputs = model(batch, output_hidden_states=True)
                    teacher_logits, teacher_hidden = teacher.get(idx, batch, hidden=True)
                else:
                    student_outputs = model(batch)
                    teacher_logits = teacher.get(idx, batch)
                student_logits = student_outputs.logits if hasattr(student_outputs, "logits") else student_outputs

                kl = kl_loss_fn(student_logits, teacher_logits, mask)
                loss = kl
                if feat_w > 0:
                    feat = feature_loss(student_outputs.hidden_states, teacher_hidden, mask)
                    loss = kl + feat_w * feat
                    total_feat += feat.detach()
                    del teacher_hidden
                total_kl += kl.detach()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1

                total_train_loss += loss.detach()

            avg_train = total_train_loss / len(dataloader)
            msg = f"Epoch {epoch} - Loss: {avg_train.item():.4f}"
            if feat_w > 0:
                msg += (f" (KL {(total_kl / len(dataloader)).item():.4f}, "
                        f"feature {(total_feat / len(dataloader)).item():.4e} x {feat_w:g})")
            print(msg + f" ({time.time() - t_epoch:.0f}s)")
            if latent_mode:
                print(format_flip_stats(latent_flip_stats(model), title=f"latent flips after epoch {epoch}"))
            if eval_every_epoch:
                model.eval()
                with torch.no_grad():
                    ppl = evaluate_ppl_after_block(model, model_name=quant_config['model_id'], dev=dev)
                model.train()
                print(f"Epoch {epoch} - Test Data PPL = {ppl:.3f}")

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
    del params_to_tune, scale_params, latent_params, optimizer, scheduler, dataloader, samples, teacher
    cleanup_memory(verbose=True)

    if latent_mode:
        print(format_flip_stats(latent_flip_stats(model), title="latent flips at KD end"))
    _finish_kd(model, kd_mode)

    model.eval()
    return model
