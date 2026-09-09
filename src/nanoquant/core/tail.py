# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Logit-level (suffix-aware) reconstruction objective for the last decoder blocks.

The block reconstruction loss weights residual-stream errors by a diagonal (or dense) output-side curvature. For
the last few blocks that proxy is at its worst: the remaining network is only a handful of full-precision blocks
plus the final norm and the LM head, and the map from the residual stream to the logits is strongly anisotropic
(4B: the last block alone adds 1.2 PPL before KD, one sixth of the total damage). Here the true objective is
cheap: push the student block's output through the *full-precision suffix* and match the teacher's logits with
the same forward KL used by the model-level KD. The teacher logits are the FP suffix applied to the FP block
output on the FP prefix (the block targets), recomputed on the fly.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def kd_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor,
               temperature: float = 1.0) -> torch.Tensor:
    """Forward KL ``KL(teacher || student)`` up to the teacher entropy, i.e. the masked teacher cross-entropy.

    Standard KD objective (Hinton et al., 2015): mean-seeking, forces the student to cover the whole teacher
    distribution. Tokens with ``mask == 0`` are ignored; the result is scaled by ``temperature**2``.

    Parameters
    ----------
    student_logits, teacher_logits : torch.Tensor
        ``(1, seqlen, vocab)``.
    mask : torch.Tensor
        ``(1, seqlen)`` token mask.
    temperature : float
        Softmax temperature.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_logprobs = F.log_softmax(student_logits / temperature, dim=-1)
    inf_mask = torch.isinf(student_logits)
    prod = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod, dim=-1).view(-1)
    loss = -torch.sum(x * mask.view(-1), dim=0) / (torch.sum(mask.view(-1), dim=0) + 1e-8)
    return (temperature ** 2) * loss


class TailLogitObjective:
    """Forward KL between the logits of the FP suffix applied to a student block output and to the block target.

    Parameters
    ----------
    suffix_blocks : sequence of nn.Module
        Full-precision decoder blocks after the block being reconstructed (possibly empty), on ``device``.
    final_norm, lm_head : nn.Module
        Final normalisation and LM head of the full-precision model, on ``device``.
    target_outputs : torch.Tensor
        ``(num_samples, seqlen, hidden)`` FP block outputs on the FP prefix (the reconstruction targets).
    kwargs : dict
        Block forward kwargs (position embeddings, masks) shared by every decoder block.
    device : str or torch.device
        Compute device.
    mix : float
        Weight of the (normalised) KL term in the tuning loss; ``1`` = pure KL, ``< 1`` mixes in the block loss
        (see :func:`nanoquant.core.compress_block._tune_loop`).
    """

    def __init__(self, suffix_blocks: Sequence[nn.Module], final_norm: nn.Module, lm_head: nn.Module,
                 target_outputs: torch.Tensor, kwargs: dict, device="cuda", mix: float = 1.0):
        self.blocks = [b.to(device) for b in suffix_blocks]
        self.norm = final_norm.to(device)
        self.head = lm_head.to(device)
        for module in (*self.blocks, self.norm, self.head):
            for p in module.parameters():
                p.requires_grad = False
        self.targets = target_outputs
        self.kwargs = kwargs
        self.mix = float(mix)
        self._entropy: dict[int, torch.Tensor] = {}

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Logits of the FP suffix (blocks, final norm, LM head) applied to residual-stream states ``hidden``."""
        h = hidden
        for block in self.blocks:
            h = block(h, **self.kwargs)[0]
        return self.head(self.norm(h))

    @torch.no_grad()
    def teacher_logits(self, idx: int) -> torch.Tensor:
        return self.logits(self.targets[idx:idx + 1].to(self.head.weight.device)).detach()

    def _mask(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.ones(logits.shape[:2], dtype=torch.int, device=logits.device)

    def kl(self, student_hidden: torch.Tensor, idx: int) -> torch.Tensor:
        """Teacher cross-entropy of the student logits for calibration sample ``idx`` (differentiable)."""
        teacher = self.teacher_logits(idx)
        return kd_kl_loss(self.logits(student_hidden), teacher, self._mask(teacher))

    @torch.no_grad()
    def entropy(self, idx: int) -> torch.Tensor:
        """Teacher entropy of sample ``idx`` (the minimum of :meth:`kl`), cached per sample."""
        if idx not in self._entropy:
            t = self.teacher_logits(idx)
            self._entropy[idx] = kd_kl_loss(t, t, self._mask(t)).detach()
        return self._entropy[idx]

    def excess_kl(self, student_hidden: torch.Tensor, idx: int) -> torch.Tensor:
        """``kl - entropy``: the proper KL divergence, zero when the student matches the teacher."""
        return self.kl(student_hidden, idx) - self.entropy(idx)
