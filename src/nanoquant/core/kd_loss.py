# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Knowledge-distillation loss of the model-level scale reconstruction stage."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
