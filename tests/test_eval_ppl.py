"""Tests for the perplexity evaluator (torch-only; lm_eval is imported lazily by the module)."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from nanoquant.utils.eval_utils import evaluate_ppl

SEQLEN, VOCAB, NSAMPLES = 8, 16, 3


class _FakeLM(nn.Module):
    """Returns fixed bf16 logits for each window, like a bf16 causal LM would."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("table", logits)  # (nsamples, seqlen, vocab)
        self.config = SimpleNamespace(max_position_embeddings=SEQLEN, model_type="fake")
        self.seqlen = SEQLEN
        self.calls = 0

    def forward(self, batch, use_cache=False):
        out = SimpleNamespace(logits=self.table[self.calls: self.calls + 1])
        self.calls += 1
        return out


def test_evaluate_ppl_computes_the_loss_in_float32():
    torch.manual_seed(0)
    logits = (torch.randn(NSAMPLES, SEQLEN, VOCAB) * 3).to(torch.bfloat16)
    tokens = torch.randint(0, VOCAB, (1, NSAMPLES * SEQLEN))
    model = _FakeLM(logits)

    ppl = evaluate_ppl(model, tokens, "cpu", "fake", None, verbose=False)

    # reference: mean token NLL over all windows, computed in float32 from the same bf16 logits
    nll = 0.0
    for i in range(NSAMPLES):
        window = tokens[:, i * SEQLEN:(i + 1) * SEQLEN]
        nll += F.cross_entropy(logits[i, :-1].float(), window[0, 1:], reduction="sum").item()
    expected = float(torch.exp(torch.tensor(nll / (NSAMPLES * (SEQLEN - 1)))))
    assert abs(ppl - expected) <= 1e-5 * expected
    # the bf16-rounded per-window losses would land on a coarse grid and miss the reference
    bf16_nll = sum(F.cross_entropy(logits[i, :-1], tokens[0, i * SEQLEN + 1:(i + 1) * SEQLEN]).item() * (SEQLEN - 1)
                   for i in range(NSAMPLES))
    assert abs(float(torch.exp(torch.tensor(bf16_nll / (NSAMPLES * (SEQLEN - 1))))) - expected) > 1e-5 * expected
