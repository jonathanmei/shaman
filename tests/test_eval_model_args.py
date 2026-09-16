"""The zero-shot harness must receive a wrapped LM, never the raw model inside ``model_args``.

lm-eval 0.4.x folds a ``model_args`` dict into every task's metadata and deep-copies each task config when it
collects results (``Task.dump_config`` -> ``dataclasses.asdict``). With ``{"pretrained": model}`` in there, the
whole model is cloned on the GPU once per task; the 14B run OOMed on six copies (jobs 5723553, 5733174).
"""

from __future__ import annotations

from unittest import mock

import torch
from torch import nn

from nanoquant.utils import eval_utils


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 2)


def test_zero_shot_passes_wrapped_lm_not_model_args() -> None:
    """``simple_evaluate`` gets an ``HFLM`` instance as ``model`` and no ``model_args`` holding the model."""
    model = _Tiny()
    tokenizer = object()
    fake_lm = object()

    with mock.patch.object(eval_utils, "HFLM", return_value=fake_lm) as hflm, \
            mock.patch.object(eval_utils, "simple_evaluate",
                              return_value={"results": {"boolq": {"acc,none": 0.5}}}) as run:
        results = eval_utils.evaluate_model(model, tokenizer, "boolq", eval_ppl="", batch_size="16")

    hflm.assert_called_once()
    kwargs = hflm.call_args.kwargs
    assert kwargs["pretrained"] is model
    assert kwargs["tokenizer"] is tokenizer
    assert kwargs["batch_size"] == "16"

    run.assert_called_once()
    run_kwargs = run.call_args.kwargs
    assert run_kwargs["model"] is fake_lm
    assert run_kwargs.get("model_args") is None
    assert run_kwargs["tasks"] == ["boolq"]
    assert results["boolq"]["acc,none"] == 0.5
    assert isinstance(next(model.parameters()), torch.Tensor)
