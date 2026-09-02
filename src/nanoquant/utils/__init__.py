# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""NanoQuant utilities.

Submodules are imported lazily (PEP 562) so that light-weight helpers can be used
without importing the dataset / evaluation stacks.
"""

import importlib

_EXPORTS = {
    "get_calib_loader": ".data_utils",
    "prepare_dataset": ".data_utils",
    "evaluate_model": ".eval_utils",
    "evaluate_ppl": ".eval_utils",
    "evaluate_ppl_after_block": ".eval_utils",
    "cache_inputs_and_kwargs": ".load_utils",
    "get_compressed_state_dict": ".load_utils",
    "load_compressed_model": ".load_utils",
    "load_model": ".load_utils",
    "load_tokenizer": ".load_utils",
    "calculate_ranks": ".utils",
    "cleanup_memory": ".utils",
    "find_layers": ".utils",
    "get_decoder_layers": ".utils",
    "get_layers_to_factorize": ".utils",
    "has_mid_scale": ".utils",
    "set_seed": ".utils",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        module = importlib.import_module(_EXPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
