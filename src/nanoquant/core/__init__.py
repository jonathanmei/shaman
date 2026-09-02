# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""NanoQuant core quantization algorithms.

Submodules are imported lazily (PEP 562) so that importing a single algorithm module
does not pull in heavy optional dependencies of its siblings.
"""

import importlib

_EXPORTS = {
    "factorize_admm_dbf": ".admm_dbf",
    "factorize_admm_nanoquant": ".admm_nq",
    "factorize_and_replace": ".compress_block",
    "tune_fact": ".compress_block",
    "tune_nonfact": ".compress_block",
    "compress_block_recon": ".compress_model",
    "compress_model_recon": ".compress_model",
    "collect_stats": ".importance",
    "get_shrunk_stats": ".importance",
    "register_stats": ".importance",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        module = importlib.import_module(_EXPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
