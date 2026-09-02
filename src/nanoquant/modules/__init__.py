# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""NanoQuant modules package.

Submodules are imported lazily (PEP 562) so that ``quant_config`` can be used without
importing the Hugging Face stack.
"""

import importlib

_EXPORTS = {
    "NanoQuantConfigDataclass": ".hub",
    "NanoQuantModel": ".hub",
    "NanoQuantLinear": ".linear",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        module = importlib.import_module(_EXPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
