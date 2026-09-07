"""Pytest configuration: make the ``src`` layout importable without installing the package."""

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _autograd_globally_enabled():
    """Make sure autograd is enabled process-wide for every test.

    When the whole suite runs in one process, autograd is found globally disabled before the first test's
    setup (each file passes on its own, and importing the test modules directly does not reproduce it), which
    broke every test that back-propagates (the Kronecker calibration tests) on ``main`` as well.
    """
    torch.set_grad_enabled(True)
    yield
    torch.set_grad_enabled(True)
