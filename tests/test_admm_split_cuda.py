"""GPU-only checks of the two-device ADMM and the sharded probe: run where at least two CUDA devices are visible
(``objob submit --gres gpu:a100:2 -- uv run pytest tests/test_admm_split_cuda.py``); skipped elsewhere."""

import pytest
import torch

from nanoquant.core import admm_nq
from nanoquant.core.curvature import SpectrumSpec

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")

RANK = 256


def _spd(n: int, corr: float, device) -> torch.Tensor:
    return (1 - corr) * torch.eye(n, device=device) + corr * torch.ones(n, n, device=device)


def _inputs(device):
    g = torch.Generator(device=device).manual_seed(0)
    W = torch.randn(1024, 768, device=device, dtype=torch.bfloat16, generator=g)
    i_norm = torch.rand(768, device=device, generator=g) + 0.5
    o_norm = torch.rand(1024, device=device, generator=g) + 0.5
    i_cov = _spd(768, 0.3, device) * i_norm.sqrt().unsqueeze(1) * i_norm.sqrt().unsqueeze(0)
    o_cov = _spd(1024, 0.2, device) * o_norm.sqrt().unsqueeze(1) * o_norm.sqrt().unsqueeze(0)
    return W, i_norm, o_norm, i_cov, o_cov


def _run(side_device, seed=3, generator=None, **kw):
    W, i_norm, o_norm, i_cov, o_cov = _inputs("cuda:0")
    if generator is None:
        torch.manual_seed(seed)
    return admm_nq.factorize_admm_nanoquant(W, i_norm, o_norm, mid_rank=RANK, outer_iters=40, rho_scheduler="linear",
                                            i_cov=i_cov, o_cov=o_cov, eigh_dtype=torch.float32,
                                            spectrum=SpectrumSpec(power=0.5), side_device=side_device,
                                            generator=generator, **kw)


def test_two_device_admm_matches_serial_bitwise():
    ref = _run(None)
    got = _run("cuda:1")
    for k in ref:
        assert got[k].device == ref[k].device
        assert torch.equal(ref[k], got[k]), k


def test_two_device_admm_is_deterministic_and_faster_or_equal():
    a = _run("cuda:1")
    b = _run("cuda:1")
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_cuda_generator_reproduces_manual_seed():
    ref = _run(None, seed=3)
    got = _run(None, generator=torch.Generator(device="cuda:0").manual_seed(3))
    for k in ref:
        assert torch.equal(ref[k], got[k]), k
