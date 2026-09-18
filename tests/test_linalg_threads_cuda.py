"""GPU-only smoke check of the fixed threaded linalg path: after the main-thread warm-up, one thread per device runs
eigh / cholesky_ex / svdvals concurrently and matches a CPU reference. Needs at least two CUDA devices
(``sbatch scripts/linalg_verify/job-linalg_repro.sh`` runs it with the fresh-process reproducer of the race itself,
which this in-process test cannot reproduce once the backend is loaded); skipped elsewhere."""

import threading

import pytest
import torch

from nanoquant.utils import utils as U

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")

N = 2048


def _spd(n: int, device, dtype) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(0)
    A = torch.randn(n, n, generator=g).to(dtype)
    return (A @ A.mT / n + torch.eye(n, dtype=dtype)).to(device)


def test_concurrent_linalg_on_distinct_devices_after_warm_up():
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    U.warm_up_linalg(devices)
    ref = torch.linalg.eigvalsh(_spd(N, "cpu", torch.float64))
    out: dict[tuple[str, torch.dtype], torch.Tensor] = {}
    lock = threading.Lock()

    def work(device: str) -> None:
        with U.device_context(device):
            U.warm_up_linalg([device])
            for dt in (torch.float32, torch.float64):
                A = _spd(N, device, dt)
                lam, _ = torch.linalg.eigh(A)
                _, info = torch.linalg.cholesky_ex(A)
                assert int(info.item()) == 0
                sv = torch.linalg.svdvals(A)
                torch.cuda.synchronize(device)
                assert torch.allclose(sv.sort().values.double(), lam.double(), rtol=1e-3, atol=1e-3)
                with lock:
                    out[(device, dt)] = lam.double().cpu()

    U.run_in_threads([lambda d=d: work(d) for d in devices])
    assert len(out) == 2 * len(devices)
    for (device, dt), lam in out.items():
        assert torch.allclose(lam, ref, rtol=1e-4, atol=1e-4), (device, dt)
