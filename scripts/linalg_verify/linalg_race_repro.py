"""Fresh-process reproducer of the CUDA linalg lazy-initialisation race (docs/issues/threaded_gpu_stages_14b.md).

Each repeat spawns a new Python process (the lazy state is per process), which starts one thread per visible GPU
and runs a large ``torch.linalg.eigh`` on its device, with or without ``warm_up_linalg`` on the main thread first.
Without the warm-up the run may raise ``lazy wrapper should be called at most once``, return garbage or hang;
with it every run must pass.

Examples
--------
::

    uv run python scripts/linalg_verify/linalg_race_repro.py --repeats 5 --size 5120

Exit status is non-zero when any warm-up run fails; the no-warm-up counts are informational (timing dependent).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def child(size: int, warm_up: bool) -> None:
    """One fresh-process trial: concurrent eigh on every visible device, optionally after the warm-up."""
    import torch

    from nanoquant.utils.utils import device_context, run_in_threads, warm_up_linalg

    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if warm_up:
        warm_up_linalg(devices)
    ref = None

    def work(device: str) -> None:
        nonlocal ref
        with device_context(device):
            if warm_up:
                warm_up_linalg([device])
            g = torch.Generator(device="cpu").manual_seed(0)
            A = torch.randn(size, size, generator=g)
            A = (A @ A.mT / size + torch.eye(size)).to(device)
            lam, _ = torch.linalg.eigh(A)
            torch.cuda.synchronize(device)
            lam = lam.cpu()
            if not torch.isfinite(lam).all() or lam.min() < 0.5:
                raise RuntimeError(f"{device}: eigenvalues corrupted (min {lam.min().item():.3g})")
            if ref is None:
                ref = lam
            elif not torch.allclose(lam, ref, rtol=1e-3, atol=1e-3):
                raise RuntimeError(f"{device}: eigenvalues differ from the first device's")

    run_in_threads([lambda d=d: work(d) for d in devices])
    print(f"ok: {len(devices)} devices, warm_up={warm_up}")


def main(argv: list[str] | None = None) -> None:
    """Run the trials of both modes in fresh subprocesses and summarise."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--size", type=int, default=5120, help="matrix size (14B factors are 5120 / 17408)")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds before a trial counts as hung")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--warm-up", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.child:
        child(args.size, args.warm_up)
        return

    import torch

    if torch.cuda.device_count() < 2:
        print("fewer than two CUDA devices, nothing to test")
        return
    summary: dict[str, dict[str, int]] = {}
    for warm_up in (False, True):
        counts = {"pass": 0, "fail": 0, "hang": 0}
        for r in range(args.repeats):
            cmd = [sys.executable, __file__, "--child", "--size", str(args.size)] + (["--warm-up"] if warm_up else [])
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, check=False)
            except subprocess.TimeoutExpired:
                counts["hang"] += 1
                print(f"[warm_up={warm_up}] trial {r + 1}: HANG (> {args.timeout:.0f}s)")
                continue
            status = "pass" if proc.returncode == 0 else "fail"
            counts[status] += 1
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-1:] or [""]
            print(f"[warm_up={warm_up}] trial {r + 1}: {status.upper()} in {time.time() - t0:.0f}s: {tail[0][:160]}")
        summary["warm_up" if warm_up else "no_warm_up"] = counts
    print("summary:", summary)
    if summary["warm_up"]["pass"] != args.repeats:
        raise SystemExit("warm-up runs did not all pass")


if __name__ == "__main__":
    main()
