"""Tests for the thread helpers behind the layer-sharded stages: the CUDA linalg warm-up (serialised, per-thread
idempotent, a no-op without CUDA) and the fail-fast pool runner (docs/issues/threaded_gpu_stages_14b.md)."""

import threading
import time

import pytest
import torch

from nanoquant.utils import utils as U


def test_warm_up_linalg_cpu_is_noop(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(U, "_warm_up_linalg_ops", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
    assert U.warm_up_linalg(["cpu"]) is None
    assert U.warm_up_linalg([torch.device("cpu"), "cpu"], dtypes=(torch.float32,)) is None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert U.warm_up_linalg(["cuda:0"]) is None
    assert calls["n"] == 0


def test_warm_up_linalg_idempotent_per_thread(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(U, "_LINALG_TLS", threading.local())
    seen: list[tuple[str, torch.dtype, str]] = []

    def recorder(device, dtype):
        assert U._LINALG_LOCK.locked()  # the warm-up itself is serialised
        seen.append((str(device), dtype, threading.current_thread().name))

    monkeypatch.setattr(U, "_warm_up_linalg_ops", recorder)
    U.warm_up_linalg(["cuda:0", "cuda:1"])
    U.warm_up_linalg(["cuda:0", "cuda:1"])  # second call on the same thread: nothing new
    U.warm_up_linalg(["cuda"])  # bare "cuda" = the current device, already warm
    assert len(seen) == 4
    assert {(d, dt) for d, dt, _ in seen} == {(f"cuda:{i}", dt) for i in (0, 1)
                                              for dt in (torch.float32, torch.float64)}
    t = threading.Thread(target=U.warm_up_linalg, args=(["cuda:1"],), name="worker")
    t.start()
    t.join()
    assert len(seen) == 6 and all(name == "worker" for _, _, name in seen[4:])


def test_run_in_threads_success_returns_none():
    out: list[int] = []
    lock = threading.Lock()

    def fn(i):
        with lock:
            out.append(i)

    assert U.run_in_threads([lambda i=i: fn(i) for i in range(3)]) is None
    assert sorted(out) == [0, 1, 2]


def test_run_in_threads_reraises_first_exception_and_stops_others():
    stop = threading.Event()
    state = {"b_exited": False, "b_iters": 0}

    def a():
        time.sleep(0.02)
        raise RuntimeError("boom")

    def b():
        while not stop.is_set():
            state["b_iters"] += 1
            time.sleep(0.005)
        state["b_exited"] = True

    t0 = time.time()
    with pytest.raises(RuntimeError, match="boom"):
        U.run_in_threads([a, b], stop=stop)
    assert stop.is_set() and state["b_exited"]  # b saw the stop flag and left; the pool was joined
    assert time.time() - t0 < 2.0
