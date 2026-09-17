# Threaded multi-GPU stages fail at 14B: `lazy wrapper should be called at most once`

Status 2026-09-17: open. Workaround in use: `parallel_devices: 1` for the 14B runs (serial probe / calibration);
`admm_parallel_sides` (two-GPU ADMM) is unaffected.

## Symptom

Branch `multi-gpu-admm-probe` (merged into `scale-sweep-calib-typeprior`) shards the rank probe and the
calibration layer groups over worker threads, one per GPU (`parallel_devices`; `core/rank_probe.py`
`measure_sensitivity` → `worker` → `probe_layer`; `core/importance.py` `_collect_kron_stats` → `run_group`).
At Qwen3-14B with four workers every attempt failed; the 0.6B screens and the 8B probe went through.

| job | stage | failure |
|---|---|---|
| 5761695, 5761696 (14B, 4 × a100) | rank probe, ~2 h 50 in | `RuntimeError: lazy wrapper should be called at most once` from `torch.linalg.eigh` in `admm_nq._normalized_curvature`, called from a probe worker thread |
| 5733276 (14B, 4 × h200) | calibration (KL fit) | `torch.linalg.eigh` "failed to converge" in `importance._damped_inverse` inside a sharded group worker; the main thread raised but the non-daemon workers kept the process alive (job RUNNING, doing nothing) |
| 5733277 (14B), 5733281 (8B) | right after the statistics, at the probe | silent for 6+ hours (5733281 burning CPU, 5733277 idle); cancelled; 5733277 then sat in COMPLETING with unkillable threads on obsidian-10 |

Traceback of the probe failure (5761695):

```
File ".../core/rank_probe.py", line 279, in measure_sensitivity
File ".../core/rank_probe.py", line 273, in worker
File ".../core/rank_probe.py", line 179, in probe_layer
File ".../core/admm_nq.py", line 504, in factorize_admm_nanoquant
    Lt, lam_L, Q_L = _normalized_curvature(o_cov.to(device), norm_o, eigh_dtype, eps, spectrum, ...
File ".../core/admm_nq.py", line 385, in _normalized_curvature
    lam, Q = torch.linalg.eigh(Sigma.to(eigh_dtype))
RuntimeError: lazy wrapper should be called at most once
```

## Reading

"lazy wrapper should be called at most once" is a PyTorch internal guard on a lazily initialised object being
entered re-entrantly or concurrently (the CUDA linear-algebra backend and its per-device handles are created on
first use). Four threads issuing `torch.linalg.eigh` on four devices race on that initialisation; depending on
timing the race raises (5761695 / 5761696), corrupts a call into a non-converging solve (5733276), or deadlocks
(the silent stalls). The 14B factors (5120² and 17408²) make the first eigh calls long and overlapping, which is
why 0.6B and 8B mostly slipped through. The refresh path (`refresh_block_curvature`) uses the same sharded
collector and is exposed too.

## Proposed fix (not yet implemented)

1. Warm up the lazy state on the main thread before any pool starts: one small `torch.linalg.eigh` (and
   `cholesky_ex`, `svdvals`) per device and per dtype in use (`float32`, `float64`), e.g. a
   `warm_up_linalg(devices)` helper in `utils/utils.py` called from `measure_sensitivity` and `_collect_kron_stats`.
2. Serialise each worker's first eigh behind a module-level `threading.Lock` (belt and braces; steady-state
   parallelism is unaffected).
3. Fail fast in the pools: `concurrent.futures.wait(..., return_when=FIRST_EXCEPTION)` and cancel the remaining
   futures, or daemon worker threads, so a raised worker cannot leave a job RUNNING with nothing to do (5733276)
   or unkillable (5733277).
4. Cache keys: `rank_probe.py` is in the `blocks` and `rank_probe` fingerprint groups and `importance.py` in
   `stats`, so the fix re-keys probes, blocks and statistics. Land it between runs and alias the cached artifacts
   (symlink `<new key>.pt -> <old key>.pt`, accepted by `ArtifactCache.load`) where recomputation is not wanted.
5. Verify with the 14B probe on 4 GPUs (the failing case) before re-enabling `parallel_devices > 1` at 14B; the
   0.6B screen (`configs/qwen3_0p6b_screen_parallel4.json`) does not reproduce the race.

## Workaround used

`parallel_devices: 1` in `configs/qwen3_14b_best_stats512.json` and `configs/qwen3_14b_best_stats512_typeprior.json`
(probe and calibration serial; ADMM split still on two GPUs, `--gres gpu:a100:2`). The 14B probe then takes about
3 to 3.5 h on an a100 and is cached, so the second arm reuses it.
