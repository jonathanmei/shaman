# Threaded multi-GPU stages fail at 14B: `lazy wrapper should be called at most once`

Status 2026-09-17: fix implemented on branch `threaded-linalg-warmup` (`warm_up_linalg` / `run_in_threads` in
`utils/utils.py`, see "Fix" below); 14B verification on 4 a100 pending (job ids below once submitted). Until it
passes the 14B configs keep the workaround `parallel_devices: 1` (serial probe / calibration); `admm_parallel_sides`
(two-GPU ADMM) is unaffected.

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

## Fix (branch `threaded-linalg-warmup`)

1. **Warm-up on the main thread.** `utils/utils.py` `warm_up_linalg(devices, dtypes=(float32, float64))` issues a
   tiny `eigh`, `cholesky_ex` + `cholesky_inverse` and `svdvals` per device and dtype and synchronises. PyTorch's
   CUDA linalg backend (`libtorch_cuda_linalg`) is loaded by the first `torch.linalg` call of the process and stays
   loaded (`cleanup_memory` only clears cuBLAS workspaces), so one main-thread call before a pool removes the race.
   Called at the top of the threaded branch of `measure_sensitivity` and, when sharding, in `_collect_kron_stats`
   (which `refresh_block_curvature` and the pipeline's statistics call share).
2. **Serialised first call per worker.** The warm-up runs under a module-level `threading.Lock` and remembers what
   it warmed per thread (`threading.local`); every worker (`worker` in the probe, `run_group` in the collector)
   calls `warm_up_linalg([device])` first, so its first linalg call is the locked warm-up and the large
   eigendecompositions run unlocked afterwards (steady-state parallelism unchanged). `core/admm_nq.py` is untouched.
3. **Fail fast.** `run_in_threads(fns, stop)` replaces the two `ThreadPoolExecutor` blocks: it waits with
   `FIRST_EXCEPTION`, sets the `stop` event (the probe worker checks it between layers), cancels the not-yet-started
   functions, joins the running ones and re-raises the first exception, so a raised worker terminates the job
   instead of leaving it RUNNING with idle siblings (5733276) or with orphaned threads (5733277).
4. **Cache keys.** `core/rank_probe.py` (`blocks`, `rank_probe` groups) and `core/importance.py` (`stats`) changed,
   so statistics, probes, blocks and KD are re-keyed. `scripts/alias_cache_keys.py <config> --kind stats|rank_probe
   --old <old key prefix>` creates the symlink alias `ArtifactCache.load` accepts; `utils/utils.py`, `scripts/`,
   configs and tests are not fingerprinted.
5. **Tests.** `tests/test_thread_pools.py` (warm-up no-op / idempotence / lock, `run_in_threads` fail-fast),
   `test_measure_sensitivity_fails_fast`, `test_sharded_collector_fails_fast` (CPU), and the GPU-gated
   `tests/test_linalg_threads_cuda.py` (needs two CUDA devices).

### Verification (cluster, pinned checkout `ob:~/code/shaman-linalg`)

- `scripts/linalg_verify/job-linalg_repro.sh` (short, 4 a100): `scripts/linalg_verify/linalg_race_repro.py` runs
  5 fresh processes per mode, each with one thread per GPU doing a 5120² float32 eigh, without and with the warm-up
  (the no-warm-up counts are timing dependent; all warm-up runs must pass), then the two GPU test modules.
- `scripts/linalg_verify/job-qwen3_14b_probe_verify.sh` (lgpus, 4 a100, 800G): `configs/qwen3_14b_probe_verify_parallel4.json`
  = the 14B 512-sample config with `parallel_devices: 4`, `cache_dir: cache_verify` (only the statistics aliased
  in, so the probe recomputes threaded) and `max_blocks: 1`. Expect `[cache] alias stats ... -> 80831d36aeb6`,
  `[cache] miss rank_probe`, 40 `[rank probe] block` lines and `... on 4 device(s)`; compare `probes` / `curves` with
  the serial artifact `f9df6a548b74` of job 5768981.

## Workaround used

`parallel_devices: 1` in `configs/qwen3_14b_best_stats512.json` and `configs/qwen3_14b_best_stats512_typeprior.json`
(probe and calibration serial; ADMM split still on two GPUs, `--gres gpu:a100:2`). The 14B probe then takes about
3 to 3.5 h on an a100 and is cached, so the second arm reuses it.
