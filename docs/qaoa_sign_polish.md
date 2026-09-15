# QAOA proof of concept: polishing a tile of stored sign bits with warm-started RQAOA

Feasibility demo only. One tiny binary subproblem is carved out of the real compression pipeline, handed to a
(simulated or IonQ) quantum computer as an Ising model, solved by warm-started recursive QAOA, and checked exactly
against brute force and against the classical bits. No claim about advantage, practicality, or perplexity: the effect
of one tile on the model is far below the ±0.5 PPL single-run spread at 0.6B (`docs/learnings.md` §3).

## Which subproblem, and why

The deployed layer is `W_hat = diag(post) S_A [diag(mid)] S_B diag(pre)` with `S_A`, `S_B` the stored ±1 bits.
ADMM chooses the bits with the element-wise sign rule in its Z-step (`src/nanoquant/core/admm_nq.py`,
`rank1_approx`). Freezing every bit except a tile `T = {(i_t, k_t)}` of `S_A` turns the curvature-weighted layer error
`tr(L (W - W_hat) R (W - W_hat)^T)` into an **unconstrained Ising model** in the tile spins `s_t`:

```
E(s) = const + sum_t h_t s_t + sum_{t<t'} J_tt' s_t s_t'
h_t     = -2 p_{i_t} (L E0 R Vs^T)[i_t, k_t]
J_tt'   =  2 p_{i_t} p_{i_t'} L[i_t, i_t'] (Vs R Vs^T)[k_t, k_t']
const   =  <E0, E0> + sum_t p_{i_t}^2 L[i_t, i_t] (Vs R Vs^T)[k_t, k_t]
```

with `Vs = [diag(mid)] S_B diag(pre)`, `p = post`, `E0 = W - diag(p) U_fixed Vs` (tile bits zeroed) and
`<X, Y> = tr(L X R Y^T)`. The couplings come from the Kronecker curvature `L` and from the Gram matrix of the rows of
`Vs`; two bits in the same output row couple even under diagonal curvature. Only for diagonal `L`, `R` and a tile
inside one mid column do all couplings vanish and the sign rule is optimal (`tests/test_sign_tile.py`). So the
classical heuristic is provably suboptimal on this subproblem, and a QPU can show a real, if tiny, improvement.

Rejected alternatives (not implemented): the rank-allocation knapsack (linear objective, all coupling from one-hot and
budget penalties, greedy already exactly optimal for the convex fitted curves), QAOA as the in-loop ADMM Z-step
(hundreds of QPU calls per layer, noisy projections destabilise ADMM), block ordering / refresh subsets (no usable
coupling or no cheap objective).

## Which layer

Late blocks are where the local Gauss-Newton sensitivity under-values damage (hence the pipeline's depth prior). The
`pick-layer` stage scores every layer of the last third of blocks by its depth-adjusted predicted loss at its allocated
rank, `predicted_loss((a + log m_l, beta), r_l)` with the run's `rank_depth_ramp`, from the cached rank-probe artifact
of the best-recipe run, drops inflated-Fisher outliers (level above 20x the window median, the `2.mlp.down_proj`
pathology of `docs/results.md`), and prints the ranking. The chosen block and module are pinned in the config so the
run is documented (`configs/qaoa_sign_polish_0p6b.json`).

The tile is the `n` least-confident bits of the layer: smallest `|A_latent|`, the ADMM X + U variable whose sign is the
stored bit. Their confidence `min(|A_latent| / median |A_latent|, 1)` feeds the warm start.

## Solver: warm-started RQAOA, one path, no variants

* **Warm start.** Qubit `q` starts in `RY(theta_q)|0>` with `P(flip) = 1/2 - c_q (1/2 - eps)` toward the ADMM bit
  (`eps = 0.1`, Egger et al.'s regularisation). The mixer is `exp(-i beta (sin theta_q X_q + cos theta_q Z_q))`, so
  the warm state is its eigenstate and the ADMM distribution is the `p = 0` baseline.
* **p = 1 per step**, angles optimised on the numpy statevector (Nelder-Mead, 8 seeded starts), cost normalised by
  `max(|h|, |J|)` so the `gamma` range is problem independent.
* **Recursion.** Sample, estimate `<Z_i>` and `<Z_i Z_j>`, fix the largest-magnitude one (`s_i = sign s_j` or
  `s_i = sign`), fold it into `h`, `J`, `const`, drop the qubit and its warm-start angle, repeat until 4 spins remain,
  brute-force those, back-substitute. On hardware each step's angles are still optimised on the simulator; the QPU only
  supplies the correlation estimates, so `n = 9` costs 5 sequential jobs, `n = 12` costs 8.
* **Circuit** (`ionq_backend.build_circuit`): `RY(theta)` init; `RZ(2 gamma h_i)`, `RZZ(2 gamma J_ij)`; mixer
  `RY(theta) RZ(2 beta) RY(-theta)`. `p = 1` on `n` spins is `n(n-1)/2` two-qubit gates: 36 at `n = 9`, 66 at
  `n = 12`, 120 at `n = 16` (simulator only above that). Verified against qiskit's statevector to 1e-10
  (`tests/test_ionq_backend.py`).

A confidently *wrong* warm start can mislead `p = 1` RQAOA (anti-aligned seeds at confidence 0.5 fail on half the
random 8-spin instances); the realistic case, mostly-right bits with the wrong ones flagged by low confidence, recovers
the optimum on every tested instance, as does a plain `|+>` start (`tests/test_rqaoa.py`).

## Running it

```
# cluster (GPU; uses the best-recipe run's cache for calibration statistics and the rank probe)
uv run python scripts/qaoa_sign_polish.py pick-layer configs/qaoa_sign_polish_0p6b.json   # then pin block/module
uv run python scripts/qaoa_sign_polish.py dump       configs/qaoa_sign_polish_0p6b.json   # instance.json, layer_state.pt
# local
uv run python scripts/qaoa_sign_polish.py solve      configs/qaoa_sign_polish_0p6b.json   # solution.json
#   backend "ionq": uv sync --extra quantum, export IONQ_API_KEY, ionq_target "simulator" (+ noise model) or "qpu.aria-1"
# cluster
uv run python scripts/qaoa_sign_polish.py verify     configs/qaoa_sign_polish_0p6b.json   # verify.json
```

`dump` factorises the layer against the FP model's calibration curvature (what the rank probe does), not the
error-fed block inputs or the fresh input factor of the block loop: the instance is a faithful stand-alone layer
problem, not literally the one the pipeline solved at that position.

## Readout (2026-09-15, Qwen3-0.6B-Base, best recipe, `configs/qaoa_sign_polish_0p6b*.json`)

`pick-layer` (cluster job 5702981, cache hit on the calibration statistics, rank probe recomputed) ranked the last
third of blocks (19-27); the top five were `27.mlp.down_proj` (score 8.0e-2, level 23.1), `27.mlp.up_proj` (2.4e-2),
`27.mlp.gate_proj` (1.8e-2), `26.mlp.up_proj` (1.4e-2), `26.mlp.down_proj` (1.1e-2). Chosen: **block 27,
`mlp.down_proj`**, allocated rank 1024 (the cap), shape 1024 x 3072. The tile bits (smallest `|A_latent|`) have
confidence 0.008-0.064, i.e. their ADMM sign is close to a coin flip; fields `|h|` up to 3.5e-7 and couplings `|J|` up
to 1.1e-8 (couplings roughly ten times weaker than fields on this tile).

| layer | n | backend | ADMM energy | RQAOA energy | brute force | gain captured | optimum hit | bits flipped | 2q gates | jobs |
|---|---|---|---|---|---|---|---|---|---|---|
| 27.mlp.down_proj | 9 | statevector | 4.606744e-2 | 4.606604e-2 | 4.606604e-2 | 100% | yes | 5 / 9 | 36 | 5 |
| 27.mlp.down_proj | 12 | statevector | 4.606744e-2 | 4.606597e-2 | 4.606597e-2 | 100% | yes | 6 / 12 | 66 | 8 |

`gain captured = (ADMM - RQAOA) / (ADMM - brute force)`. The ADMM bits were **not** optimal on either tile: the exact
optimum lowers the layer error by 1.40e-6 (n = 9) and 1.47e-6 (n = 12), i.e. by 3.1e-5 and 3.2e-5 relative.
`verify` on the cluster re-applied the RQAOA bits to the saved layer: fp32 `mahalanobis_weight_error`
4.606745e-2 -> 4.606604e-2 (n = 9, measured drop 1.408e-6 vs 1.403e-6 predicted) and -> 4.606597e-2 (n = 12,
1.475e-6 vs 1.472e-6), the residual being fp32 accumulation. Every recursion step fixed a variable with
|correlation| >= 0.93 (mostly single-spin `<Z>` because the fields dominate; one pair elimination at n = 9).

**IonQ stages not run.** `configs/qaoa_sign_polish_0p6b_n9_ionq_sim.json` (Aria-1 noise model) and
`..._n9_ionq_qpu.json` are ready, and the circuits are validated against qiskit's statevector, but the API rejected the
submission with `403 Your account is disabled` for the token in `QISKIT_IONQ_API_TOKEN`; re-run `solve` with those
configs once the account is re-enabled (`IONQ_API_KEY` or `QISKIT_IONQ_API_TOKEN`).

Caveat repeated: a 3e-5 relative change of one layer's weight error is invisible in perplexity; the demo shows the
pipeline -> Ising -> RQAOA -> pipeline loop closes exactly, nothing more.
