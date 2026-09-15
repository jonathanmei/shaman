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

## Beyond the statevector: 64 and 256 bits, depth 2 and 3

The statevector solver doubles in cost per added bit (measured RQAOA wall time 0.23 s at 9 bits, 0.7 s at 12, 1.6 s at
14, 4.9 s at 16) and stops near 22 bits; no QPU has 256 qubits and one dense p = 1 layer at 256 bits is 32 640
two-qubit gates. Larger tiles run on two other engines.

**Pauli propagation** (`nanoquant.quantum.pauli_prop`). The observable is propagated backwards through the layers as a
sum of product operators `coef * prod_q (alpha_q I + x_q X + y_q Y + z_q Z)`. A mixer layer rotates each site's
`(x, y, z)` about the warm-start axis by `-2 beta` (exact). A cost layer acts through the off-diagonal sites `A`:
`U_C^dag P U_C = P exp(-2i gamma sum_{q in A} Z_q (h_q + sum_{r notin A} J_qr Z_r))`, and resolving the `Z_q` on `A`
into eigenvalues `z_q` gives, per `(A, z_A)`, one product operator again (`o_q -> ((x + i z y) X + (y - i z x) Y) / 2`
on `A`, `d_r -> (a C - i b S) I + (b C - i a S) Z` with `C, S = cos, sin(2 gamma sum_q z_q J_qr)` elsewhere). The
observable's own sites (roots) are enumerated exactly (`3^|roots|` choices per layer); every other site only carries
off-diagonal content of order `sin(2 gamma J)`, so choosing it into `A` is a perturbative order and `k` bounds the
total number of such choices. `k = 0` is exact at p = 1; at p = 2 the measured error ladder on a weak-coupling
8-spin instance is 8e-2, 4e-3, 2e-4 for k = 0, 1, 2 and machine precision when untruncated. `max_active` limits the
eligible non-root sites to the strongest-coupled ones and `pair_top` limits which `<Z_i Z_j>` are evaluated (the energy
the optimiser sees then sums those pairs only). The bits RQAOA returns are always scored exactly; the truncation only
influences which variable each step fixes.

Measured cost of one `(energy, z, zz)` evaluation on the laptop (the recursion does one per step plus a few dozen per
angle optimisation): 64 bits p = 1 all pairs 0.09 s; p = 2 (k = 1, 16 active, 572 pairs) 4.2 s; p = 3 (k = 1, 16
active, 298 pairs) 87 s; 256 bits p = 1 all 32 640 pairs 4.6 s; p = 2 (k = 1, 32 active, 1222 pairs) 118 s. Hence the
run design: full optimisation each step at p = 1; angles carried and re-optimised every 16 steps at p = 2 on 64 bits;
for p = 3 and for 256 bits at p = 2 the angles are optimised once on a 24- or 32-spin sub-instance (normalised by the
cost scale) and held fixed (`angles_sub_tile`), with `pair_top` 4 and `max_active` 8-16 (`depth_overrides` in the
configs).

**MPS cross-check** (`nanoquant.quantum.mps_check`, quimb `CircuitMPS`, bond dimension capped). Same circuit as the
IonQ backend; non-adjacent RZZ gates go through quimb's swap-and-split, which is where the bond cap bites on an
all-to-all cost layer. Used only in the `crosscheck` stage: statevector vs Pauli propagation (k = 0, 1, 2) vs MPS
(chi = 16, 64) on a 16-bit sub-instance, and Pauli propagation vs MPS against each other on the 32-bit dump.

Classical references above 22 bits (no brute force): greedy single-flip descent from the ADMM bits and best-of-8
simulated annealing; `fraction_captured` is then relative to the better of the two and flagged as such.

## Readout: 64 and 256 bits, depths 1-3 (2026-09-15, jobs 5710925-5710927, run records under `runs/`)

Same layer (`27.mlp.down_proj`, rank 1024), tiles nested by the least-confident-bit rule. Energies are the
curvature-weighted layer error; the ADMM bits give 4.606744e-2 on every tile. Reference above 22 bits = better of
greedy 1-flip descent from the ADMM bits and best-of-8 simulated annealing (they agreed on both tiles).

| tile | p | engine | RQAOA energy | gain captured | 1-flip local min | bits flipped | wall time | verify (fp32 drop) |
|---|---|---|---|---|---|---|---|---|
| 64 | 1 | exact p = 1, all pairs, re-optimised each step | 4.606017e-2 | 100% (= reference) | yes | 29 / 64 | 3.5 min | 7.264e-6 vs 7.272e-6 predicted |
| 64 | 2 | Pauli k = 1, 16 active, top-8 pairs, re-opt every 16 | 4.606017e-2 (same bits as p = 1) | 100% | yes | 29 / 64 | 2.6 h (2.5 h in the 8-start first step) | same |
| 64 | 3 | Pauli k = 1, 8 active, top-4 pairs, angles from 24-spin sub-tile | 4.606017e-2 (same bits) | 100% | yes | 29 / 64 | 54 min (47 min sub-tile optimisation) | same |
| 256 | 1 | exact p = 1, top-8 pairs, re-opt every 4 | 4.605325e-2 | 97% | no | 78 / 256 | 23 min | 1.419e-5 vs 1.420e-5 |
| 256 | 2 | Pauli k = 1, 16 active, top-4 pairs, angles from 32-spin sub-tile | 4.605302e-2 | 98% | no | 76 / 256 | 53 min (10 min sub-tile) | 1.442e-5 vs 1.443e-5 |

Relative layer-error change: -1.6e-4 (64 bits), -3.1e-4 (256 bits). Depth did not change the answer at 64 bits and
moved 8 of 256 bits at 256 bits (p = 2 slightly better than p = 1, still short of a 1-flip local minimum: the
restricted pair set and fixed angles leave a residue that one greedy pass would close).

**Cross-check on the 32-bit tile** (`runs/qaoa_sign_polish/qwen3_0p6b_n32_crosscheck/crosscheck.json`). Angles were
optimised on the statevector of the 16-bit sub-instance per depth and reused on the full tile.

| depth | sub-instance (16 bits) vs statevector, max |d<ZZ>| | full 32 bits, methods vs each other |
|---|---|---|
| 1 | Pauli k = 0..2: 6e-14; MPS chi = 16: 1.6e-6, chi = 64: 1.1e-7 | all agree to 1e-6 |
| 2 | Pauli k = 0: 1.8e-1, k = 1: 4.6e-2, k = 2: 1.3e-2; MPS chi = 16: 1.9e-3, chi = 64: 3.4e-6 | Pauli k = 1 vs k = 2: 7e-2; MPS chi = 16 vs chi = 64: ~2e-2 apart; k = 2 vs chi = 64: 1.9e-2 |
| 3 | Pauli k = 1: 4.1e-2, k = 2: 3.8e-3; MPS chi = 16: 3.1e-4, chi = 64: 6.5e-7 | Pauli k = 0 vs k = 1: 8e-2; MPS chi = 16 vs chi = 64: 4e-2 |

Reading: at p = 1 everything is exact, as designed. At p >= 2 on the *real* tile the optimised angles put the coupling
phases at `gamma J ~ 0.1-0.2 rad`, so the perturbative truncation converges slowly (a few percent error in
individual correlations at k = 1, ~1% at k = 2) while the MPS at chi = 64 is exact on 16 qubits; on 32 qubits neither
engine is converged (chi = 16 vs 64 differ by 2e-2, k = 1 vs 2 by 7e-2). Energies nevertheless agree to ~3e-8 because
the fields dominate. So the p > 1 recursions above ran on correlations with a few percent error; that was enough to
make the same eliminations as p = 1 at 64 bits, and the 256-bit p = 2 result should be read as consistent with p = 1
rather than independently confirmed. The right p > 1 engine for a definitive statement at 32-64 bits is the MPS at
chi >= 64 (exact-checked at 16), which cost 37-45 min per evaluation on the cluster (50-80 s on a laptop: the cluster
numpy/quimb run was effectively single-threaded).

Lessons for the next round: `python -u` in Slurm jobs (stdout is block-buffered, logs appear only at exit); never 8
random angle starts at a few seconds per evaluation (2.5 h at 64 bits p = 2) - optimise on a sub-tile or carry angles;
check BLAS threading on the cluster before MPS runs.

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

# larger tiles / deeper circuits: one cluster job each, everything (dump, solve per depth, verify per depth) in-process
objob submit --partition gpus --time 04:00:00 --mem 64G --gres gpu:a100:1 -- \
  uv run python scripts/qaoa_sign_polish.py all        configs/qaoa_sign_polish_0p6b_n64_p123.json
objob submit ... -- uv run python scripts/qaoa_sign_polish.py all        configs/qaoa_sign_polish_0p6b_n256_p12.json
objob submit ... -- uv run python scripts/qaoa_sign_polish.py crosscheck configs/qaoa_sign_polish_0p6b_n32_crosscheck.json  # dumps first if needed
```

The 9/12-bit runs used a local `solve` only because the IonQ backend needs the API token and outbound internet; the
64/256-bit solves are minutes to hours of numpy, so they run where the dump is.

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
