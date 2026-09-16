#!/bin/bash
# Scale sweep (2026-09-16): 512-sample statistics x type-weight prior at Qwen3-8B / 14B, 4 GPUs per arm.
# Run on the cluster login node from a pinned checkout at ~/code/shaman-sweep (cache/ and checkpoints/
# symlinked to ~/code/shaman). `--smoke` submits only the 0.6B smoke of the new code paths (short partition).
# Hardware: h200 for the 14B 512-sample arms, h100 for the type-prior arms, a100 for the 8B 512-sample arms.
set -euo pipefail
cd "$(dirname "$0")"
if [ "${1:-}" = "--smoke" ]; then
  sbatch job-qwen3_0p6b_smoke_sweep.sh
  exit 0
fi
for f in job-qwen3_14b_best_stats512.sh job-qwen3_14b_best_stats512_typeprior.sh \
         job-qwen3_8b_best_typeprior.sh job-qwen3_14b_best_typeprior.sh \
         job-qwen3_8b_best_stats512.sh job-qwen3_8b_best_stats512_typeprior.sh; do
  sbatch "$f"
done
