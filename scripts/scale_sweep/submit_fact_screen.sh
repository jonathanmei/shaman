#!/bin/bash
# Factor-tuning screen (2026-09-18, 0.6B, 4 blocks): control | keep-best | fact lr x0.3 | both | ADMM 800 iters.
set -euo pipefail
cd "$(dirname "$0")"
for f in job-qwen3_0p6b_fact_ctrl.sh job-qwen3_0p6b_fact_best.sh job-qwen3_0p6b_fact_lr03.sh job-qwen3_0p6b_fact_best_lr03.sh job-qwen3_0p6b_fact_admm800.sh; do sbatch "$f"; done
