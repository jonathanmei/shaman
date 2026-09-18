#!/bin/bash
#SBATCH --job-name=sw-14b_stats1024
#SBATCH --partition=lgpus
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=800G
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --requeue
#SBATCH --output=/home/jonathan.mei/code/shaman/.objob/logs/%x-%j.out

# 14B with 1024 statistics samples (follow-up of the 512-sample arm, PPL 10.91): threaded probe / calibration on
# four GPUs with the linalg warm-up fix (branch threaded-linalg-warmup, verified by job 5771120).
set -euo pipefail
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
cd ~/code/shaman-sweep
env PYTHONUNBUFFERED=1 uv run nanoquant configs/qwen3_14b_best_stats1024.json
