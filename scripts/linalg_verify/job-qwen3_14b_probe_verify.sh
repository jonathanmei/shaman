#!/bin/bash
# 14B rank probe on 4 a100 with the threaded (parallel_devices 4) path after the linalg warm-up fix: the failing case
# of docs/issues/threaded_gpu_stages_14b.md. Separate cache dir (cache_verify) with only the 512-sample statistics
# aliased in, so the probe recomputes; max_blocks 1 stops after one block. Pinned checkout ~/code/shaman-linalg.
#SBATCH --job-name=verify-14b_probe4
#SBATCH --partition=lgpus
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=800G
#SBATCH --gres=gpu:a100:4
#SBATCH --output=/home/jonathan.mei/code/shaman/.objob/logs/%x-%j.out

set -euo pipefail
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
cd ~/code/shaman-linalg
nvidia-smi -L
env PYTHONUNBUFFERED=1 uv run nanoquant configs/qwen3_14b_probe_verify_parallel4.json
