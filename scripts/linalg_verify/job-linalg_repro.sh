#!/bin/bash
# Fresh-process reproducer of the CUDA linalg lazy-init race + the GPU-gated thread tests, 4 a100 (short partition).
# Run on the login node from the pinned checkout ~/code/shaman-linalg (docs/issues/threaded_gpu_stages_14b.md).
#SBATCH --job-name=linalg-repro
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:4
#SBATCH --output=/home/jonathan.mei/code/shaman/.objob/logs/%x-%j.out

set -euo pipefail
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
cd ~/code/shaman-linalg
nvidia-smi -L
env PYTHONUNBUFFERED=1 uv run python scripts/linalg_verify/linalg_race_repro.py --repeats 5 --size 5120
env PYTHONUNBUFFERED=1 uv run pytest tests/test_linalg_threads_cuda.py tests/test_admm_split_cuda.py -q
