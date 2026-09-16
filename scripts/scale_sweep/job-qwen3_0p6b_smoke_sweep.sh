#!/bin/bash
#SBATCH --job-name=sw-smoke
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:2
#SBATCH --requeue
#SBATCH --output=/home/jonathan.mei/code/shaman/.objob/logs/%x-%j.out

set -euo pipefail
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
cd ~/code/shaman-sweep
env PYTHONUNBUFFERED=1 uv run nanoquant configs/qwen3_0p6b_smoke_sweep.json
