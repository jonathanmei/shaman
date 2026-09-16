#!/bin/bash
#SBATCH --job-name=sw-8b_stats512
#SBATCH --partition=lgpus
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --gres=gpu:a100:4
#SBATCH --requeue
#SBATCH --output=/home/jonathan.mei/code/shaman/.objob/logs/%x-%j.out

set -euo pipefail
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
cd ~/code/shaman-sweep
env PYTHONUNBUFFERED=1 uv run nanoquant configs/qwen3_8b_best_stats512.json
