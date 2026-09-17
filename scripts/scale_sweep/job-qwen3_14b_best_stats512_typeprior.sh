#!/bin/bash
#SBATCH --job-name=sw-14b_stats512_typeprior
#SBATCH --partition=lgpus
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=800G
#SBATCH --gres=gpu:a100:4
#SBATCH --requeue
#SBATCH --output=/home/jonathan.mei/code/shaman/.objob/logs/%x-%j.out

set -euo pipefail
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
cd ~/code/shaman-sweep
env PYTHONUNBUFFERED=1 uv run nanoquant configs/qwen3_14b_best_stats512_typeprior.json
