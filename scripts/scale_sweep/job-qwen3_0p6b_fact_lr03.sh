#!/bin/bash
#SBATCH --job-name=sw-fact_lr03
#SBATCH --partition=gpus
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue
#SBATCH --output=/home/jonathan.mei/code/shaman/.objob/logs/%x-%j.out

set -euo pipefail
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
cd ~/code/shaman-sweep
env PYTHONUNBUFFERED=1 uv run nanoquant configs/qwen3_0p6b_fact_lr03.json
