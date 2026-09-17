#!/bin/bash
#SBATCH --job-name=sw-0p6b_eval
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue
#SBATCH --output=/home/jonathan.mei/code/shaman/.objob/logs/%x-%j.out

# Eval-only pass over the 0.6B best checkpoint (job 5617843, PPL 22.96, zero-shot never run):
# from_pretrained_quantize loads checkpoints/qwen3_0p6b_kl_ms_ramp_parity.pt and skips straight to evaluation.
set -euo pipefail
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
cd ~/code/shaman-sweep
env PYTHONUNBUFFERED=1 uv run nanoquant configs/qwen3_0p6b_best_eval.json
