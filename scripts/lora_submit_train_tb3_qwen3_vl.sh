#!/bin/bash
#SBATCH --job-name=ma_vlcm_qwen3vl
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gpus=a100:1

set -e

if [ -n "${MA_VLCM_ROOT:-}" ]; then
    REPO_ROOT=$(cd "$MA_VLCM_ROOT" && pwd)
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/lora_run_train_tb3_qwen3_vl.sh" ]; then
    REPO_ROOT=$(cd "$SLURM_SUBMIT_DIR" && pwd)
else
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi

cd "$REPO_ROOT"
mkdir -p logs 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-32}}"
export PYTHONUNBUFFERED=1
export DATASET_PROFILE="${DATASET_PROFILE:-tb3_isaac}"

bash scripts/lora_run_train_tb3_qwen3_vl.sh
