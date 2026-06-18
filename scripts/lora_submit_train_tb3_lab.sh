#!/bin/bash
#SBATCH --job-name=ma_vlcm_tb3_lora
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gpus=h100:1

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

mkdir -p logs

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/tb3_lab}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$REPO_ROOT/checkpoints/NewFinal_0.5B.pt}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/outputs/checkpoints/tb3_lab}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
export SAVE_DIR NUM_PROCESSES

echo "Submitting TB3 lab LoRA fine-tune"
echo "Data: $DATA_DIR"
echo "Resume checkpoint: $RESUME_CHECKPOINT"
echo "Save dir: $SAVE_DIR"

bash "$SCRIPT_DIR/lora_run_train_tb3_lab.sh" "$DATA_DIR" "$RESUME_CHECKPOINT"
