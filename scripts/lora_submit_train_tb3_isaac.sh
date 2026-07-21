#!/bin/bash
#SBATCH --job-name=ma_vlcm_tb3_isaac
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gpus=h100:1

set -e

if [ -n "${MA_VLCM_ROOT:-}" ]; then
    REPO_ROOT=$(cd "$MA_VLCM_ROOT" && pwd)
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/lora_run_train_tb3_lab.sh" ]; then
    REPO_ROOT=$(cd "$SLURM_SUBMIT_DIR" && pwd)
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/lora_run_train_tb3_lab.sh" ]; then
    REPO_ROOT=$(cd "$SLURM_SUBMIT_DIR/.." && pwd)
else
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi
SCRIPT_DIR="$REPO_ROOT/scripts"
cd "$REPO_ROOT"

# Slurm opens these paths before the script starts, so create logs/ once from
# the login node before the first submission as shown in the README.
mkdir -p logs 2>/dev/null || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

USER_NAME="${USER:-$(id -un 2>/dev/null || echo user)}"
if [ -n "${MA_VLCM_SCRATCH_ROOT:-}" ]; then
    SCRATCH_ROOT="$MA_VLCM_SCRATCH_ROOT"
elif [ -n "${SCRATCH:-}" ]; then
    SCRATCH_ROOT="$SCRATCH/ma_vlcm"
elif [ -d "/scratch/$USER_NAME" ]; then
    SCRATCH_ROOT="/scratch/$USER_NAME/ma_vlcm"
elif [ -d "/scratch/aparame" ]; then
    SCRATCH_ROOT="/scratch/aparame/ma_vlcm"
elif [ -n "${SLURM_TMPDIR:-}" ]; then
    SCRATCH_ROOT="$SLURM_TMPDIR/ma_vlcm"
else
    SCRATCH_ROOT="$REPO_ROOT/.scratch"
fi

export MA_VLCM_SCRATCH_ROOT="$SCRATCH_ROOT"
export DATASET_PROFILE=tb3_isaac
HF_DATASET_REPO="${HF_DATASET_REPO:-adi2440/tb3-isaac-vlcm}"
DATA_DIR="${DATA_DIR:-hf://datasets/$HF_DATASET_REPO/**/*.tar}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
SAVE_DIR="${SAVE_DIR:-$SCRATCH_ROOT/checkpoints/tb3_isaac}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-${EPOCHS:-20}}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-tb3_isaac_0.5B}"
TRAIN_FROM_SCRATCH="${TRAIN_FROM_SCRATCH:-0}"
export SAVE_DIR NUM_PROCESSES HF_DATASET_REPO TOTAL_EPOCHS MIXED_PRECISION
export WANDB_RUN_PREFIX TRAIN_FROM_SCRATCH

echo "Submitting Isaac Sim TurtleBot3 MA-VLCM LoRA fine-tune"
echo "Data: $DATA_DIR"
echo "Resume checkpoint: ${RESUME_CHECKPOINT:-automatic default}"
echo "Save dir: $SAVE_DIR"
echo "Total epoch target: $TOTAL_EPOCHS"
echo "Mixed precision: $MIXED_PRECISION"

bash "$SCRIPT_DIR/lora_run_train_tb3_lab.sh" "$DATA_DIR" "$RESUME_CHECKPOINT"
