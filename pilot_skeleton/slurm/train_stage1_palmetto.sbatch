#!/usr/bin/env bash
# Clemson Palmetto 2 Slurm template. Edit partition, paths, and environment before use.
# Check current options with: sinfo --Format PartitionName
# See: https://docs.rcd.clemson.edu/palmetto/job_management/submit/

#SBATCH --job-name=ma_vlcm_stage1
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12gb
#SBATCH --gpus-per-node=a100:2
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/anshuln/MA_VLCM/MA-VLCM}"
cd "$PROJECT_ROOT"

CONDA_ENV="${CONDA_ENV:-internvl}"
source "$(conda info --base)/etc/profile.d/conda.sh"
source activate "$CONDA_ENV"

# Use Clemson-approved persistent or scratch storage for caches and outputs.
# Provide HF_TOKEN and WANDB_API_KEY through approved secret/environment handling;
# never store credentials in this script or the repository.
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/checkpoints/stage1"

PILOT_DATASET="${PILOT_DATASET:-shulnak09/formation-vlcm-polygon-3to6}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/pilot_runs/iteration1}"

echo "Job: ${SLURM_JOB_ID:-interactive}"
echo "Host: $(hostname)"
python --version
nvidia-smi

python "$PROJECT_ROOT/pilot_skeleton/train_stage1_pseudocode.py" \
  --dataset "$PILOT_DATASET" \
  --dataset-revision "${HF_DATASET_REVISION:-main}" \
  --model-id "${VLM_MODEL:-Qwen/Qwen2-VL-2B-Instruct}" \
  --model-revision "${VLM_REVISION:-main}" \
  --output-dir "$RUN_DIR" \
  --epochs "${EPOCHS:-10}" \
  --wandb-project "${WANDB_PROJECT:-ma-vlcm-value-pilot}" \
  --wandb-entity "${WANDB_ENTITY:-}" \
  --wandb-run-name "${WANDB_RUN_NAME:-palmetto-${SLURM_JOB_ID:-local}}" \
  --wandb-mode "${WANDB_MODE:-online}"
