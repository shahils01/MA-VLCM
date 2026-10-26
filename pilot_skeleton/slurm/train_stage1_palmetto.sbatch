#!/usr/bin/env bash
# Clemson Palmetto 2 Slurm template. Edit GPU, partition, paths, and env before use.
# Current Palmetto batch jobs typically use work1; owner partitions depend on allocation.
# Check current options with: sinfo --Format PartitionName
# See: https://docs.rcd.clemson.edu/palmetto/job_management/submit/

#SBATCH --job-name=ma_vlcm_stage1
#SBATCH --partition=work1
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# TODO: set this to the MA-VLCM checkout path visible on the compute node.
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"

# Create this environment before submission on a compatible node/software stack.
PYTHON_ENV="${PYTHON_ENV:-$PROJECT_ROOT/.venv}"
source "$PYTHON_ENV/bin/activate"

# Point HF_HOME and model/data outputs to Clemson-approved storage with enough space.
# TODO: export HF_TOKEN securely before sbatch only when private Hub assets require it.
# Do not write a token into this file, the repository, or job logs.
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/checkpoints/stage1"

echo "Job: ${SLURM_JOB_ID:-interactive}"
echo "Host: $(hostname)"
python --version
nvidia-smi

python "$PROJECT_ROOT/pilot_skeleton/train_stage1_pseudocode.py" \
  --dataset "${HF_DATASET:-shulnak09/formation-vlcm-polygon-3to6}" \
  --dataset-revision "${HF_DATASET_REVISION:-main}" \
  --model-id "${VLM_MODEL:-Qwen/Qwen2-VL-2B-Instruct}" \
  --model-revision "${VLM_REVISION:-main}" \
  --output-dir "$PROJECT_ROOT/pilot_runs/iteration1" \
  --epochs "${EPOCHS:-10}"
