#!/usr/bin/env bash
# Submit from the MA-VLCM checkout: mkdir -p logs; sbatch train_palmetto.sbatch
#SBATCH --job-name=smolvlm_value
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12gb
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"
module load anaconda3
set +u
source activate "${CONDA_ENV:-MARL}"
set -u
export NUM_GPUS="${NUM_GPUS:-2}"
export DATASET="${DATASET:-$HOME/Dataset/formation_vlcm_polygon_3to6}"
export RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/outputs/smolvlm-${SLURM_JOB_ID}}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
# WANDB_API_KEY/HF_TOKEN belong in the environment or login state, never here.
nvidia-smi
exec bash "$PROJECT_ROOT/train.sh" "$@"
