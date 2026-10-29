#!/usr/bin/env bash
# Submit from MA-VLCM after creating logs; this does not submit/change the LoRA job.
#SBATCH --job-name=vlcm_frozen_2000
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if ! command -v conda >/dev/null 2>&1 && [[ -z "${CONDA_BASE:-}" ]]; then
        module load anaconda3
    fi
    conda_base="${CONDA_BASE:-$(conda info --base)}"
    set +u
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-MARL}"
    set -u
    export PYTHON_BIN="$(command -v python)"
fi
export DATASET="${DATASET:-$HOME/Dataset/formation_vlcm_polygon_3to6_10hz_inner00m}"
export EXTERNAL_DATASET="${EXTERNAL_DATASET:-$HOME/Dataset/formation_validation_seed7300001}"
export RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/outputs/frozen_2000_${SLURM_JOB_ID}}"
export CACHE_DIR="${FROZEN_CACHE_DIR:-$PROJECT_ROOT/outputs/cache_frozen_2000}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
exec bash "$PROJECT_ROOT/train_frozen_2000.sh"
