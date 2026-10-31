#!/usr/bin/env bash
# Submit from the MA-VLCM checkout: mkdir -p logs; BACKBONE_MODE=frozen sbatch train_palmetto.sbatch
#SBATCH --job-name=smolvlm_value
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
module load anaconda3
set +u
source activate "${CONDA_ENV:-MARL}"
set -u
export NUM_GPUS="${NUM_GPUS:-1}"
export DATASET="${DATASET:-/home/anshuln/Dataset/polygon_formation_python_env}"
export BACKBONE_MODE="${BACKBONE_MODE:-frozen}"
export RETURN_SOURCE="${RETURN_SOURCE:-control}"
export RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/outputs/smolvlm-${SLURM_JOB_ID}}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
expected_episodes="${EXPECTED_EPISODES:-3000}"
if [[ ! -d "$DATASET/formation_control" ]]; then
    echo "Dataset not found: $DATASET/formation_control" >&2
    exit 1
fi
episode_count=$(find "$DATASET/formation_control" -type f -name 'episode_*.tar' | wc -l)
if [[ "$episode_count" -ne "$expected_episodes" ]]; then
    echo "Expected $expected_episodes episode archives in $DATASET; found $episode_count" >&2
    exit 1
fi
echo "Dataset: $DATASET ($episode_count episodes)"
echo "Backbone: $BACKBONE_MODE; return source: $RETURN_SOURCE; GPU processes: $NUM_GPUS"
# WANDB_API_KEY/HF_TOKEN belong in the environment or login state, never here.
nvidia-smi
exec bash "$PROJECT_ROOT/train.sh" "$@"
