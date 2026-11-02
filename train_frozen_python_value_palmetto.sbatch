#!/usr/bin/env bash
#SBATCH --job-name=mappo_frozen_vlcm
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"
module load anaconda3
set +u
source activate "${CONDA_ENV:-internvl}"
set -u
exec bash "$PROJECT_ROOT/train_frozen_python_value.sh" "$@"
