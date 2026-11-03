#!/usr/bin/env bash
#SBATCH --job-name=vlcm_joint
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --gpus-per-node=a100:2
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
export FORMATION_DATASET="${FORMATION_DATASET:-/home/anshuln/Dataset/polygon_formation_python_env}"
export GOTOGOAL_DATASET="${GOTOGOAL_DATASET:-/home/anshuln/Dataset/manyagent_gotogoal_python_2000}"
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-/home/anshuln/MA_VLCM/MA-VLCM/outputs/smolvlm-18918211/best.pt}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    module load anaconda3
    set +u
    source activate "${CONDA_ENV:-internvl}"
    set -u
fi
exec bash "$PROJECT_ROOT/train_multitask_frozen.sh" "$@"
