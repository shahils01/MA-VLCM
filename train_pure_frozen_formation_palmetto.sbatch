#!/usr/bin/env bash
#SBATCH --job-name=pure_frozen_form_s1
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Submit this script using sbatch; all checks require a compute allocation.' >&2
  exit 2
fi
export PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the isolated MA-VLCM snapshot}"
export RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to a new experiment directory}"
export VLCM_CHECKPOINT="${VLCM_CHECKPOINT:-/home/$USER/MA_VLCM/MA-VLCM/outputs/smolvlm-18918211/best.pt}"
export PYTHON_BIN="${PYTHON_BIN:-/home/$USER/.conda/envs/internvl/bin/python}"
export SEED=1 NUM_ENVS=16 ROLLOUT_STEPS=300
export NUM_ENV_STEPS="${NUM_ENV_STEPS:-8000000}"
export SAVE_INTERVAL=5 KEEP_CHECKPOINTS=3
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT=formation-mappo-vlcm
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WANDB_DISABLE_GIT=true WANDB_DISABLE_CODE=true
cd "$PROJECT_ROOT"
exec srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  --gpus-per-task=a100:1 --gpu-bind=single:1 \
  bash "$PROJECT_ROOT/pure_frozen_formation_worker.sh"
