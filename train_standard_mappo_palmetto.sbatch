#!/usr/bin/env bash
#SBATCH --job-name=mappo_mlp_8m
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=16gb
#SBATCH --array=0-2%3
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Three independent tasks, up to three A100s total. No workload on login nodes.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Submit with sbatch train_standard_mappo_palmetto.sbatch [--check]. Checks also require an allocation.' >&2
  exit 2
fi
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export VLCM_CODE_ROOT="$PROJECT_ROOT"
export MARL_ROOT="${MARL_ROOT:-$(dirname "$PROJECT_ROOT")/MARL_data_collection}"
export DATASET_ROOT="${DATASET_ROOT:-/home/${USER}/Dataset}"
export VLCM_FORMATION_ROOT="${VLCM_FORMATION_ROOT:-$PROJECT_ROOT/outputs/frozen_joint_runtime_assets/formation}"
export VLCM_NAVIGATION_ROOT="${VLCM_NAVIGATION_ROOT:-$PROJECT_ROOT/outputs/frozen_joint_runtime_assets/navigation_without_obstacles}"
export RUN_ROOT="${RUN_ROOT:-$MARL_ROOT/outputs/standard_mappo_palmetto_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}"
export FROZEN_POLICIES="${FROZEN_POLICIES:-$PROJECT_ROOT/outputs/frozen_joint_palmetto_20948172}"
export NUM_ENV_STEPS="${NUM_ENV_STEPS:-8000000}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DISABLE_GIT=true WANDB_DISABLE_CODE=true PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ -z "${PYTHON_BIN:-}" ]]; then
  module load anaconda3
  set +u
  source activate "${CONDA_ENV:-internvl}"
  set -u
  export PYTHON_BIN="$(command -v python)"
fi
if [[ -d "$PROJECT_ROOT/outputs/training_dependencies" ]]; then
  export PYTHONPATH="$PROJECT_ROOT/outputs/training_dependencies${PYTHONPATH:+:$PYTHONPATH}"
fi
cd "$PROJECT_ROOT"
exec srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
  --gpus-per-task=a100:1 --gpu-bind=single:1 \
  bash "$PROJECT_ROOT/standard_mappo_palmetto_worker.sh" "${1:-train}"
