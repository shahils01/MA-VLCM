#!/usr/bin/env bash
#SBATCH --job-name=score_frozen_latest
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=24gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch; compute allocation required' >&2; exit 2; }
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export VLCM_CODE_ROOT="$PROJECT_ROOT"
export MARL_ROOT="${MARL_ROOT:-$PROJECT_ROOT/outputs/mappo_comparison_20261119/code/MARL_data_collection}"
export VLCM_FORMATION_ROOT="$PROJECT_ROOT/outputs/frozen_joint_runtime_assets/formation"
export VLCM_NAVIGATION_ROOT="$PROJECT_ROOT/outputs/frozen_joint_runtime_assets/navigation_without_obstacles"
export PYTHONPATH="$PROJECT_ROOT/outputs/training_dependencies${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec srun --nodes=1 --ntasks=1 --cpus-per-task=4 --gpus-per-task=a100:1 --gpu-bind=single:1 \
  "${PYTHON_BIN:-$HOME/.conda/envs/internvl/bin/python}" "$MARL_ROOT/evaluate_frozen_joint_policies.py" score \
  --policies "$PROJECT_ROOT/outputs/frozen_joint_palmetto_20948172" --checkpoint latest \
  --output "${EVAL_OUTPUT:?Set EVAL_OUTPUT to the transferred rollout folder}" \
  --critic "$PROJECT_ROOT/outputs/critic35_sequential/joint_with_obstacles/best.pt" \
  --cases navigation_without_obstacles formation_without_obstacles formation_physical_obstacles navigation_with_obstacles \
  --device cuda:0 --vision-batch 4
