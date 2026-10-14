#!/bin/bash
#SBATCH --job-name=qwen_relational_eval
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gpus=a100:1
# Qwen only; matched held-out episodes from BOTH GTG and avoid-obstacles datasets.
set -euo pipefail
EVAL_ROOT=/scratch/aparame/VLCM/ma_vlcm/evaluations/qwen_relational_native224_best_vs75_20261014
test -f "$EVAL_ROOT/PREPARATION_COMPLETE.json"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export MPLCONFIGDIR="$EVAL_ROOT/mpl_cache"
mkdir -p "$MPLCONFIGDIR"
python3 "$EVAL_ROOT/run_evaluation.py"
