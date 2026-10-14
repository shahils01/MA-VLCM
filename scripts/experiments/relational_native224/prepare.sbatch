#!/bin/bash
#SBATCH --job-name=qwen_eval_prepare
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
python3 /scratch/aparame/VLCM/ma_vlcm/evaluations/prepare_relational_inference_20261014.py
