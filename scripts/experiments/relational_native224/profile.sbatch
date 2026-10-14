#!/bin/bash
#SBATCH --job-name=qwen_relational_2a100_profile
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=a100:2
#SBATCH --mem=128G
#SBATCH --time=01:00:00
set -euo pipefail
RUN_ROOT=${1:?run root}
test -f "$RUN_ROOT/PREFERENCE_LOADER_READY.json"
STAGE="$RUN_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export PYTHONPATH="$RUN_ROOT/source/src:$RUN_ROOT/source"
export HF_HOME=/scratch/aparame/VLCM/ma_vlcm/cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export MPLCONFIGDIR="$RUN_ROOT/profiling/mpl"
DRY_RUN=1 bash "$RUN_ROOT/train.sbatch" "$RUN_ROOT" > "$RUN_ROOT/dry_run.txt"
cd "$RUN_ROOT/source"
apptainer exec --nv -B /scratch:/scratch -B /home:/home \
    --env PYTHONPATH="$PYTHONPATH" --env HF_HOME="$HF_HOME" \
    --env HF_HUB_CACHE="$HF_HUB_CACHE" --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    --env PYTHONDONTWRITEBYTECODE=1 --env WANDB_MODE=disabled \
    /home/aparame/Research/MA-VLCM/ma_vlcm.sif python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 "$RUN_ROOT/profile_distributed.py" "$RUN_ROOT"
test -f "$RUN_ROOT/PROFILE_READY.json"
