#!/bin/bash
#SBATCH --job-name=qwen_relational224_75
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=a100:2
#SBATCH --mem=128G
#SBATCH --time=48:00:00
# ALWAYS train with BOTH tb3-isaac-vlcm and tb3-isaac-avoid-obstacles-vlcm.
# User-requested 8-frame experiment; initialize fresh adapters/head/optimizer.
set -euo pipefail
RUN_ROOT=${1:?provide a new experiment run root}
test -f "$RUN_ROOT/UPLOAD_READY.json"
test -f "$RUN_ROOT/READY.json"
export BACKBONE_PROFILE=qwen3_vl
export CONTAINER_PATH=/home/aparame/Research/MA-VLCM/ma_vlcm.sif
export MA_VLCM_SCRATCH_ROOT=/scratch/aparame/ma_vlcm
export HF_HOME=/scratch/aparame/VLCM/ma_vlcm/cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export SAVE_DIR="$RUN_ROOT/checkpoints/qwen3_vl"
export WANDB_DIR="$RUN_ROOT/wandb/qwen3_vl"
export WANDB_RUN_PREFIX=relational_native224_2500_qwen3_vl_clip8_epoch75_20261012
export TASK_DOMAINS=goal_to_goal,static_obstacles TRAINED_AGENT_COUNTS=3,4,5,6
export LAYOUT_SPLIT=original_holdout_preserved_new_scenario_hash42
export TB3_BALANCE_MODE=domain BALANCE_TB3_SOURCES=1 SUCCESS_ONLY=0
export LOSS_TYPE=categorical_ce VALUE_DISTRIBUTION_BINS=21 VALUE_OUTPUT_ACTIVATION=identity
export PROGRESS_DISTANCE_MODE=euclidean TRAIN_FROM_SCRATCH=1
export RESUME_CHECKPOINT= DEFAULT_RESUME_CHECKPOINT=
# Full accuracy configuration. Override one knob per matched ablation.
export TB3_IMAGE_SIZE="${TB3_IMAGE_SIZE:-224}"
export CRITIC_READOUT="${CRITIC_READOUT:-query}"
export AGENT_POOLING="${AGENT_POOLING:-attention_max}"
export POTENTIAL_AUX_WEIGHT="${POTENTIAL_AUX_WEIGHT:-0.5}"
export TEMPORAL_PAIR_OFFSETS="${TEMPORAL_PAIR_OFFSETS:-1,2,4}"
export EVENT_SAMPLING_BOOST="${EVENT_SAMPLING_BOOST:-3}"
export EVENT_LOSS_BOOST="${EVENT_LOSS_BOOST:-3}"
export EVENT_DELTA_THRESHOLD=0.03
export CLIP_LEN=8 TOTAL_EPOCHS=75 SAMPLES_PER_EPOCH=5000 SPLIT_SEED=42 VAL_SPLIT=0.2
export VAL_SHARDS="$RUN_ROOT/data/validation"
export TEMPORAL_CONSISTENCY_WEIGHT="${TEMPORAL_CONSISTENCY_WEIGHT:-1.0}" CONDITIONAL_LIPSCHITZ_WEIGHT="${CONDITIONAL_LIPSCHITZ_WEIGHT:-1.0}"
export HEAD_LR=0.0003 BACKBONE_LR=0.00003 VISION_LR=0.00001
export NUM_WORKERS=4 OMP_NUM_THREADS=1 LOADER_PREFETCH_FACTOR=1 DISABLE_PERSISTENT_WORKERS=1
export BATCH_SIZE=32 GRAD_ACCUM_STEPS=4
export LORA_R=32 LORA_ALPHA=64 RELATIONAL_CRITIC=1
export NUM_PROCESSES=2
export ACCELERATE_DISTRIBUTED_ARGS="--multi_gpu --num_machines 1 --dynamo_backend no --main_process_port $((20000+${SLURM_JOB_ID:-0}%30000))"
export QWEN_LOGITS_TO_KEEP=1 DISABLE_GRADIENT_CHECKPOINTING=0
export APF_TARGET_TRANSFORM=causal_gaussian_sigma1_window5_collision_override
export PREFERENCE_LOSS_WEIGHT=0.5 PREFERENCE_INTERVAL=4 PREFERENCE_SHARDS="$RUN_ROOT/preferences"
if [ "${DRY_RUN:-0}" != 1 ]; then
    test -f "$RUN_ROOT/CHECKS_READY.json"
    test -f "$RUN_ROOT/PROFILE_READY.json"
    test -f "$RUN_ROOT/PREFERENCE_LOADER_READY.json"
fi
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export POTENTIAL_CONFIG="$RUN_ROOT/source/configs/potential_v1.json"
cd "$RUN_ROOT/source"
if [ "${DRY_RUN:-0}" != 1 ]; then
    test -f "$RUN_ROOT/CHECKS_READY.json"
    python3 - "$RUN_ROOT/PROFILE_READY.json" <<'CHECK'
import json,sys
p=json.load(open(sys.argv[1]));assert p["status"]=="passed" and p["num_processes"]==2 and p["strict_checkpoint_reload"]
assert all(x["peak_allocated_gib"]<.9*x["device_total_gib"] for x in p["ranks"])
CHECK
fi
bash scripts/lora_run_train_tb3_potential.sh "$RUN_ROOT/data/train"
