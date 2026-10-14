#!/bin/bash
#SBATCH --job-name=qwen_full224_8f75
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=a100:1
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
export WANDB_RUN_PREFIX=full224_categorical21_signed_suboptimal500_qwen3_vl_clip8_epoch75_20261006
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
export QWEN_LOGITS_TO_KEEP=1 DISABLE_GRADIENT_CHECKPOINTING=0
# The short profiling dependency selects a measured batch/worker configuration
# with at least 20% peak-allocated VRAM headroom. No profiling weights are used.
if [ -f "$RUN_ROOT/PROFILE_READY.json" ]; then
    mapfile -t PROFILE_SETTINGS < <(python3 - "$RUN_ROOT/PROFILE_READY.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['clip_len']==8 and p['epochs']==75 and p['effective_batch_size']==128
assert p['image_size']==int(__import__('os').environ['TB3_IMAGE_SIZE'])
assert p['accuracy_configuration']=={name: __import__('os').environ[name] for name in ('CRITIC_READOUT','AGENT_POOLING','POTENTIAL_AUX_WEIGHT','TEMPORAL_PAIR_OFFSETS','EVENT_SAMPLING_BOOST','EVENT_LOSS_BOOST')}
assert p['batch_size'] in (32,64,128) and p['batch_size']*p['grad_accum_steps']==128
assert p['num_workers'] in (4,8) and p['qwen_logits_to_keep']==1
assert p['value_and_gradient_equivalence']['passed']
assert p['peak_allocated_gib'] < .80*p['device_total_gib']
for value in (p['batch_size'],p['grad_accum_steps'],p['num_workers'],int(p['disable_gradient_checkpointing'])):
    print(value)
PY
)
    test "${#PROFILE_SETTINGS[@]}" = 4
    export BATCH_SIZE="${PROFILE_SETTINGS[0]}" GRAD_ACCUM_STEPS="${PROFILE_SETTINGS[1]}"
    export NUM_WORKERS="${PROFILE_SETTINGS[2]}" DISABLE_GRADIENT_CHECKPOINTING="${PROFILE_SETTINGS[3]}"
elif [ "${DRY_RUN:-0}" != 1 ]; then
    echo 'Batch profiling must complete successfully before this fresh training run.' >&2
    exit 2
fi
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export POTENTIAL_CONFIG="$RUN_ROOT/source/configs/potential_v1.json"
cd "$RUN_ROOT/source"
if [ "${DRY_RUN:-0}" != 1 ]; then
    test -f "$RUN_ROOT/CHECKS_READY.json"
    python3 tools/finalize_qwen8_full_profile.py "$RUN_ROOT"
fi
bash scripts/lora_run_train_tb3_potential.sh "$RUN_ROOT/data/train"
