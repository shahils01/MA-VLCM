#!/usr/bin/env bash
# Standalone supervised critic experiment; independent of the LoRA/MAPPO jobs.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
if (( $# )); then
    echo "Configure this launcher with environment variables; see README_FROZEN_2000.md" >&2
    exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-$PROJECT_ROOT/../Ground_Robot_MARL/datasets/formation_vlcm_polygon_3to6_10hz_inner00m}"
EXTERNAL_DATASET="${EXTERNAL_DATASET:-$PROJECT_ROOT/../Ground_Robot_MARL/datasets/formation_validation_seed7300001}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/outputs/frozen_2000_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/outputs/cache_frozen_2000}"
case "${FROZEN_PROFILE:-baseline900}" in
    baseline900) default_seed=43; default_hz=5; default_clip=6 ;;
    lora) default_seed=42; default_hz=10; default_clip=8 ;;
    *) echo "FROZEN_PROFILE must be baseline900 or lora" >&2; exit 2 ;;
esac
SEED="${FROZEN_SEED:-$default_seed}"
IMAGE_HZ="${FROZEN_IMAGE_HZ:-$default_hz}"
CLIP_LEN="${FROZEN_CLIP_LEN:-$default_clip}"
if [[ -e "$RUN_DIR" ]]; then
    echo "Choose a new RUN_DIR; refusing to overwrite $RUN_DIR" >&2
    exit 1
fi
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" "$PROJECT_ROOT/prepare_frozen_2000.py" \
    --dataset "$DATASET" --external-dataset "$EXTERNAL_DATASET" \
    --cache-dir "$CACHE_DIR" --output "$RUN_DIR/preflight.json" \
    --seed "$SEED" --gamma 0.99 --image-hz "$IMAGE_HZ"
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
    exit 0
fi
# Explicit overrides prevent inherited LoRA environment variables changing this run.
export DATASET RUN_DIR CACHE_DIR PYTHON_BIN
export BACKBONE_MODE=frozen NUM_GPUS=1 RETURN_SOURCE=control
export WANDB_RUN_NAME="${FROZEN_RUN_NAME:-smolvlm-frozen-2000-${SLURM_JOB_ID:-local}}"
export BATCH_SIZE="${FROZEN_BATCH_SIZE:-64}" VL_BATCH_SIZE="${FROZEN_VL_BATCH_SIZE:-4}"
export GRAD_ACCUM_STEPS=1 MIXED_PRECISION=bf16
export EPOCHS="${FROZEN_EPOCHS:-30}" PATIENCE="${FROZEN_PATIENCE:-6}"
export MODEL_ID=HuggingFaceTB/SmolVLM-500M-Instruct
export MODEL_REVISION="${FROZEN_MODEL_REVISION:-a7da5b986cb59b408707209984f360a5f4ad7e47}"
bash "$PROJECT_ROOT/train.sh" \
    --backbone-mode frozen --return-source control --policy-stage all --max-episodes 0 \
    --image-size 128 --image-hz "$IMAGE_HZ" --clip-len "$CLIP_LEN" --clip-stride 1 \
    --seed "$SEED" --gamma 0.99 --lr "${FROZEN_LR:-0.0001}" \
    --d-model 128 --gnn-layers 1 --temporal-layers 1 --attention-heads 4 --dropout 0.2 \
    --validation-fraction 0.1 --test-fraction 0.1 \
    --checkpoint-every "${FROZEN_CHECKPOINT_EVERY:-2}" \
    --checkpoint-every-steps "${FROZEN_CHECKPOINT_EVERY_STEPS:-0}" --log-every-steps 50
# Select best.pt using internal validation only, then score the separate 60 once.
"$PYTHON_BIN" "$PROJECT_ROOT/evaluate.py" \
    --checkpoint "$RUN_DIR/best.pt" --dataset "$EXTERNAL_DATASET" \
    --split external --expected-episodes 60 --cache-dir "$CACHE_DIR" \
    --output-dir "$RUN_DIR/external60" --batch-size "$BATCH_SIZE"
echo "Finished frozen 2000-episode experiment: $RUN_DIR"
