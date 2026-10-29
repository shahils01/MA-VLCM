#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/anshul/miniconda3/envs/env_isaaclab/bin/python}"
CRITIC="${CRITIC:-vlcm}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/outputs/mappo_${CRITIC}_seed${SEED:-1}_$(date +%Y%m%d_%H%M%S)}"
transfer_args=()
if [[ "$CRITIC" != "mlp" ]]; then
    transfer_args+=(--critic-checkpoint "${VLCM_CHECKPOINT:-${SCRIPT_DIR}/outputs/zero_margin_900/best.pt}")
fi
# Optional warm-start of the successful actor; omit ACTOR_CHECKPOINT for scratch.
if [[ -n "${ACTOR_CHECKPOINT:-}" ]]; then
    transfer_args+=(--actor-checkpoint "$ACTOR_CHECKPOINT")
fi
exec "$PYTHON_BIN" "$SCRIPT_DIR/train_mappo.py" \
    --critic "$CRITIC" --output-dir "$RUN_DIR" --seed "${SEED:-1}" \
    --num-envs "${NUM_ENVS:-16}" --num-env-steps "${NUM_ENV_STEPS:-2000000}" \
    --rollout-steps 300 --agent-counts 3 4 5 6 \
    --formation-spacing 1.0 --formation-tolerance 0.05 \
    --initial-position-range 1.5 --min-start-separation 0.55 --observation-scale 3.0 \
    --control-dt 0.05 --camera-stride 4 --clip-len 6 \
    --view-size-m 6.096 --tile-floor --image-size "${IMAGE_SIZE:-256}" \
    --gamma 0.99 --gae-lambda 0.95 --ppo-epochs 5 --critic-epochs 5 \
    --actor-lr "${ACTOR_LR:-0.0005}" --critic-lr "${CRITIC_LR:-0.0005}" \
    --save-interval "${SAVE_INTERVAL:-2}" --keep-checkpoints "${KEEP_CHECKPOINTS:-2}" \
    --wandb-project "${WANDB_PROJECT:-formation-mappo-vlcm}" \
    --wandb-mode "${WANDB_MODE:-online}" "${transfer_args[@]}" "$@"
