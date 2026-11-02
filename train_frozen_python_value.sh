#!/usr/bin/env bash
# Actor from scratch; the supervised Python VLCM is the sole, frozen critic.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${VLCM_CHECKPOINT:-$ROOT/outputs/smolvlm-18918211/best.pt}"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/mappo_frozen_python_seed${SEED:-1}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
exec "$PYTHON_BIN" "$ROOT/train_mappo.py" \
  --critic vlcm --critic-checkpoint "$CHECKPOINT" \
  --freeze-critic --frozen-python-supervised --actor-from-scratch \
  --output-dir "$RUN_DIR" --seed "${SEED:-1}" \
  --num-envs "${NUM_ENVS:-4}" --num-env-steps "${NUM_ENV_STEPS:-200000}" \
  --rollout-steps "${ROLLOUT_STEPS:-100}" --agent-counts 3 4 5 6 \
  --formation-spacing .5 --formation-tolerance .075 \
  --initial-position-range 1.25 --min-start-separation .4 --observation-scale 3 \
  --formation-hold-steps 20 --env-max-steps 400 --control-dt .05 \
  --camera-stride 2 --clip-len 8 --view-size-m 3.048 --no-tile-floor --image-size 256 \
  --graph-radius 1 --gamma .99 --gae-lambda "${GAE_LAMBDA:-.95}" \
  --critic-warmup-updates 0 --actor-lr "${ACTOR_LR:-.0005}" \
  --save-interval "${SAVE_INTERVAL:-2}" --keep-checkpoints "${KEEP_CHECKPOINTS:-2}" \
  --wandb-project "${WANDB_PROJECT:-formation-mappo-vlcm}" \
  --wandb-mode "${WANDB_MODE:-online}" "$@"
