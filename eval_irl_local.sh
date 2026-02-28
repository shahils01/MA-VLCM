#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-checkpoints_irl_local/irl_local_iter_600.pt}"
SCENARIO="${SCENARIO:-ManyAgentGoToGoal-v0}"

python eval_irl_local_policy.py \
  --checkpoint "${CHECKPOINT}" \
  --scenario "${SCENARIO}" \
  --eval_episodes 3 \
  --max_episode_steps 500 \
  --action_type continuous \
  --action_dim 2 \
  --save_video \
  --video_path eval_outputs/irl_policy_eval.mp4

# For live rendering instead:
# python eval_irl_local_policy.py \
#   --checkpoint "${CHECKPOINT}" \
#   --scenario "${SCENARIO}" \
#   --eval_episodes 10 \
#   --max_episode_steps 500 \
#   --action_type continuous \
#   --action_dim 2 \
#   --render \
#   --render_mode human
