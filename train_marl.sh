#!/usr/bin/env bash
set -euo pipefail

CRITIC_CHECKPOINT="${CRITIC_CHECKPOINT:-/path/to/ma_vlcm_checkpoint.pt}"
ENV_REPO="${ENV_REPO:-/Users/shahilshaik/Bayesian-Trust-Modeling}"
SCENARIO="${SCENARIO:-ManyAgentGoToGoal-v0}"
SAVE_DIR="${SAVE_DIR:-checkpoints_marl_frozen_critic}"
DEVICE="${DEVICE:-cuda}"

python train_marl_frozen_critic.py \
  --critic_checkpoint "${CRITIC_CHECKPOINT}" \
  --env_repo "${ENV_REPO}" \
  --scenario "${SCENARIO}" \
  --env_kwargs '{"num_agents": 5, "max_steps": 100, "show_communication": false}' \
  --device "${DEVICE}" \
  --num_envs 4 \
  --iters 400 \
  --rollout_steps 128 \
  --clip_len 12 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --ppo_epochs 4 \
  --mini_batch_size 128 \
  --actor_lr 3e-4 \
  --policy_hidden_dim 128 \
  --policy_video_source env \
  --frame_store_size 224 \
  --critic_batch_size 8 \
  --normalize_advantages \
  --eval_interval 25 \
  --eval_episodes 5 \
  --save_every 50 \
  --save_dir "${SAVE_DIR}"
