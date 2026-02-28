#!/usr/bin/env bash
set -euo pipefail

# Override these with environment variables if needed.
TRAIN_SHARDS="/scratch/shahils/data/gotogoal_new_pt_225/shard-{000000..000260}.tar"
SCENARIO="${SCENARIO:-ManyAgentGoToGoal-v0}"
SAVE_DIR="${SAVE_DIR:-checkpoints_irl_local}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
VL_MODEL_NAME="${VL_MODEL_NAME:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}"  # 0.5B default

accelerate launch --num_processes "${NUM_PROCESSES}" train_irl_local_policy.py \
  --scenario "${SCENARIO}" \
  --train_shards "${TRAIN_SHARDS}" \
  --num_envs 4 \
  --seed 1 \
  --rollout_steps 128 \
  --rollout_buffer_steps 4096 \
  --iters 2000 \
  --clip_len 10 \
  --critic_updates 2 \
  --actor_updates 1 \
  --policy_batch_size 8 \
  --expert_batch_size 2 \
  --num_workers 2 \
  --entropy_coef 0.001 \
  --score_scale 1.0 \
  --critic_lr 3e-5 \
  --actor_lr 1e-4 \
  --vl_backend llava_video \
  --vl_model_name "${VL_MODEL_NAME}" \
  --vl_dtype bfloat16 \
  --robot_obs_dim 40 \
  --d_model 256 \
  --temporal_layers 2 \
  --temporal_heads 4 \
  --temporal_dropout 0.1 \
  --gnn_layers 2 \
  --fusion_hidden 512 \
  --policy_hidden_dim 256 \
  --action_type continuous \
  --action_dim 2 \
  --log_every 20 \
  --save_every 200 \
  --save_dir "${SAVE_DIR}" \
  --text_prompt_template "You are a critic model. The video of a team of robots (denoted as circular dots \
  with heading denoted by an arrow) is: <video>, and the robot team observations over ten episodes is <obs>. \
  The goal for each robot is denoted by the same color square box. The robots have to go to their designated goal \
  without colliding with one another. They also have to be efficient by taking the shortest path. How good or bad are \
  the team of robots doing to accomplish the given task?"
