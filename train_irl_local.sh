#!/usr/bin/env bash
set -euo pipefail

# Override these with environment variables if needed.
TRAIN_SHARDS="/scratch/shahils/data/gotogoal_pt_225/shard-{000000..000111}.tar"
SCENARIO="${SCENARIO:-ManyAgentGoToGoal-v0}"
SAVE_DIR="${SAVE_DIR:-checkpoints_irl_local}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
VL_MODEL_NAME="${VL_MODEL_NAME:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}"  # llava-onevision-qwen2-0.5b-ov-hf or LLaVA-NeXT-Video-7B-hf
WANDB_PROJECT="${WANDB_PROJECT:-ma-vlcm-irl}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-irl-local}"
PEFT_MODE="${PEFT_MODE:-qlora}" # none / lora / qlora

MPLBACKEND=Agg accelerate launch --num_processes "${NUM_PROCESSES}" train_irl_local_policy.py \
  --scenario "${SCENARIO}" \
  --train_shards "${TRAIN_SHARDS}" \
  --num_envs 1 \
  --seed 1 \
  --rollout_steps 128 \
  --rollout_buffer_steps 4096 \
  --policy_video_source env \
  --frame_store_size 660 \
  --iters 2000 \
  --clip_len 15 \
  --critic_updates 1 \
  --actor_updates 1 \
  --policy_batch_size 4 \
  --expert_batch_size 4 \
  --num_workers 2 \
  --entropy_coef 0.001 \
  --expert_done_reduce all \
  --expert_sanity_batches 3 \
  --score_scale 1.0 \
  --disc_tanh_temp 20.0 \
  --raw_score_l2_coef 1e-4 \
  --lambda_feat_contrastive 0.1 \
  --feat_contrastive_margin 1.0 \
  --critic_lr 1e-4 \
  --actor_lr 1e-4 \
  --critic_grad_clip 1.0 \
  --mixed_precision bf16 \
  --grad_accum_steps 64 \
  --ddp_find_unused_parameters \
  --vl_backend llava_video \
  --vl_model_name "${VL_MODEL_NAME}" \
  --vl_dtype bfloat16 \
  --obs_summary_tokens 2 \
  --robot_obs_dim 40 \
  --d_model 256 \
  --temporal_layers 2 \
  --temporal_heads 4 \
  --temporal_dropout 0.1 \
  --gnn_layers 2 \
  --fusion_hidden 512 \
  --peft "${PEFT_MODE}" \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,v_proj" \
  --policy_hidden_dim 256 \
  --action_type continuous \
  --action_dim 2 \
  --log_every 5 \
  --eval_interval 25 \
  --eval_episodes 5 \
  --eval_max_episode_steps 500 \
  --save_every 20 \
  --save_dir "${SAVE_DIR}" \
  --wandb \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --text_prompt_template "You are a critic model. The video of a team of robots (denoted as circular dots \
  with heading denoted by an arrow) is: <video>, and the robot team observations over ten episodes is <obs>. \
  The goal for each robot is denoted by the same color square box. The robots have to go to their designated goal \
  without colliding with one another. They also have to be efficient by taking the shortest path. How good or bad are \
  the team of robots doing to accomplish the given task?"
