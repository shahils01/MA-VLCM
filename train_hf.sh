VL_MODEL_PRESET="${VL_MODEL_PRESET:-llava_onevision_0p5b}"  # llava_onevision_0p5b / llava_next_video_7b

accelerate launch --num_processes 2 train.py \
  --train_shards "https://huggingface.co/datasets/shulnak09/OFFROAD_MARL/tree/main/Varied_Traversability_new/*.tar" \
  --batch_size 2 \
  --num_workers 16 \
  --grad_accum_steps 16 \
  --mixed_precision bf16 \
  --allow_tf32 \
  --value_pooling last_token_logits \
  --vl_logits_to_keep 128 \
  --epochs 500 \
  --clip_len 20 \
  --clip_stride 20 \
  --log_every 50 \
  --robot_source obs \
  --reward_reduce mean \
  --done_reduce any \
  --gamma 0.99 \
  --loss_type td \
  --text_mode raw \
  --return_mode nstep \
  --n_step 20 \
  --vl_backend llava_video \
  --vl_model_preset $VL_MODEL_PRESET \
  --text_prompt_template "You are a critic model. The video of a team of robots (denoted as circular dots\
  with heading denoted by an arrow) is: <video>, and the robot team observations over ten episodes is <obs>.\
  The goal for each robot is denoted by the same color square box. The robots have to go to their designated goal \
  without colliding with one another. They also have to be efficient by taking the shortest parth. How Good or Bad are \
  the team of robots doing to accomplish the given task?" \
  --return_horizon trajectory \
  --wandb \
  --wandb_project ma-vlcm \
  --wandb_run_name ma-vlcm-ddp \
  --peft qlora \
  --resume_checkpoint checkpoints/ckpt_epoch_3.pt \
  --num_robots 5 \
  --robot_obs_dim 44 \
