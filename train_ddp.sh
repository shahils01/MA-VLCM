# TRAIN_SHARDS="/scratch/shahils/data/gotogoal_new_pt_0/shard-{000000..000040}.tar::/scratch/shahils/data/gotogoal_new_pt_15/shard-{000000..000110}.tar::/scratch/shahils/data/gotogoal_new_pt_30/shard-{000000..000290}.tar::/scratch/shahils/data/gotogoal_new_pt_45/shard-{000000..000280}.tar::/scratch/shahils/data/gotogoal_new_pt_225/shard-{000000..000260}.tar"
TRAIN_SHARDS="/scratch/shahils/data/gotogoal_pt_0/shard-{000000..000280}.tar::/scratch/shahils/data/gotogoal_pt_15/shard-{000000..000250}.tar::/scratch/shahils/data/gotogoal_pt_30/shard-{000000..000150}.tar::/scratch/shahils/data/gotogoal_new_pt_45/shard-{000000..000120}.tar::/scratch/shahils/data/gotogoal_new_pt_225/shard-{000000..000110}.tar"
# TRAIN_SHARDS="/scratch/shahils/VLCM_Data_Collection/OFFROAD/dataset_2/shard-{000000..000030}.tar"
# TRAIN_SHARDS="/scratch/shahils/VLCM_Data_Collection/RWARE/rware:rware-tiny-2ag-hard-v2/2026-02-01/trajectory_{112930..113153}_success.tar"
VL_MODEL_PRESET="${VL_MODEL_PRESET:-llava_onevision_0p5b}"  # llava_onevision_0p5b / llava_next_video_7b

accelerate launch --num_processes 8 train.py \
  --train_shards $TRAIN_SHARDS \
  --batch_size 16 \
  --num_workers 1 \
  --grad_accum_steps 32 \
  --mixed_precision bf16 \
  --allow_tf32 \
  --value_pooling hidden_mean \
  --obs_summary_tokens 1 \
  --vl_logits_to_keep 128 \
  --epochs 500 \
  --clip_len 15 \
  --clip_stride 15 \
  --log_every 50 \
  --robot_source obs \
  --reward_reduce mean \
  --done_reduce all \
  --gamma 0.95 \
  --loss_type td_contrastive \
  --contrastive_multidepth \
  --contrastive_depth_offsets "0,2,4,8" \
  --contrastive_depth_weights "1.0, 0.99, 0.97, 0.95" \
  --text_mode raw \
  --return_mode nstep \
  --n_step 64 \
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
  --resume_checkpoint /scratch/shahils/MA-VLCM/checkpoints/ckpt_epoch_29.pt \
  # --disable_vl_cache \
  # --lora_r 16 \
  # --lora_alpha 32 \
  # --lora_dropout 0.05 \
  # --gradient_checkpointing \
  # --lora_target_modules "q_proj,v_proj" \

  # --text_prompt_template "You are a critic model. The video of a team of robots (denoted as circular dots\
  # with heading denoted by an arrow) is: <video>, and the robot team observations over ten episodes is <obs>.\
  # The goal for each robot is denoted by the same color square box. The robots have to go to their designated goal \
  # without colliding with one another. They also have to be efficient by taking the shortest parth. How Good or Bad are \
  # the team of robots doing to accomplish the given task?" \

  # --debug_decode_text \
  # --debug_decode_every 50 \
  # --debug_decode_max_tokens 256 \

  # --train_shards "/scratch/shahils/data/wds_gotogoal/shard-{000002..000080}.tar::/scratch/shahils/data/wds_gotogoal_bad/shard-{000000..000004}.tar" \
