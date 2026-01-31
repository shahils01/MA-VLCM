accelerate launch --num_processes 2 train.py \
  --train_shards "/scratch/shahils/data/wds_gotogoal/shard-{000000..000080}.tar" \
  --batch_size 2 \
  --clip_len 10 \
  --clip_stride 10 \
  --robot_source obs \
  --reward_reduce mean \
  --done_reduce any \
  --gamma 0.99 \
  --text_mode raw \
  --return_mode td \
  --n_step 50 \
  --vl_backend llava_video \
  --vl_model_name llava-hf/LLaVA-NeXT-Video-7B-hf \
  --text_prompt_template "You are a critic model. You are given video frames, robot state sequences, and a graph adjacency per timestep for a robot team. Assess how good or bad the current policy is at the task and respond with a single scalar judgment." \
  --fsdp \
  --fsdp_min_num_params 1000000 \
  --mixed_precision fp16

