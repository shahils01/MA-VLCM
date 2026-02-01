python train.py \
  --train_shards "/scratch/shahils/data/wds_gotogoal/shard-{000000..000080}.tar" \
  --batch_size 2 \
  --clip_len 10 \
  --clip_stride 10 \
  --robot_source obs \
  --reward_reduce mean \
  --done_reduce any \
  --gamma 0.99 \
  --text_mode raw \
  --return_mode nstep \
  --n_step 10 \
  --freeze_vl \
  --vl_backend llava_video \
  --vl_model_name llava-hf/LLaVA-NeXT-Video-7B-hf \
  --text_prompt_template "<video><obs>You are a critic model. You are given video of a tean of robots \
  (denoted as circular dots with heading denoted by an arrow). The goal for each robot is denoted by the \
  same color square box. The robots have to go to their designated goal without colliding with one another. \
  They also have to be efficient by taking the shortest parth. How Good or Bad are the team of robots doing \
  to accomplish the given task? Also tell me why and what you see. Keep your answer short." \
  --peft lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
