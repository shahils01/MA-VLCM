accelerate launch --num_processes 2 train.py \
  --train_shards "/scratch/shahils/data/wds_gotogoal/shard-{000000..000080}.tar" \
  --batch_size 2 \
  --epochs 500 \
  --clip_len 5 \
  --clip_stride 5 \
  --robot_source obs \
  --reward_reduce mean \
  --done_reduce any \
  --gamma 0.99 \
  --text_mode raw \
  --return_mode nstep \
  --n_step 5 \
  --vl_backend llava_video \
  --vl_model_name llava-hf/LLaVA-NeXT-Video-7B-hf \
  --text_prompt_template "You are a critic model. The video of a team of robots (denoted as circular dots\
  with heading denoted by an arrow) is: <video>, and the robot team observations over ten episodes is <obs>.\
  The goal for each robot is denoted by the same color square box. The robots have to go to their designated goal \
  without colliding with one another. They also have to be efficient by taking the shortest parth. How Good or Bad are \
  the team of robots doing to accomplish the given task?" \
  --peft lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --wandb \
  --wandb_project ma-vlcm \
  --wandb_run_name ma-vlcm-ddp
