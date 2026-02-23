python3 train.py \
  --train_shards "/home/anshul/Research/Postdoc/RL/MA-VLCM/test_data/shard-{000000..000004}.tar" \
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
  --vl_backend llava_video \
  --vl_model_name /workspace/models \
  --text_prompt_template "You are a critic model. The video of a team of robots (denoted as circular dots\
  with heading denoted by an arrow) is: <video> and the robot team observations over ten episodes is <obs>.\
  The goal for each robot is denoted by the same color square box. The robots have to go to their designated goal \
  without colliding with one another. They also have to be efficient by taking the shortest parth. How Good or Bad are \
  the team of robots doing to accomplish the given task? Also tell me why and what you see. Keep your answer short." \
  --peft lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --wandb \
  --wandb_project ma-vlcm \
  --wandb_run_name ma-vlcm-single
