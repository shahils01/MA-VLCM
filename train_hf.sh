accelerate launch --num_processes 4 train.py \
  --data_backend webdataset \
  --train_shards "https://huggingface.co/datasets/shulnak09/OFFROAD_MARL/resolve/main/shard-{000000..000013}.tar" \
  --hf_streaming \
  --batch_size 2 \
  --epochs 500 \
  --clip_len 15 \
  --clip_stride 15 \
  --robot_source obs \
  --reward_reduce mean \
  --done_reduce any \
  --gamma 0.99 \
  --text_mode raw \
  --return_mode nstep \
  --n_step 5 \
  --vl_backend llava_video \
  --vl_model_name llava-hf/LLaVA-NeXT-Video-7B-hf \
  --peft lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --wandb \
  --wandb_project ma-vlcm \
  --wandb_run_name ma-vlcm-hf

  # --hf_dataset "https://huggingface.co/datasets/shulnak09/OFFROAD_MARL/resolve/main/shard-{000000..000014}.tar" \
