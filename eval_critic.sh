VL_MODEL_PRESET="${VL_MODEL_PRESET:-llava_onevision_0p5b}"  # llava_onevision_0p5b / llava_next_video_7b / internvl3_5_{1b,2b,4b,8b}

accelerate launch --num_processes 1 eval_critic.py \
  --eval_shards "https://huggingface.co/datasets/shulnak09/OFFROAD_MARL/tree/main/Varied_Traversability_new/*.tar" \
  --batch_size 2 \
  --num_workers 2 \
  --max_samples 10 \
  --print_samples 10 \
  --value_pooling hidden_mean \
  --vl_model_preset "$VL_MODEL_PRESET" \
  --checkpoint checkpoints/ckpt_epoch_29.pt \
  --num_robots 3

# Alternative shards:
# --eval_shards "/scratch/shahils/data/gotogoal_pt_0/shard-{000281..000285}.tar::/scratch/shahils/data/gotogoal_pt_15/shard-{000251..000287}.tar::/scratch/shahils/data/gotogoal_pt_30/shard-{000151..000159}.tar::/scratch/shahils/data/gotogoal_pt_45/shard-{000121..000129}.tar"

# For explicit good and bad trajectory sampling
# python eval_critic.py \
#   --checkpoint checkpoints/ckpt_epoch_2.pt \
#   --vl_model_preset "$VL_MODEL_PRESET" \
#   --good_shards "/scratch/shahils/data/gotogoal_new_pt_225/shard-{000112..000268}.tar" \
#   --bad_shards "/scratch/shahils/data/gotogoal_new_pt_0/shard-{000000..000040}.tar" \
#   --batch_size 2 \
#   --num_workers 4 \
#   --max_samples 512 \
#   --print_samples 20
