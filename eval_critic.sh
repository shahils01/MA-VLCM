# python eval_critic.py \
#   --eval_shards "/scratch/shahils/data/gotogoal_new_pt_225/shard-{000112..000268}.tar" \
#   --batch_size 2 \
#   --num_workers 4 \
#   --max_samples 512 \
#   --print_samples 40 \
#   --value_pooling last_token_logits \
#   --checkpoint checkpoints/ckpt_epoch_3.pt \



# For explicit good and bad trajectory sampling
python eval_critic.py \
  --checkpoint checkpoints/ckpt_epoch_4.pt \
  --good_shards "/scratch/shahils/data/gotogoal_new_pt_225/shard-{000112..000268}.tar" \
  --bad_shards "/scratch/shahils/data/gotogoal_new_pt_0/shard-{000000..000040}.tar" \
  --batch_size 2 \
  --num_workers 4 \
  --max_samples 512 \
  --print_samples 20
