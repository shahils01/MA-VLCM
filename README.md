# MA-VLCM (Multimodal Value Model)

Backbone used here:
- **Vision/Text**: DeepSeek VLM (DeepSeek-VL by default; DeepSeek-VL2 optional)
- **Temporal**: Transformer encoder over frame embeddings
- **Graph**: dense adjacency message passing over robot nodes per timestep
- **Fusion**: MLP by default; optional MoE MLP

This is a minimal, clean starting point for video + text + robot state + dynamic graph, with all key sizes configurable.

## Install
```
pip install -r requirements.txt
```

For DeepSeek-VL2 (MoE backbone), follow the official DeepSeek-VL2 repo install steps and then use `--vl_backend deepseek_vl2`.

## WebDataset expected keys
Each **frame** sample should contain these keys:
- `image.png`
- `obs.npy` and/or `state.npy`
- `edge_index.npy`
- `caption.txt`
- `rewards.npy`
- `dones.npy`

The loader groups consecutive frames by episode id (parsed from `__key__` like `000000_000123`) and builds clips of length `--clip_len`.

## Hugging Face-hosted shards
If your data is stored on Hugging Face as `shard-*.tar`, use the same WebDataset loader by pointing `--train_shards` to HF `resolve` URLs.

## Value loss (TD)
We use TD(0) with reward from `rewards.npy`:
```
V_loss = MSE(V(s_t), r_t + gamma * (1 - done_t) * V(s_{t+1}))
```

## Run
```
python train.py \
  --train_shards "/path/to/train/{00000..00099}.tar" \
  --batch_size 4 \
  --clip_len 8 \
  --clip_stride 1 \
  --robot_source obs \
  --reward_reduce mean \
  --done_reduce any \
  --gamma 0.99 \
  --text_mode raw \
  --preprocess_in_loader \
  --wandb \
  --wandb_project ma-vlcm \
  --wandb_run_name local-debug \
  --epochs 2
```

### Run from Hugging Face-hosted WebDataset shards
```
python train.py \
  --train_shards "https://huggingface.co/datasets/your-org/your-dataset/resolve/main/shard-{000000..000099}.tar" \
  --batch_size 4 \
  --clip_len 8 \
  --clip_stride 1 \
  --robot_source obs \
  --reward_reduce mean \
  --done_reduce any \
  --gamma 0.99 \
  --text_mode raw \
  --preprocess_in_loader \
  --wandb \
  --wandb_project ma-vlcm \
  --wandb_run_name hf-debug \
  --epochs 2
```

## Weights & Biases tracking
- Login once on the machine: `wandb login`
- Enable logging with `--wandb`
- Useful options:
  - `--wandb_project ma-vlcm`
  - `--wandb_entity <team_or_user>`
  - `--wandb_run_name <name>`
  - `--wandb_tags ddp,llava,lora`

## Multi-GPU (FSDP)
To shard the model across GPUs (instead of full model replica per GPU), launch with Accelerate and enable FSDP:
```
accelerate launch --num_processes 4 train.py \
  --train_shards "/path/to/train/{00000..00099}.tar" \
  --batch_size 4 \
  --clip_len 8 \
  --clip_stride 1 \
  --robot_source obs \
  --reward_reduce mean \
  --done_reduce any \
  --gamma 0.99 \
  --text_mode raw \
  --preprocess_in_loader \
  --epochs 2 \
  --fsdp \
  --fsdp_min_num_params 1000000 \
  --wandb \
  --wandb_project ma-vlcm \
  --wandb_run_name fsdp-run
```
Notes:
- Use `--fsdp_cpu_offload` if GPU memory is tight (slower).
- `--fsdp_min_num_params` controls how aggressively submodules are wrapped/sharded.

## DeepSeek VLM choices
- Default: `--vl_backend deepseek_vl --vl_model_name deepseek-community/deepseek-vl-1.3b-base`
- For MoE backbone: use `--vl_backend deepseek_vl2` and a DeepSeek-VL2 model name (requires DeepSeek-VL2 repo)

## Notes
- Images are preprocessed using the DeepSeek VLM processor (resizes/crops to the expected size).
- If your `image.png` is already preprocessed for the model, set `--video_preprocessed` and feed tensors instead of PIL.
