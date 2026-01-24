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
Each sample should contain these tensors (depending on `--text_mode`):
- `video.pth`: `[T, C, H, W]`
- `robot_obs.pth`: `[T, N, obs_dim]`
- `adj.pth`: `[T, N, N]`
- `text_emb.pth`: `[text_dim]` (when `--text_mode emb`)
- `text.txt`: raw text (when `--text_mode raw`)
- `value.pth`: `[]` or `[1]` scalar value

## Run
```
python train.py \
  --train_shards "/path/to/train/{00000..00099}.tar" \
  --val_shards "/path/to/val/{00000..00009}.tar" \
  --batch_size 4 \
  --epochs 2
```

## DeepSeek VLM choices
- Default: `--vl_backend deepseek_vl --vl_model_name deepseek-community/deepseek-vl-1.3b-base`
- For MoE backbone: use `--vl_backend deepseek_vl2` and a DeepSeek-VL2 model name (requires DeepSeek-VL2 repo)

## Notes
- If your `video.pth` tensors are already normalized for the DeepSeek image encoder, use `--video_preprocessed`.
