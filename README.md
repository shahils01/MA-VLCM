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
- `rewards.npy` (or `state.npy` if you use `--value_source state`)

The loader will group consecutive frames by episode id (parsed from `__key__` like `000000_000123`) and build clips of length `--clip_len`.

## Run
```
python train.py \
  --train_shards "/path/to/train/{00000..00099}.tar" \
  --val_shards "/path/to/val/{00000..00009}.tar" \
  --batch_size 4 \
  --clip_len 8 \
  --clip_stride 1 \
  --robot_source obs \
  --value_source rewards \
  --value_reduce mean \
  --value_time last \
  --text_mode raw \
  --epochs 2
```

## DeepSeek VLM choices
- Default: `--vl_backend deepseek_vl --vl_model_name deepseek-community/deepseek-vl-1.3b-base`
- For MoE backbone: use `--vl_backend deepseek_vl2` and a DeepSeek-VL2 model name (requires DeepSeek-VL2 repo)

## Notes
- If your `video.pth` tensors are already normalized for the DeepSeek image encoder, use `--video_preprocessed`.
- Your `.npy` payloads appear to be stored as raw arrays (not .npy files), so the loader reads them directly and falls back to `np.load`/`torch.load` only if needed.
