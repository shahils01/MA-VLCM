# MA-VLCM (Multimodal Value Model)

Backbone used here:
- **Vision/Text/Video**: LLaVA-NeXT-Video (default)
- **Optional Vision/Text backbones**: DeepSeek-VL, DeepSeek-VL2
- **Temporal**: Transformer encoder over frame embeddings
- **Graph**: dense adjacency message passing over robot nodes per timestep
- **Fusion**: MLP by default; optional MoE MLP

This is a minimal, clean starting point for video + text + robot state + dynamic graph, with all key sizes configurable.

## Install
```
pip install -r requirements.txt
```

This now installs the LLaVA-NeXT-Video dependencies via `transformers>=4.46.0`.

Optional backbones:
- DeepSeek-VL2: see `requirements-vl2.txt` and the official DeepSeek-VL2 repo setup.
- DeepSeek-VL: install from its official repo if you want to run `--vl_backend deepseek_vl`.

Quick sanity check for LLaVA-NeXT-Video:
```
python -c "from transformers import LlavaNextVideoProcessor; from transformers.models.llava_next_video import LlavaNextVideoForConditionalGeneration; print('LLaVA-NeXT-Video ready')"
```

## WebDataset expected keys
Each **frame** sample should contain these keys:
- `image.png`
- `obs.npy` and/or `state.npy`
- `edge_index.npy`
- `caption.txt`
- `rewards.npy`
- `dones.npy`

The loader groups consecutive frames by episode id (parsed from `__key__` like `000000_000123`) and builds clips of length `--clip_len`.

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
  --vl_backend llava_video \
  --vl_model_name llava-hf/LLaVA-NeXT-Video-7B-32K-hf \
  --preprocess_in_loader \
  --epochs 2
```

## VLM backbone choices
- Default/recommended: `--vl_backend llava_video --vl_model_name llava-hf/LLaVA-NeXT-Video-7B-32K-hf`
- DeepSeek-VL: `--vl_backend deepseek_vl --vl_model_name deepseek-community/deepseek-vl-1.3b-base`
- DeepSeek-VL2: `--vl_backend deepseek_vl2` with a DeepSeek-VL2 model name (requires DeepSeek-VL2 repo install)

## Notes
- Images are preprocessed using the selected VLM processor (resizes/crops to the expected size).
- If your `image.png` is already preprocessed for the model, set `--video_preprocessed` and feed tensors instead of PIL.
