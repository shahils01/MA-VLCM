"""Run inside the allocated GPU step before launching a long LoRA job."""
import os
from PIL import Image
import torch
from model import ModelConfig, SmolVLMBackbone

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("LoRA smoke test needs exactly one visible CUDA GPU")
cfg = ModelConfig(model_id=os.environ.get("MODEL_ID", ModelConfig.model_id),
                  model_revision=os.environ.get("MODEL_REVISION", "main"),
                  backbone_mode="lora", vl_batch_size=1, image_size=512)
backbone = SmolVLMBackbone(cfg, torch.device("cuda"))
features = backbone([Image.new("RGB", (256, 256), "white")], ["Guide robots to their goals."])
if not torch.isfinite(features).all():
    raise RuntimeError("LoRA smoke test produced non-finite features")
features[:, 0].sum().backward()
trainable = [(name, p) for name, p in backbone.named_parameters() if p.requires_grad]
if not trainable or any("lora_" not in name for name, _ in trainable):
    raise RuntimeError("Unexpected trainable backbone parameters")
if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum().item() > 0
           for _, p in trainable):
    raise RuntimeError("LoRA smoke test found no finite nonzero adapter gradient")
print(f"LoRA GPU smoke passed: {len(trainable)} trainable adapter tensors", flush=True)
