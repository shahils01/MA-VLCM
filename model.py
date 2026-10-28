"""SmolVLM image-language features + framewise GAT + temporal value regression."""

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn

from gat import GraphAttentionEncoder


@dataclass
class ModelConfig:
    model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    model_revision: str = "main"
    backbone_mode: str = "frozen"  # frozen | lora | full
    vl_dtype: str = "bfloat16"
    vl_hidden_dim: int = 960
    vl_batch_size: int = 4
    image_size: int = 512
    robot_obs_dim: int = 10
    clip_len: int = 6
    d_model: int = 128
    gnn_layers: int = 1
    attention_heads: int = 4
    temporal_layers: int = 1
    dropout: float = 0.2
    lora_rank: int = 8


class SmolVLMBackbone(nn.Module):
    """Encode each image with its instruction; skip generation and vocabulary logits."""

    def __init__(self, cfg, device):
        super().__init__()
        from transformers import AutoProcessor, Idefics3Model

        self.cfg = cfg
        dtype = getattr(torch, cfg.vl_dtype) if device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(
            cfg.model_id, revision=cfg.model_revision, size={"longest_edge": cfg.image_size}
        )
        self.model = Idefics3Model.from_pretrained(
            cfg.model_id, revision=cfg.model_revision, torch_dtype=dtype,
            attn_implementation="sdpa",
        ).to(device)
        self.hidden_size = self.model.config.text_config.hidden_size
        self.frozen = cfg.backbone_mode == "frozen"
        self.model.requires_grad_(cfg.backbone_mode == "full")
        if cfg.backbone_mode == "lora":
            from peft import LoraConfig, get_peft_model

            self.model = get_peft_model(self.model, LoraConfig(
                r=cfg.lora_rank, lora_alpha=2 * cfg.lora_rank, lora_dropout=0.05,
                target_modules=r"text_model\..*\.(q_proj|v_proj)$",
            ))
        if not self.frozen:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.train(not self.frozen)

    def train(self, mode=True):
        return super().train(mode and not self.frozen)

    def forward(self, images, prompts):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        features = []
        for start in range(0, len(images), self.cfg.vl_batch_size):
            stop = start + self.cfg.vl_batch_size
            texts = [self.processor.apply_chat_template([
                {"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": prompt}
                ]}
            ], add_generation_prompt=True) for prompt in prompts[start:stop]]
            # A nested image list denotes one image per example, rather than a multi-image example.
            inputs = self.processor(
                text=texts, images=[[image] for image in images[start:stop]],
                padding=True, return_tensors="pt",
            )
            inputs = {
                key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
                for key, value in inputs.items()
            }
            with torch.no_grad() if self.frozen else nullcontext():
                output = self.model(**inputs, use_cache=False, return_dict=True)
                # The final non-padding token attends to the image and all instruction tokens.
                positions = torch.arange(inputs["input_ids"].shape[1], device=device)
                last = (inputs["attention_mask"] * positions).amax(-1)
                pooled = output.last_hidden_state[torch.arange(len(last), device=device), last]
            features.append(pooled.float())
        return torch.cat(features)


class MultimodalValueModel(nn.Module):
    def __init__(self, cfg, device=torch.device("cpu"), load_backbone=True):
        super().__init__()
        self.cfg = cfg
        self.backbone = SmolVLMBackbone(cfg, device) if load_backbone else None
        self.register_buffer("obs_mean", torch.zeros(cfg.robot_obs_dim))
        self.register_buffer("obs_std", torch.ones(cfg.robot_obs_dim))
        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_std", torch.tensor(1.0))
        self.visual = nn.Sequential(nn.LayerNorm(cfg.vl_hidden_dim), nn.Linear(cfg.vl_hidden_dim, cfg.d_model))
        self.graph = GraphAttentionEncoder(
            cfg.robot_obs_dim, cfg.d_model, cfg.gnn_layers, cfg.attention_heads, cfg.dropout
        )
        self.fuse = nn.Sequential(nn.Linear(2 * cfg.d_model, cfg.d_model), nn.GELU())
        self.time_embedding = nn.Sequential(nn.Linear(1, cfg.d_model), nn.Tanh())
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.attention_heads, dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, cfg.temporal_layers, enable_nested_tensor=False)
        self.value_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(), nn.Linear(cfg.d_model, 1),
        )

    def forward(self, batch):
        features = batch.get("vl_features")
        if features is None:
            b, t = batch["frame_mask"].shape
            images = [image for clip in batch["images"] for image in clip]
            prompts = [prompt for prompt in batch["prompt"] for _ in range(t)]
            features = self.backbone(images, prompts).reshape(b, t, -1)
        observations = (batch["robot_obs"] - self.obs_mean) / self.obs_std
        graph = self.graph(observations, batch["adj"], batch["robot_mask"])
        visual = self.visual(features.float())
        fused = self.fuse(torch.cat((visual, graph), dim=-1))
        fused = fused + self.time_embedding(batch["times"].unsqueeze(-1))
        temporal = self.temporal(fused, src_key_padding_mask=~batch["frame_mask"])
        # Every clip contains only observations up to its anchor, which is always the last slot.
        normalized = self.value_head(temporal[:, -1]).squeeze(-1)
        return normalized * self.target_std + self.target_mean

    def checkpoint_state(self):
        if self.cfg.backbone_mode == "full":
            return {name: value.detach().cpu() for name, value in self.state_dict().items()}
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        buffers = {name for name, _ in self.named_buffers() if not name.startswith("backbone.")}
        return {
            name: value.detach().cpu() for name, value in self.state_dict().items()
            if name in trainable or name in buffers
        }
