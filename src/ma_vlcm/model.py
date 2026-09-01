import math
from dataclasses import dataclass
from typing import Any, Optional
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    # Backbone
    vl_backend: str = "llava_video"
    vl_model_name: str = "llava-hf/LLaVA-NeXT-Video-7B-32K-hf"
    vl_dtype: str = "bfloat16"  # float16 | bfloat16 | float32
    vl_max_text_len: int = 256
    freeze_vl: bool = False
    freeze_vision_tower: bool = False
    quantization_config: Optional[Any] = None

    # Video
    video_channels: int = 3
    video_frames: int = 8
    video_preprocessed: bool = False
    video_mean: tuple = (0.5, 0.5, 0.5)
    video_std: tuple = (0.5, 0.5, 0.5)

    # Robots / graph
    num_robots: int = 8
    robot_obs_dim: int = 16

    # Text (placeholder: supply pre-embedded text vectors)
    text_dim: int = 512
    task_domain_conditioning: bool = False
    num_task_domains: int = 3  # unknown, goal_to_goal, static_obstacles

    # Model dims
    d_model: int = 256
    temporal_layers: int = 2
    temporal_heads: int = 4
    temporal_dropout: float = 0.1

    # GNN
    gnn_layers: int = 4

    # Fusion
    fusion_hidden: int = 512
    use_moe: bool = False
    moe_experts: int = 4
    moe_top_k: int = 2

    # Debug
    debug_save_video: bool = False

    # Contrastive
    contrastive_multidepth: bool = False
    contrastive_depth_offsets: tuple = (0,)


class VisionLanguageBackbone(nn.Module):
    """Backbone wrapper for the supported Hugging Face video VLMs."""

    def __init__(self, cfg: ModelConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        if cfg.vl_dtype == "float16":
            dtype = torch.float16
        elif cfg.vl_dtype == "float32":
            dtype = torch.float32
        else:
            dtype = torch.bfloat16

        try:
            from transformers import (
                AutoProcessor,
                AutoTokenizer,
                LlavaNextVideoProcessor,
            )
            try:
                from transformers import LlavaOnevisionProcessor
            except ImportError:
                LlavaOnevisionProcessor = None

            try:
                from transformers.models.llava_next_video import (
                    LlavaNextVideoForConditionalGeneration,
                )
            except ImportError:
                LlavaNextVideoForConditionalGeneration = None
            
            try:
                from transformers.models.llava_onevision import (
                    LlavaOnevisionForConditionalGeneration,
                )
            except ImportError:
                try:
                    from transformers import LlavaOnevisionForConditionalGeneration
                except ImportError:
                    LlavaOnevisionForConditionalGeneration = None

            try:
                from transformers.models.auto.modeling_auto import (
                    AutoModelForVision2Seq,
                )
            except Exception:
                AutoModelForVision2Seq = None
        except Exception as e:
            raise ImportError(
                "The selected vision-language backend requires Transformers."
            ) from e

        if cfg.vl_backend == "qwen3_vl":
            try:
                from transformers import Qwen3VLForConditionalGeneration
            except ImportError as exc:
                raise ImportError(
                    "qwen3_vl requires a Transformers version containing "
                    "Qwen3VLForConditionalGeneration. Install the repository's "
                    "git-based Transformers requirement."
                ) from exc
            self.processor = AutoProcessor.from_pretrained(cfg.vl_model_name)
        elif cfg.vl_backend == "llava_onevision":
            if LlavaOnevisionProcessor is None:
                raise ImportError("LlavaOnevisionProcessor not available in your transformers version.")
            self.processor = LlavaOnevisionProcessor.from_pretrained(cfg.vl_model_name)
        else:
            self.processor = LlavaNextVideoProcessor.from_pretrained(cfg.vl_model_name)

        self.tokenizer = getattr(
            self.processor, "tokenizer", None
        ) or AutoTokenizer.from_pretrained(cfg.vl_model_name)
        if "<obs>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<obs>"]})

        model_kwargs = {"torch_dtype": dtype}
        if cfg.quantization_config is not None:
            model_kwargs["quantization_config"] = cfg.quantization_config

        # Load directly to target device to bypass CPU RAM buffering
        # This is safe because each process has its own GPU (accelerator.device)
        model_kwargs["device_map"] = {"": device}

        if cfg.vl_backend == "qwen3_vl":
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                cfg.vl_model_name, **model_kwargs
            )
        elif cfg.vl_backend == "llava_onevision":
            if LlavaOnevisionForConditionalGeneration is None:
                raise ImportError("LlavaOnevisionForConditionalGeneration not available.")
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                cfg.vl_model_name, **model_kwargs
            )
        else:
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                cfg.vl_model_name, **model_kwargs
            )

        if "<obs>" in self.tokenizer.get_vocab() and hasattr(
            self.model, "resize_token_embeddings"
        ):
            self.model.resize_token_embeddings(len(self.tokenizer))

        # self.model.to(device) # Handled by device_map
        if cfg.freeze_vl:
            # Freeze everything first
            for p in self.model.parameters():
                p.requires_grad = False
            # Optionally unfreeze the vision tower for fine-tuning
            if not cfg.freeze_vision_tower:
                vision_tower = self.get_vision_tower()
                if vision_tower is not None:
                    for p in vision_tower.parameters():
                        p.requires_grad = True
                    print(f"[ModelConfig] Vision tower UNFROZEN ({sum(p.numel() for p in vision_tower.parameters()):,} params)")
                else:
                    print("[ModelConfig] WARNING: Could not find vision_tower attribute; all VL params remain frozen.")

        self._dtype = dtype

    def get_vision_tower(self):
        """Return the visual encoder for optimizer grouping and visual LoRA."""
        for path in (
            ("base_model", "model", "model", "vision_tower"),
            ("base_model", "model", "model", "visual"),
            ("model", "model", "model", "vision_tower"),
            ("model", "model", "model", "visual"),
            ("base_model", "model", "vision_tower"),
            ("base_model", "model", "visual"),
            ("model", "model", "vision_tower"),
            ("model", "model", "visual"),
            ("model", "vision_tower"),
            ("model", "visual"),
            ("vision_tower",),
            ("visual",),
        ):
            module = self.model
            for attr in path:
                module = getattr(module, attr, None)
                if module is None:
                    break
            if module is not None:
                return module
        return None

    def get_input_embeddings(self):
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()
        if hasattr(self.model, "language_model") and hasattr(
            self.model.language_model, "get_input_embeddings"
        ):
            return self.model.language_model.get_input_embeddings()
        if hasattr(self.model, "model") and hasattr(
            self.model.model, "get_input_embeddings"
        ):
            return self.model.model.get_input_embeddings()
        raise AttributeError("Could not access input embeddings on the VLM backbone.")

    def _move_inputs_to_device(self, inputs):
        moved = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                if k in (
                    "pixel_values",
                    "pixel_values_videos",
                    "video_values",
                    "video",
                    "videos",
                ):
                    moved[k] = v.to(self.device, dtype=self._dtype)
                else:
                    moved[k] = v.to(self.device)
            else:
                moved[k] = v
        return moved

    def prepare_inputs(
        self, text, videos, padding=False, truncation=False, max_length=None
    ):
        # For LLaVA video prompts, truncation can break special token alignment.
        # Only pass max_length when truncation is explicitly enabled.
        if truncation and max_length is None:
            max_length = self.cfg.vl_max_text_len
        inputs = self.processor(
            text=text,
            videos=videos,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        return self._move_inputs_to_device(inputs)


class VJEPA2VideoBackbone(nn.Module):
    """Video-only V-JEPA2 encoder used as a compact critic representation."""

    def __init__(self, cfg: ModelConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        if cfg.vl_dtype == "float16":
            dtype = torch.float16
        elif cfg.vl_dtype == "float32":
            dtype = torch.float32
        else:
            dtype = torch.bfloat16

        try:
            from transformers import AutoVideoProcessor, VJEPA2Model
        except ImportError as exc:
            raise ImportError(
                "vjepa2 requires a Transformers version containing VJEPA2Model "
                "and AutoVideoProcessor."
            ) from exc

        self.processor = AutoVideoProcessor.from_pretrained(cfg.vl_model_name)
        model_kwargs = {"torch_dtype": dtype, "device_map": {"": device}}
        self.model = VJEPA2Model.from_pretrained(cfg.vl_model_name, **model_kwargs)
        self.hidden_size = int(self.model.config.hidden_size)
        self.tokenizer = None
        self._dtype = dtype

        if cfg.freeze_vl or cfg.freeze_vision_tower:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            # Match VLM freeze semantics: --freeze_vl freezes the backbone as
            # a starting point, while an unfrozen vision tower permits visual
            # full fine-tuning. V-JEPA2 consists entirely of that vision tower.
            if cfg.freeze_vl and not cfg.freeze_vision_tower:
                for parameter in self.model.parameters():
                    parameter.requires_grad = True

        # The critic uses encoder features with skip_predictor=True. Keeping the
        # unused JEPA prediction branch frozen avoids wasted optimizer state and
        # unused-parameter failures in full-finetune DDP runs.
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None:
            for parameter in predictor.parameters():
                parameter.requires_grad = False

    def get_vision_tower(self):
        return self.model

    def _move_inputs_to_device(self, inputs):
        moved = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if key == "pixel_values_videos":
                    moved[key] = value.to(self.device, dtype=self._dtype)
                else:
                    moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved


class TemporalTransformer(nn.Module):
    def __init__(self, d_model: int, layers: int, heads: int, dropout: float):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        t = x.size(1)
        pos = self.pos_emb[:, :t, :]
        x = self.dropout(x + pos)
        return self.encoder(x)


class RobotEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class DenseGraphEncoder(nn.Module):
    """Simple dense adjacency message passing. adj is [B, T, N, N]."""

    def __init__(self, d_model: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(layers)]
        )

    def forward(self, node_feats, adj):
        # node_feats: [B, T, N, D]
        # adj: [B, T, N, N]
        h = node_feats
        for layer in self.layers:
            # normalize adjacency to avoid scale blowup
            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
            adj_norm = adj / deg
            agg = torch.matmul(adj_norm, h)  # [B, T, N, D]
            h = F.gelu(layer(agg))
        return h


class MoEFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, experts: int, top_k: int):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, d_model),
                )
                for _ in range(experts)
            ]
        )
        self.gate = nn.Linear(d_model, experts)
        self.top_k = top_k

    def forward(self, x):
        # x: [B, D]
        logits = self.gate(x)
        topk = torch.topk(logits, k=self.top_k, dim=-1)
        weights = F.softmax(topk.values, dim=-1)
        out = torch.zeros_like(x)
        for i in range(self.top_k):
            idx = topk.indices[:, i]
            w = weights[:, i].unsqueeze(-1)
            expert_out = torch.stack(
                [self.experts[j](x[b]) for b, j in enumerate(idx)], dim=0
            )
            out = out + w * expert_out
        return out


class MultimodalValueModel(nn.Module):
    def __init__(self, cfg: ModelConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.debug_save_video = cfg.debug_save_video
        self.visual_only = cfg.vl_backend == "vjepa2"
        if self.visual_only:
            self.backbone = VJEPA2VideoBackbone(cfg, device=device)
            lm_hidden = self.backbone.hidden_size
        else:
            self.backbone = VisionLanguageBackbone(cfg, device=device)
            lm_hidden = self.backbone.get_input_embeddings().embedding_dim
            self.obs_to_lm = nn.Linear(cfg.d_model, lm_hidden)

        try:
            from .gat import GNN_Model
        except Exception as e:
            raise ImportError(
                "Failed to import GNN_Model from aero_gnn.py. "
                "Make sure torch-geometric and torch-scatter are installed."
            ) from e

        # Use the checkpoint-declared row width (8-D legacy or 9-D progress v2).
        self.robot_node_dim = cfg.robot_obs_dim
        gnn_heads = max(1, cfg.temporal_heads)
        gnn_hidden = max(1, math.ceil(cfg.d_model / gnn_heads))
        gnn_args = SimpleNamespace(
            num_heads=gnn_heads,
            iterations=max(1, cfg.gnn_layers),
            dropout=cfg.temporal_dropout,
            num_layers=1,
            add_dropout=False,
            algorithm_name="mappo_dgnn",
            lambd_gnn=1.0,
        )
        self.robot_gnn = GNN_Model(
            args=gnn_args,
            in_channels=self.robot_node_dim,
            hid_channels=gnn_hidden,
            out_channels=cfg.d_model,
            num_agents=cfg.num_robots,
        )

        # self.robot_enc = RobotEncoder(cfg.robot_obs_dim, cfg.d_model)
        self.task_domain_embedding = None
        if self.visual_only and cfg.task_domain_conditioning:
            self.task_domain_embedding = nn.Embedding(
                cfg.num_task_domains, cfg.d_model
            )
            # Preserve an existing checkpoint at initialization; domain-specific
            # offsets are learned without changing the critic-head dimensions.
            nn.init.zeros_(self.task_domain_embedding.weight)
        head_input_dim = lm_hidden + cfg.d_model
        self.value_head = nn.Linear(head_input_dim, 1)
        self.success_head = nn.Linear(head_input_dim, 1)
        nn.init.zeros_(self.success_head.weight)
        nn.init.zeros_(self.success_head.bias)
        # Initialize bias to roughly the mean TD target (return over clip_len)
        # to prevent max_grad_norm from choking the MSE loss component.
        nn.init.constant_(self.value_head.bias, 50.0)

    def _adj_to_batched_edge_index(self, adj: torch.Tensor) -> torch.Tensor:
        # adj: [B, N, N] -> edge_index over flattened batch nodes [2, E]
        bsz, num_nodes, _ = adj.shape
        nz = (adj > 0).nonzero(as_tuple=False)
        if nz.numel() == 0:
            # Fallback to self-loops when there are no edges.
            base = torch.arange(bsz * num_nodes, device=adj.device, dtype=torch.long)
            return torch.stack([base, base], dim=0)

        batch_idx = nz[:, 0]
        src = nz[:, 1]
        dst = nz[:, 2]
        flat_src = batch_idx * num_nodes + src
        flat_dst = batch_idx * num_nodes + dst
        return torch.stack([flat_src.long(), flat_dst.long()], dim=0)

    def forward(
        self,
        video,
        robot_obs,
        adj,
        text_emb=None,
        text_raw=None,
        text_ids=None,
        text_mask=None,
        image_sizes=None,
        return_features=False,
        precomputed_visual_features=None,
    ):
        # video: torch.Tensor [B, T, C, H, W], list of list of PIL images, or preprocessed inputs dict
        # robot_obs: [B, T, N, obs_dim]
        # adj: [B, T, N, N]
        # text_emb: [B, text_dim] or text_raw: list[str]

        inputs = None
        video_list = None
        task_domain_ids = None
        if isinstance(video, dict):
            inputs = dict(video)
            task_domain_ids = inputs.pop("task_domain_ids", None)

        bsz = robot_obs.shape[0]
        # # print('robot_obs shape = ', robot_obs.shape)
        # robot_obs = robot_obs[:, -1, :, :8].reshape(-1, 40)
        # # print('robot_obs shape after = ', robot_obs.shape)
        # robot_feats = self.robot_enc(robot_obs)

        num_nodes = robot_obs.shape[2]
        if adj.shape[-2:] != (num_nodes, num_nodes):
            raise RuntimeError(
                "adjacency shape does not match robot observations: "
                f"robot_obs has {num_nodes} nodes but adj is {tuple(adj.shape[-2:])}."
            )

        # Use only the last-step robot obs and encode team structure with GNN.
        robot_last = robot_obs[
            :, -1, :, : self.robot_node_dim
        ].contiguous()  # [B, N, robot_node_dim]
        adj_last = adj[:, -1, :, :].contiguous()  # [B, N, N]
        edge_index = self._adj_to_batched_edge_index(adj_last)
        robot_node_feats = self.robot_gnn(robot_last, edge_index)  # [B, N, d_model]
        # A true diagonal entry marks a real node. Offline mixed-cardinality
        # batches zero-pad both observations and adjacency, so exclude padded
        # nodes from team pooling. Online environments can pass any N directly.
        node_mask = torch.diagonal(adj_last, dim1=-2, dim2=-1).gt(0)
        empty_graph = ~node_mask.any(dim=1)
        if empty_graph.any():
            node_mask = node_mask.clone()
            node_mask[empty_graph] = True
        node_weights = node_mask.to(robot_node_feats.dtype).unsqueeze(-1)
        robot_team_feat = (robot_node_feats * node_weights).sum(dim=1)
        robot_team_feat = robot_team_feat / node_weights.sum(dim=1).clamp(min=1.0)

        if self.visual_only:
            output = None
            if precomputed_visual_features is None:
                inputs = self.backbone._move_inputs_to_device(inputs)
                output = self.backbone.model(
                    **inputs,
                    skip_predictor=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                final_hidden = output.last_hidden_state
                pooled = final_hidden.mean(dim=1)
            else:
                pooled = torch.as_tensor(precomputed_visual_features)
                if pooled.shape[0] != bsz:
                    raise RuntimeError("precomputed visual feature batch mismatch")
            pooled = pooled.to(
                dtype=self.value_head.weight.dtype,
                device=self.value_head.weight.device,
            )

            multidepth_features = None
            if (
                return_features
                and self.cfg.contrastive_multidepth
                and output is not None
            ):
                multidepth_features = []
                hidden_states = output.hidden_states or ()
                for offset in self.cfg.contrastive_depth_offsets:
                    idx = -(1 + offset)
                    if abs(idx) <= len(hidden_states):
                        feature = hidden_states[idx].mean(dim=1)
                        multidepth_features.append(
                            feature.to(
                                dtype=self.value_head.weight.dtype,
                                device=self.value_head.weight.device,
                            )
                        )

            if self.task_domain_embedding is not None:
                if task_domain_ids is None:
                    task_domain_ids = torch.zeros(
                        bsz, dtype=torch.long, device=pooled.device
                    )
                else:
                    task_domain_ids = torch.as_tensor(
                        task_domain_ids, dtype=torch.long, device=pooled.device
                    ).view(-1)
                task_domain_ids = task_domain_ids.clamp(
                    min=0, max=self.cfg.num_task_domains - 1
                )
                domain_feature = self.task_domain_embedding(task_domain_ids).to(
                    dtype=pooled.dtype
                )
                robot_team_feat = robot_team_feat + domain_feature
            value_head_input = torch.cat((pooled, robot_team_feat), dim=-1)
            value = self.value_head(value_head_input).squeeze(-1)
            if return_features:
                return {
                    "value": value,
                    "success_logit": self.success_head(value_head_input).squeeze(-1),
                    "vlm_feature": pooled,
                    "vlm_multidepth_features": multidepth_features,
                    "robot_team_feature": robot_team_feat,
                    "value_features": value_head_input,
                }
            return value

        # Manual VLM forward: inject robot embeddings at <obs> token positions.
        inputs = self.backbone._move_inputs_to_device(inputs)
        input_ids = inputs["input_ids"].clone()
        attn_mask = inputs.get("attention_mask")
        if attn_mask is not None:
            inputs["attention_mask"] = attn_mask.clone()
        inputs_embeds = self.backbone.get_input_embeddings()(input_ids)

        # Pool team graph features to one token and inject at <obs>.
        obs_token = self.obs_to_lm(robot_team_feat.unsqueeze(1))
        obs_token = obs_token.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        obs_token_id = self.backbone.tokenizer.convert_tokens_to_ids("<obs>")
        if obs_token_id is not None and obs_token_id >= 0:
            obs_mask = input_ids.eq(obs_token_id)

            if obs_mask.any():
                # Avoid in-place updates that can break autograd version tracking.
                obs_mask = obs_mask.unsqueeze(-1)
                if input_ids.shape[0] == bsz:
                    # input_ids: [B, S]
                    obs_token_exp = obs_token.expand(-1, inputs_embeds.size(1), -1)
                elif input_ids.shape[1] == bsz:
                    # input_ids: [S, B]
                    obs_token_exp = obs_token.transpose(0, 1).expand(
                        inputs_embeds.size(0), -1, -1
                    )
                else:
                    raise RuntimeError(
                        f"Unexpected input_ids shape {tuple(input_ids.shape)} for batch size {bsz}."
                    )
                inputs_embeds = torch.where(obs_mask, obs_token_exp, inputs_embeds)

        inputs.pop("input_ids", None)
        inputs["inputs_embeds"] = inputs_embeds
        output = self.backbone.model(
            **inputs, output_hidden_states=True, return_dict=True
        )

        final_hidden = output.hidden_states[-1]
        attn = inputs.get("attention_mask")
        if attn is not None:
            mask = attn.unsqueeze(-1)
            pooled = (final_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = final_hidden[:, -1, :]

        pooled = pooled.to(
            dtype=self.value_head.weight.dtype, device=self.value_head.weight.device
        )
        
        multidepth_features = None
        if return_features and self.cfg.contrastive_multidepth:
            multidepth_features = []
            for offset in self.cfg.contrastive_depth_offsets:
                idx = -(1 + offset)
                if abs(idx) <= len(output.hidden_states):
                    h = output.hidden_states[idx]
                    if attn is not None:
                        h_pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    else:
                        h_pooled = h[:, -1, :]
                    multidepth_features.append(h_pooled.to(dtype=self.value_head.weight.dtype, device=self.value_head.weight.device))

        value_head_input = torch.cat((pooled, robot_team_feat), dim=-1)
        value = self.value_head(value_head_input).squeeze(-1)
        # print('value shape = ', value.shape)
        
        if return_features:
            return {
                "value": value,
                "success_logit": self.success_head(value_head_input).squeeze(-1),
                "vlm_feature": pooled,
                "vlm_multidepth_features": multidepth_features,
                "robot_team_feature": robot_team_feat,
                "value_features": value_head_input,
            }
        return value
