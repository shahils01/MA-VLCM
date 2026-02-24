import math
import inspect
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
    quantization_config: Optional[Any] = None

    # Video
    video_channels: int = 3
    video_height: int = 224
    video_width: int = 224
    video_frames: int = 8
    video_preprocessed: bool = False
    video_mean: tuple = (0.5, 0.5, 0.5)
    video_std: tuple = (0.5, 0.5, 0.5)

    # Robots / graph
    num_robots: int = 8
    robot_obs_dim: int = 16

    # Text (placeholder: supply pre-embedded text vectors)
    text_dim: int = 512

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
    # Value head pooling strategy:
    # - hidden_mean: pool final hidden states over tokens (higher memory)
    # - last_token_logits: use last-token logits as VLM feature (lower memory)
    value_pooling: str = "hidden_mean"
    # If the backend forward supports it, keep logits only for last K tokens.
    logits_to_keep: int = 1




class LLaVAVideoBackbone(nn.Module):
    """Backbone wrapper for LLaVA-style video models using HF interfaces."""

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
            from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, LlavaNextVideoProcessor
            from transformers.models.llava_next_video import LlavaNextVideoForConditionalGeneration
            try:
                from transformers.models.auto.modeling_auto import AutoModelForVision2Seq
            except Exception:
                AutoModelForVision2Seq = None
        except Exception as e:
            raise ImportError("LLaVA-Video backend requires transformers installed.") from e

        self.processor = LlavaNextVideoProcessor.from_pretrained(cfg.vl_model_name)
        self.tokenizer = getattr(self.processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            cfg.vl_model_name
        )
        if "<obs>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<obs>"]})

        model_kwargs = {"torch_dtype": dtype}
        if cfg.quantization_config is not None:
            model_kwargs["quantization_config"] = cfg.quantization_config
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            cfg.vl_model_name, **model_kwargs
        )
        if "<obs>" in self.tokenizer.get_vocab() and hasattr(self.model, "resize_token_embeddings"):
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(device)
        if cfg.freeze_vl:
            for p in self.model.parameters():
                p.requires_grad = False

        self._dtype = dtype

    def get_input_embeddings(self):
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "get_input_embeddings"):
            return self.model.language_model.get_input_embeddings()
        if hasattr(self.model, "model") and hasattr(self.model.model, "get_input_embeddings"):
            return self.model.model.get_input_embeddings()
        raise AttributeError("Could not access input embeddings on LLaVA backbone.")

    def _move_inputs_to_device(self, inputs):
        moved = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                if k in ("pixel_values", "pixel_values_videos", "video_values", "video", "videos"):
                    moved[k] = v.to(self.device, dtype=self._dtype)
                else:
                    moved[k] = v.to(self.device)
            else:
                moved[k] = v
        return moved

    def prepare_inputs(self, text, videos, padding=False, truncation=False, max_length=None):
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
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(layers)])

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
            expert_out = torch.stack([self.experts[j](x[b]) for b, j in enumerate(idx)], dim=0)
            out = out + w * expert_out
        return out


class MultimodalValueModel(nn.Module):
    def __init__(self, cfg: ModelConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.debug_save_video = cfg.debug_save_video
        self.backbone = LLaVAVideoBackbone(cfg, device=device)
        try:
            self._backbone_forward_params = set(inspect.signature(self.backbone.model.forward).parameters.keys())
        except Exception:
            self._backbone_forward_params = set()
        lm_hidden = self.backbone.get_input_embeddings().embedding_dim
        self.obs_to_lm = nn.Linear(cfg.d_model, lm_hidden)

        try:
            from gat import GNN_Model
        except Exception as e:
            raise ImportError(
                "Failed to import GNN_Model from aero_gnn.py. "
                "Make sure torch-geometric and torch-scatter are installed."
            ) from e

        # Keep the same node feature slice used before flattening (first 8 dims per robot).
        self.robot_node_dim = min(8, cfg.robot_obs_dim)
        gnn_heads = max(1, cfg.temporal_heads)
        gnn_hidden = max(1, math.ceil(cfg.d_model / gnn_heads))
        gnn_args = SimpleNamespace(
            num_heads=gnn_heads,
            iterations=max(1, cfg.gnn_layers),
            dropout=cfg.temporal_dropout,
            num_layers=3,
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
        if cfg.value_pooling == "last_token_logits":
            vl_feat_dim = int(getattr(self.backbone.model.config, "vocab_size", 0))
            if vl_feat_dim <= 0:
                raise RuntimeError("Invalid vocab size on VLM backbone for logits-based value pooling.")
        else:
            vl_feat_dim = lm_hidden
        self.value_head = nn.Linear(vl_feat_dim + cfg.d_model, 1)

    def _decode_debug_text(self, logits: torch.Tensor, attention_mask: Optional[torch.Tensor], max_tokens: int):
        # Decode greedy token predictions from the LM head for quick introspection.
        pred_ids = logits.argmax(dim=-1)
        if max_tokens is not None and max_tokens > 0 and pred_ids.size(1) > max_tokens:
            pred_ids = pred_ids[:, -max_tokens:]

        if attention_mask is not None and attention_mask.shape[:2] == logits.shape[:2]:
            trimmed_ids = []
            for i in range(pred_ids.size(0)):
                valid_len = int(attention_mask[i].sum().item())
                if valid_len <= 0:
                    sample_ids = pred_ids[i, -1:]
                else:
                    start = max(0, valid_len - max_tokens)
                    sample_ids = logits[i, start:valid_len, :].argmax(dim=-1)
                trimmed_ids.append(sample_ids)
            text = [
                self.backbone.tokenizer.decode(ids.detach().cpu().tolist(), skip_special_tokens=True)
                for ids in trimmed_ids
            ]
            return text

        return self.backbone.tokenizer.batch_decode(
            pred_ids.detach().cpu(), skip_special_tokens=True
        )

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
        return_debug: bool = False,
        debug_max_tokens: int = 32,
    ):
        # video: torch.Tensor [B, T, C, H, W], list of list of PIL images, or preprocessed inputs dict
        # robot_obs: [B, T, N, obs_dim]
        # adj: [B, T, N, N]
        # text_emb: [B, text_dim] or text_raw: list[str]

        inputs = None
        video_list = None
        if isinstance(video, dict):
            inputs = video
        
        bsz = robot_obs.shape[0]
        # # print('robot_obs shape = ', robot_obs.shape)
        # robot_obs = robot_obs[:, -1, :, :8].reshape(-1, 40)
        # # print('robot_obs shape after = ', robot_obs.shape)
        # robot_feats = self.robot_enc(robot_obs)

        num_nodes = robot_obs.shape[2]
        if num_nodes != self.cfg.num_robots:
            raise RuntimeError(
                f"robot_obs has {num_nodes} nodes, but config expects {self.cfg.num_robots}. "
                "Set --num_robots to match your dataset."
            )

        # Use only the last-step robot obs and encode team structure with GNN.
        robot_last = robot_obs[:, -1, :, : self.robot_node_dim].contiguous()  # [B, N, robot_node_dim]
        adj_last = adj[:, -1, :, :].contiguous()  # [B, N, N]
        edge_index = self._adj_to_batched_edge_index(adj_last)
        robot_node_feats = self.robot_gnn(robot_last, edge_index)  # [B, N, d_model]
        robot_team_feat = robot_node_feats.mean(dim=1)  # [B, d_model]

        # Manual forward: build inputs_embeds and inject robot embeddings at <obs> token positions.
        inputs = self.backbone._move_inputs_to_device(inputs)
        input_ids = inputs["input_ids"]
        attn_mask = inputs.get("attention_mask")
        inputs_embeds = self.backbone.get_input_embeddings()(input_ids)

        # Pool team graph features to one token and inject at <obs>.
        obs_token = self.obs_to_lm(robot_team_feat.unsqueeze(1))
        obs_token = obs_token.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        obs_token_id = self.backbone.tokenizer.convert_tokens_to_ids("<obs>")
        if obs_token_id is not None and obs_token_id >= 0:
            obs_mask = input_ids.eq(obs_token_id)

            if obs_mask.any():
                # Avoid dense broadcast replacements over the full sequence.
                inputs_embeds = inputs_embeds.clone()
                if input_ids.shape[0] == bsz:
                    # input_ids: [B, S]
                    b_idx, s_idx = obs_mask.nonzero(as_tuple=True)
                    inputs_embeds[b_idx, s_idx, :] = obs_token[b_idx, 0, :]
                elif input_ids.shape[1] == bsz:
                    # input_ids: [S, B]
                    s_idx, b_idx = obs_mask.nonzero(as_tuple=True)
                    inputs_embeds[s_idx, b_idx, :] = obs_token[b_idx, 0, :]
                else:
                    raise RuntimeError(
                        f"Unexpected input_ids shape {tuple(input_ids.shape)} for batch size {bsz}."
                    )

        inputs.pop("input_ids", None)
        inputs["inputs_embeds"] = inputs_embeds
        forward_kwargs = {"return_dict": True}
        if hasattr(self.backbone.model, "config") and hasattr(self.backbone.model.config, "use_cache"):
            # KV cache is only useful for autoregressive generation, not value regression training.
            forward_kwargs["use_cache"] = False
        if self.cfg.value_pooling == "hidden_mean":
            forward_kwargs["output_hidden_states"] = True
        if self.cfg.logits_to_keep > 0:
            # Some transformers versions expose either logits_to_keep or num_logits_to_keep.
            if "logits_to_keep" in self._backbone_forward_params:
                forward_kwargs["logits_to_keep"] = self.cfg.logits_to_keep
            elif "num_logits_to_keep" in self._backbone_forward_params:
                forward_kwargs["num_logits_to_keep"] = self.cfg.logits_to_keep

        output = self.backbone.model(**inputs, **forward_kwargs)

        final_hidden = None
        if self.cfg.value_pooling == "hidden_mean":
            final_hidden = output.hidden_states[-1]
        debug_text = None
        if return_debug:
            logits = getattr(output, "logits", None)
            if logits is None:
                lm_head = self.backbone.model.get_output_embeddings()
                if lm_head is None:
                    raise RuntimeError("Unable to decode debug text: no LM logits or output embeddings available.")
                logits = lm_head(final_hidden)
            debug_text = self._decode_debug_text(logits, inputs.get("attention_mask"), max_tokens=debug_max_tokens)

        attn = inputs.get("attention_mask")
        if self.cfg.value_pooling == "hidden_mean":
            if final_hidden is None:
                raise RuntimeError("hidden_mean value pooling requested but hidden states were not returned.")
            if attn is not None:
                mask = attn.unsqueeze(-1)
                pooled = (final_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = final_hidden[:, -1, :]
        else:
            logits = getattr(output, "logits", None)
            if logits is None:
                raise RuntimeError("logits-based value pooling requested but model output has no logits.")
            pooled = logits[:, -1, :]

        pooled = pooled.to(dtype=self.value_head.weight.dtype, device=self.value_head.weight.device)

        value_head_input = torch.cat((pooled, robot_team_feat), dim=-1)
        value = self.value_head(value_head_input).squeeze(-1)
        if return_debug:
            return {"value": value, "debug_text": debug_text}
        return value
