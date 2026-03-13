import math
import inspect
import os
import time
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
    debug_save_video: bool = True
    # Value head pooling strategy:
    # - hidden_mean: pool final hidden states over tokens (higher memory)
    # - last_token_logits: use last-token logits as VLM feature (lower memory)
    value_pooling: str = "hidden_mean"
    # If the backend forward supports it, keep logits only for last K tokens.
    logits_to_keep: int = 1
    # Number of learned graph-summary tokens to inject at <obs> positions.
    obs_summary_tokens: int = 2
    # Multi-depth contrastive supervision settings.
    contrastive_multidepth: bool = False
    # Offsets from final hidden layer (0=last, 1=one before, ...).
    contrastive_depth_offsets: tuple = (0,)




class LLaVAVideoBackbone(nn.Module):
    """Backbone wrapper for HF multimodal backbones used by this project."""

    @staticmethod
    def _normalize_media_size(image_size):
        if isinstance(image_size, (tuple, list)):
            if len(image_size) >= 2:
                return {"height": int(image_size[0]), "width": int(image_size[1])}
            if len(image_size) == 1:
                size = int(image_size[0])
                return {"height": size, "width": size}
        if image_size is None:
            return None
        size = int(image_size)
        return {"height": size, "width": size}

    def _configure_processor_media_size(self, cfg_hf):
        vision_cfg = getattr(cfg_hf, "vision_config", None)
        image_size = getattr(vision_cfg, "image_size", None) if vision_cfg is not None else None
        media_size = self._normalize_media_size(image_size)
        if media_size is None:
            return

        for proc_name in ("image_processor", "video_processor"):
            proc = getattr(self.processor, proc_name, None)
            if proc is None:
                continue
            if hasattr(proc, "size"):
                proc.size = dict(media_size)
            if hasattr(proc, "crop_size"):
                proc.crop_size = dict(media_size)

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
            from transformers import AutoConfig, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
            from transformers.utils import logging as hf_logging
            try:
                from transformers.models.auto.modeling_auto import AutoModelForVision2Seq
            except Exception:
                AutoModelForVision2Seq = None
            try:
                from transformers.models.auto.modeling_auto import AutoModelForImageTextToText
            except Exception:
                AutoModelForImageTextToText = None
            try:
                from transformers.models.llava_next_video import LlavaNextVideoForConditionalGeneration
            except Exception:
                LlavaNextVideoForConditionalGeneration = None
            try:
                from transformers.models.llava_onevision import LlavaOnevisionForConditionalGeneration
            except Exception:
                LlavaOnevisionForConditionalGeneration = None
        except Exception as e:
            raise ImportError("HF multimodal backends require transformers installed.") from e

        trust_remote_code = cfg.vl_backend == "internvl"
        cfg_hf = AutoConfig.from_pretrained(cfg.vl_model_name, trust_remote_code=trust_remote_code)
        model_type = str(getattr(cfg_hf, "model_type", "")).lower()

        self.processor = AutoProcessor.from_pretrained(
            cfg.vl_model_name,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            cfg.vl_model_name,
            trust_remote_code=trust_remote_code,
        )
        self._configure_processor_media_size(cfg_hf)
        if "<obs>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<obs>"]})

        model_kwargs = {"torch_dtype": dtype}
        if cfg.quantization_config is not None:
            model_kwargs["quantization_config"] = cfg.quantization_config
        load_info = None
        if cfg.vl_backend == "internvl":
            model_kwargs["trust_remote_code"] = trust_remote_code
            if AutoModelForImageTextToText is not None:
                self.model = AutoModelForImageTextToText.from_pretrained(cfg.vl_model_name, **model_kwargs)
            elif AutoModelForVision2Seq is not None:
                self.model = AutoModelForVision2Seq.from_pretrained(cfg.vl_model_name, **model_kwargs)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(cfg.vl_model_name, **model_kwargs)
        elif model_type == "llava_next_video":
            if LlavaNextVideoForConditionalGeneration is None:
                raise ImportError("This transformers build does not provide LlavaNextVideoForConditionalGeneration.")
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                cfg.vl_model_name, **model_kwargs
            )
        elif model_type == "llava_onevision":
            if LlavaOnevisionForConditionalGeneration is None:
                raise ImportError("This transformers build does not provide LlavaOnevisionForConditionalGeneration.")
            # OneVision checkpoints can rely on tied LM head weights. Handle that case explicitly.
            old_verbosity = hf_logging.get_verbosity()
            try:
                hf_logging.set_verbosity_error()
                self.model, load_info = LlavaOnevisionForConditionalGeneration.from_pretrained(
                    cfg.vl_model_name,
                    output_loading_info=True,
                    **model_kwargs,
                )
            finally:
                hf_logging.set_verbosity(old_verbosity)
        elif AutoModelForVision2Seq is not None:
            self.model = AutoModelForVision2Seq.from_pretrained(cfg.vl_model_name, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.vl_model_name, **model_kwargs)
        if "<obs>" in self.tokenizer.get_vocab() and hasattr(self.model, "resize_token_embeddings"):
            self.model.resize_token_embeddings(len(self.tokenizer))
        if hasattr(self.model, "tie_weights"):
            try:
                self.model.tie_weights()
            except Exception:
                pass
        if load_info is not None:
            missing = set(load_info.get("missing_keys", []))
            if missing and missing != {"lm_head.weight"}:
                raise RuntimeError(
                    f"Unexpected missing checkpoint keys for {cfg.vl_model_name}: {sorted(missing)}"
                )

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
        processor_kwargs = dict(
            text=text,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        try:
            processor_kwargs["videos"] = videos
            inputs = self.processor(**processor_kwargs)
        except TypeError:
            processor_kwargs.pop("videos", None)
            processor_kwargs["images"] = videos
            inputs = self.processor(**processor_kwargs)
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
        self._debug_video_saved = False
        self._debug_video_dir = os.environ.get("MA_VLCM_DEBUG_VIDEO_DIR", "debug_samples/model_forward")
        self.backbone = LLaVAVideoBackbone(cfg, device=device)
        try:
            self._backbone_forward_params = set(inspect.signature(self.backbone.model.forward).parameters.keys())
        except Exception:
            self._backbone_forward_params = set()
        lm_hidden = self.backbone.get_input_embeddings().embedding_dim
        self.obs_to_lm = nn.Linear(cfg.d_model, lm_hidden)
        self.token_attn_pool = nn.Linear(lm_hidden, 1)
        self.robot_temporal = TemporalTransformer(
            d_model=cfg.d_model,
            layers=cfg.temporal_layers,
            heads=cfg.temporal_heads,
            dropout=cfg.temporal_dropout,
        )
        if int(cfg.obs_summary_tokens) < 1:
            raise ValueError("obs_summary_tokens must be >= 1")
        self.obs_summary_tokens = int(cfg.obs_summary_tokens)
        self.obs_queries = nn.Parameter(torch.randn(self.obs_summary_tokens, cfg.d_model) * (cfg.d_model ** -0.5))

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
            vl_feat_dim = self._infer_vocab_size()
            if vl_feat_dim <= 0:
                raise RuntimeError(
                    "Unable to infer vocab size for logits-based value pooling. "
                    "Use --value_pooling hidden_mean or check model output embeddings."
                )
        else:
            vl_feat_dim = lm_hidden
        self.value_head = nn.Linear(vl_feat_dim + cfg.d_model, 1)

    def _backbone_uses_gradient_checkpointing(self) -> bool:
        model = self.backbone.model
        if bool(getattr(model, "is_gradient_checkpointing", False)):
            return True
        if bool(getattr(model, "gradient_checkpointing", False)):
            return True
        cfg = getattr(model, "config", None)
        return bool(getattr(cfg, "gradient_checkpointing", False))

    def _attention_max_pool(self, hidden: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # hidden: [B, S, D] -> [B, D] via attention-weighted max pooling over sequence tokens.
        scores = self.token_attn_pool(hidden).squeeze(-1)  # [B, S]
        if attn_mask is not None and attn_mask.shape[:2] == hidden.shape[:2]:
            valid = attn_mask.bool()
            scores = scores.masked_fill(~valid, -1e9)
        else:
            valid = None
        attn = F.softmax(scores, dim=1)  # [B, S]
        weighted = hidden * attn.unsqueeze(-1)  # [B, S, D]
        if valid is not None:
            weighted = weighted.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        pooled = weighted.max(dim=1).values  # [B, D]

        # Fallback for rows that became all -inf due to masking.
        bad = ~torch.isfinite(pooled).all(dim=1)
        if bad.any():
            fallback = hidden[bad].mean(dim=1)
            pooled = pooled.clone()
            pooled[bad] = fallback
        return pooled

    @staticmethod
    def _clone_nondiff_inputs(inputs: dict) -> dict:
        cloned = {}
        for key, value in inputs.items():
            if torch.is_tensor(value) and not torch.is_floating_point(value):
                cloned[key] = value.clone()
            else:
                cloned[key] = value
        return cloned

    def _maybe_save_debug_video_from_inputs(self, inputs: dict):
        # if (not self.debug_save_video) or self._debug_video_saved:
        #     return
        if not isinstance(inputs, dict):
            return

        video = None
        for key in ("pixel_values_videos", "pixel_values", "video_values", "videos"):
            val = inputs.get(key, None)
            if torch.is_tensor(val):
                video = val
                break
        if video is None:
            return

        # Robustly reduce to [T,H,W,C] for a single sample.
        x = video.detach().float().cpu()
        while x.dim() > 5:
            x = x[0]
        if x.dim() == 5:
            x = x[0]
        if x.dim() == 4:
            if x.shape[1] in (1, 3):  # [T,C,H,W]
                frames = x.permute(0, 2, 3, 1).contiguous()
            elif x.shape[-1] in (1, 3):  # [T,H,W,C]
                frames = x
            elif x.shape[0] in (1, 3):  # [C,T,H,W]
                frames = x.permute(1, 2, 3, 0).contiguous()
            else:
                return
        elif x.dim() == 3:
            if x.shape[0] in (1, 3):  # [C,H,W]
                frames = x.permute(1, 2, 0).unsqueeze(0).contiguous()
            elif x.shape[-1] in (1, 3):  # [H,W,C]
                frames = x.unsqueeze(0).contiguous()
            else:
                return
        else:
            return

        mn = float(frames.min().item())
        mx = float(frames.max().item())
        mean = float(frames.mean().item())

        # Convert common normalized ranges into uint8 for mp4 visualization.
        if mn >= -1.01 and mx <= 1.01 and mn < 0.0:
            frames_u8 = ((frames + 1.0) * 127.5).clamp(0.0, 255.0).to(torch.uint8)
        elif mx <= 1.01:
            frames_u8 = (frames * 255.0).clamp(0.0, 255.0).to(torch.uint8)
        else:
            frames_u8 = frames.clamp(0.0, 255.0).to(torch.uint8)

        if frames_u8.shape[-1] == 1:
            frames_u8 = frames_u8.repeat(1, 1, 1, 3)

        os.makedirs(self._debug_video_dir, exist_ok=True)
        out_path = os.path.join(
            self._debug_video_dir,
            f"forward_debug_{int(time.time())}_min{mn:.4f}_max{mx:.4f}_mean{mean:.4f}.mp4",
        )
        try:
            import imageio.v2 as imageio

            imageio.mimsave(out_path, frames_u8.numpy(), fps=4)
            print(f"[MA-VLCM debug] saved forward input video: {out_path}")
        except Exception as e:
            print(f"[MA-VLCM debug] failed to save mp4 ({e}); path={out_path}")
        finally:
            self._debug_video_saved = True

    def _infer_vocab_size(self) -> int:
        # Prefer LM head/output embeddings shape, then fall back to config fields.
        get_out = getattr(self.backbone.model, "get_output_embeddings", None)
        if callable(get_out):
            try:
                out_emb = get_out()
            except Exception:
                out_emb = None
            if out_emb is not None and hasattr(out_emb, "weight") and out_emb.weight is not None:
                return int(out_emb.weight.shape[0])

        cfg = getattr(self.backbone.model, "config", None)
        candidates = []
        if cfg is not None:
            candidates.append(getattr(cfg, "vocab_size", 0))
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is not None:
                candidates.append(getattr(text_cfg, "vocab_size", 0))
            lm_cfg = getattr(getattr(self.backbone.model, "language_model", None), "config", None)
            if lm_cfg is not None:
                candidates.append(getattr(lm_cfg, "vocab_size", 0))
        for v in candidates:
            try:
                iv = int(v)
            except Exception:
                iv = 0
            if iv > 0:
                return iv
        return 0

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

    def _encode_robot_temporal(self, robot_obs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # robot_obs: [B, T, N, robot_node_dim], adj: [B, T, N, N] -> [B, K, d_model]
        bsz, tlen = robot_obs.shape[0], robot_obs.shape[1]
        step_team = []
        for t in range(tlen):
            robot_t = robot_obs[:, t, :, :].contiguous()  # [B, N, robot_node_dim]
            adj_t = adj[:, t, :, :].contiguous()  # [B, N, N]
            edge_index = self._adj_to_batched_edge_index(adj_t).to(device=robot_t.device)
            node_feats = self.robot_gnn(robot_t, edge_index)  # [B, N, d_model]
            step_team.append(node_feats.mean(dim=1))  # [B, d_model]

        team_seq = torch.stack(step_team, dim=1)  # [B, T, d_model]
        team_seq = self.robot_temporal(team_seq)  # [B, T, d_model]

        # Learned query pooling over time to produce K summary tokens.
        queries = self.obs_queries.unsqueeze(0).expand(bsz, -1, -1)  # [B, K, d_model]
        attn_logits = torch.einsum("bkd,btd->bkt", queries, team_seq) / math.sqrt(float(self.cfg.d_model))
        attn = F.softmax(attn_logits, dim=-1)
        summary = torch.einsum("bkt,btd->bkd", attn, team_seq)  # [B, K, d_model]
        return summary

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
        return_features: bool = False,
        debug_max_tokens: int = 32,
        debug_video=False,
    ):
        # video: torch.Tensor [B, T, C, H, W], list of list of PIL images, or preprocessed inputs dict
        # robot_obs: [B, T, N, obs_dim]
        # adj: [B, T, N, N]
        # text_emb: [B, text_dim] or text_raw: list[str]

        inputs = None
        video_list = None
        if isinstance(video, dict):
            inputs = video

        # Keep graph tensors on the same device as the robot GNN weights.
        gnn_device = next(self.robot_gnn.parameters()).device
        if robot_obs.device != gnn_device:
            robot_obs = robot_obs.to(gnn_device)
        if adj.device != gnn_device:
            adj = adj.to(gnn_device)
        
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

        # Encode per-step graph features, then summarize temporally into K tokens.
        robot_seq = robot_obs[:, :, :, : self.robot_node_dim].contiguous()  # [B, T, N, robot_node_dim]
        robot_summary_tokens = self._encode_robot_temporal(robot_seq, adj)  # [B, K, d_model]
        robot_team_feat = robot_summary_tokens.mean(dim=1)  # [B, d_model]

        # Manual forward: build inputs_embeds and inject robot embeddings at <obs> token positions.
        inputs = self._clone_nondiff_inputs(self.backbone._move_inputs_to_device(inputs))
        if debug_video:
            self._maybe_save_debug_video_from_inputs(inputs)
        # Keep token metadata isolated from any in-place mutations inside the HF forward.
        input_ids = inputs["input_ids"]
        attn_mask = inputs.get("attention_mask")
        attn_for_pool = attn_mask.clone() if attn_mask is not None else None
        inputs_embeds = self.backbone.get_input_embeddings()(input_ids)

        # Project K graph-summary tokens to LM space and inject at <obs> positions.
        obs_token = self.obs_to_lm(robot_summary_tokens)
        obs_token = obs_token.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)  # [B, K, H]

        obs_token_id = self.backbone.tokenizer.convert_tokens_to_ids("<obs>")
        if obs_token_id is not None and obs_token_id >= 0:
            obs_mask = input_ids.eq(obs_token_id)

            if obs_mask.any():
                if input_ids.shape[0] == bsz:
                    # input_ids: [B, S]
                    replaced = inputs_embeds.clone()
                    k = obs_token.size(1)
                    for b in range(bsz):
                        pos = obs_mask[b].nonzero(as_tuple=False).view(-1)
                        if pos.numel() == 0:
                            continue
                        use_n = min(int(pos.numel()), int(k))
                        replaced[b, pos[:use_n], :] = obs_token[b, :use_n, :]
                        if pos.numel() > use_n:
                            tail = obs_token[b, use_n - 1 : use_n, :].expand(pos.numel() - use_n, -1)
                            replaced[b, pos[use_n:], :] = tail
                    inputs_embeds = replaced
                elif input_ids.shape[1] == bsz:
                    # input_ids: [S, B]
                    replaced = inputs_embeds.clone()
                    k = obs_token.size(1)
                    for b in range(bsz):
                        pos = obs_mask[:, b].nonzero(as_tuple=False).view(-1)
                        if pos.numel() == 0:
                            continue
                        use_n = min(int(pos.numel()), int(k))
                        replaced[pos[:use_n], b, :] = obs_token[b, :use_n, :]
                        if pos.numel() > use_n:
                            tail = obs_token[b, use_n - 1 : use_n, :].expand(pos.numel() - use_n, -1)
                            replaced[pos[use_n:], b, :] = tail
                    inputs_embeds = replaced
                else:
                    raise RuntimeError(
                        f"Unexpected input_ids shape {tuple(input_ids.shape)} for batch size {bsz}."
                    )

        model_inputs = {k: v for k, v in inputs.items() if k != "input_ids"}
        if attn_mask is not None:
            # Keep model mask separate from the copy used for value pooling/debug text.
            model_inputs["attention_mask"] = attn_mask.clone()
        if not inputs_embeds.requires_grad and self._backbone_uses_gradient_checkpointing():
            # We feed `inputs_embeds` directly, so the usual embedding-layer hook that
            # marks checkpoint inputs as requiring grad does not fire. Without this,
            # LoRA/QLoRA weights inside checkpointed blocks can appear unused to DDP.
            inputs_embeds = inputs_embeds.requires_grad_(True)
        model_inputs["inputs_embeds"] = inputs_embeds
        forward_kwargs = {"return_dict": True}
        if hasattr(self.backbone.model, "config") and hasattr(self.backbone.model.config, "use_cache"):
            # KV cache is only useful for autoregressive generation, not value regression training.
            forward_kwargs["use_cache"] = False
        need_hidden_states = self.cfg.value_pooling == "hidden_mean" or (
            return_features and bool(getattr(self.cfg, "contrastive_multidepth", False))
        )
        if need_hidden_states:
            forward_kwargs["output_hidden_states"] = True
        if self.cfg.logits_to_keep > 0:
            # Some transformers versions expose either logits_to_keep or num_logits_to_keep.
            if "logits_to_keep" in self._backbone_forward_params:
                forward_kwargs["logits_to_keep"] = self.cfg.logits_to_keep
            elif "num_logits_to_keep" in self._backbone_forward_params:
                forward_kwargs["num_logits_to_keep"] = self.cfg.logits_to_keep

        output = self.backbone.model(**model_inputs, **forward_kwargs)

        hidden_states = getattr(output, "hidden_states", None)
        final_hidden = None
        if hidden_states is not None and len(hidden_states) > 0:
            final_hidden = hidden_states[-1]
        debug_text = None
        if return_debug:
            logits = getattr(output, "logits", None)
            if logits is None:
                lm_head = self.backbone.model.get_output_embeddings()
                if lm_head is None:
                    raise RuntimeError("Unable to decode debug text: no LM logits or output embeddings available.")
                logits = lm_head(final_hidden)
            debug_text = self._decode_debug_text(logits, attn_for_pool, max_tokens=debug_max_tokens)

        attn = attn_for_pool
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

        vlm_multidepth_features = None
        if return_features and bool(getattr(self.cfg, "contrastive_multidepth", False)):
            if hidden_states is None or len(hidden_states) == 0:
                raise RuntimeError(
                    "Multi-depth contrastive supervision requested but hidden states were not returned by the backbone."
                )
            depth_feats = []
            offsets = tuple(getattr(self.cfg, "contrastive_depth_offsets", (0,)))
            for off in offsets:
                off_i = max(0, int(off))
                idx = max(0, len(hidden_states) - 1 - off_i)
                feat = self._attention_max_pool(hidden_states[idx], attn_for_pool)
                depth_feats.append(feat)
            vlm_multidepth_features = depth_feats

        value_head_input = torch.cat((pooled, robot_team_feat), dim=-1)
        value = self.value_head(value_head_input).squeeze(-1)
        if return_debug or return_features:
            out = {"value": value, "vlm_feature": pooled}
            if vlm_multidepth_features is not None:
                out["vlm_multidepth_features"] = vlm_multidepth_features
            if return_debug:
                out["debug_text"] = debug_text
            return out
        return value
