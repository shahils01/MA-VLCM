import math
from dataclasses import dataclass

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
    gnn_layers: int = 2

    # Fusion
    fusion_hidden: int = 512
    use_moe: bool = False
    moe_experts: int = 4
    moe_top_k: int = 2

    # Debug
    debug_save_video: bool = False




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

        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            cfg.vl_model_name, torch_dtype=dtype
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
                if k in ("pixel_values", "video", "videos"):
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
        lm_hidden = self.backbone.get_input_embeddings().embedding_dim
        self.obs_to_lm = nn.Linear(cfg.d_model, lm_hidden)

        # self.video_temporal = TemporalTransformer(
        #     cfg.d_model, cfg.temporal_layers, cfg.temporal_heads, cfg.temporal_dropout
        # )

        self.robot_enc = RobotEncoder(cfg.robot_obs_dim, cfg.d_model)
        # self.graph_enc = DenseGraphEncoder(cfg.d_model, cfg.gnn_layers)
        # self.graph_temporal = TemporalTransformer(
        #     cfg.d_model, cfg.temporal_layers, cfg.temporal_heads, cfg.temporal_dropout
        # )

        # self.text_proj = nn.Linear(cfg.text_dim, cfg.d_model)

        # fused_dim = cfg.d_model * 3
        # if cfg.use_moe:
        #     self.fusion = MoEFeedForward(fused_dim, cfg.fusion_hidden, cfg.moe_experts, cfg.moe_top_k)
        # else:
        #     self.fusion = nn.Sequential(
        #         nn.Linear(fused_dim, cfg.fusion_hidden),
        #         nn.GELU(),
        #         nn.Linear(cfg.fusion_hidden, fused_dim),
        #     )

        self.value_head = nn.Linear(lm_hidden, 1)

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
    ):
        # video: torch.Tensor [B, T, C, H, W] or list of list of PIL images
        # robot_obs: [B, T, N, obs_dim]
        # adj: [B, T, N, N]
        # text_emb: [B, text_dim] or text_raw: list[str]

        video = video.squeeze().clip(0,1)
        video_list = [
            video[i].permute(0, 2, 3, 1).to(dtype=torch.float16)#.cpu().numpy()
            for i in range(video.shape[0])
        ]

        import imageio
        import numpy as np
        def save_video_mp4(_video, path, fps=30):
            """
            video: np.ndarray (T, H, W, C)
            path: output file path, e.g. 'output.mp4'
            """
            _video = _video.cpu().numpy()
            # Ensure uint8
            if _video.dtype != np.uint8:
                # _video = np.clip(_video, 0, 1)
                _video = (_video * 255).astype(np.uint8)

            writer = imageio.get_writer(path, fps=fps, codec='libx264')
            for frame in _video:
                writer.append_data(frame)
            writer.close()

        if self.debug_save_video:
            for i, _video in enumerate(video_list):
                save_video_mp4(_video, f"video_{i}.mp4", fps=24)

        text_raw = "<video><obs>You are a critic model. You are given video of a tean of robots (denoted as circular dots with heading denoted by an arrow).\
                    The goal for each robot is denoted by the same color square box. The robots have to go to their designated goal\
                    without colliding with one another. They also have to be efficient by taking the shortest parth.\
                    How Good or Bad are the team of robots doing to accomplish the given task? Also tell me why and what you see. Keep your answer short."

        # text_raw = "<video>You are a critic model. What colors do you see in this video? How many frames you see in this video?"
        text_list = [text_raw] * len(video_list)

        inputs = self.backbone.prepare_inputs(text=text_list, videos=video_list, padding=False)

        robot_obs = robot_obs[:,:,:,:8].reshape(-1,40)
        # print('robot_obs shape after = ', robot_obs.shape)

        robot_feats = self.robot_enc(robot_obs)
        # print('robot_feats shape = ', robot_feats.shape)
        # robot_feats = self.graph_enc(robot_feats, adj)
        # print('robot_feats shape after GNN = ', robot_feats.shape)

        # print('robot_feats shape = ', robot_feats.shape)

        # Manual forward: build inputs_embeds and inject robot embeddings at <obs> token positions.
        inputs = self.backbone._move_inputs_to_device(inputs)
        input_ids = inputs["input_ids"]
        inputs_embeds = self.backbone.get_input_embeddings()(input_ids)

        # Pool robot features to a single token per batch, then project to LM hidden size.
        bsz = input_ids.shape[0]
        if robot_feats.shape[0] % bsz == 0:
            robot_seq = robot_feats.view(bsz, -1, robot_feats.shape[-1])
            obs_token = self.obs_to_lm(robot_seq.mean(dim=1, keepdim=True))
        else:
            obs_token = self.obs_to_lm(robot_feats.mean(dim=0, keepdim=True)).unsqueeze(0)
        obs_token = obs_token.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        obs_token_id = self.backbone.tokenizer.convert_tokens_to_ids("<obs>")
        if obs_token_id is not None and obs_token_id >= 0:
            obs_mask = input_ids.eq(obs_token_id)
            for b in range(bsz):
                if obs_mask[b].any():
                    inputs_embeds[b, obs_mask[b]] = obs_token[b].expand(obs_mask[b].sum(), -1)

        inputs.pop("input_ids", None)
        inputs["inputs_embeds"] = inputs_embeds
        output = self.backbone.model(**inputs, output_hidden_states=True, return_dict=True)

        final_hidden = output.hidden_states[-1]
        attn = inputs.get("attention_mask")
        if attn is not None:
            mask = attn.unsqueeze(-1)
            pooled = (final_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = final_hidden[:, -1, :]

        pooled = pooled.to(dtype=self.value_head.weight.dtype, device=self.value_head.weight.device)
        value = self.value_head(pooled).squeeze(-1)
        # print('value shape = ', value.shape)
        return value
