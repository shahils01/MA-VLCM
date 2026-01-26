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
            from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
            from transformers.models.llava_next_video import LlavaNextVideoForConditionalGeneration
            try:
                from transformers.models.auto.modeling_auto import AutoModelForVision2Seq
            except Exception:
                AutoModelForVision2Seq = None
        except Exception as e:
            raise ImportError("LLaVA-Video backend requires transformers installed.") from e

        self.processor = AutoProcessor.from_pretrained(cfg.vl_model_name)
        self.tokenizer = getattr(self.processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            cfg.vl_model_name
        )


        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            cfg.vl_model_name, torch_dtype=dtype
        )

        self.model.to(device)
        if cfg.freeze_vl:
            for p in self.model.parameters():
                p.requires_grad = False

        self._language_model = self._get_language_model()
        self.text_hidden_size = self._language_model.config.hidden_size
        vision_tower = self._get_vision_tower()
        if vision_tower is not None and hasattr(vision_tower, "config") and hasattr(vision_tower.config, "hidden_size"):
            self.vision_hidden_size = vision_tower.config.hidden_size
        else:
            self.vision_hidden_size = self.text_hidden_size
        self._dtype = dtype

    def _get_language_model(self):
        if hasattr(self.model, "language_model"):
            return self.model.language_model
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            return self.model.model.language_model
        if hasattr(self.model, "llm"):
            return self.model.llm
        raise AttributeError("Could not find language model on LLaVA backend.")

    def _get_vision_tower(self):
        if hasattr(self.model, "get_vision_tower"):
            return self.model.get_vision_tower()
        if hasattr(self.model, "vision_tower"):
            return self.model.vision_tower
        if hasattr(self.model, "model") and hasattr(self.model.model, "vision_tower"):
            return self.model.model.vision_tower
        return None

    def _processor_images(self, images):
        if hasattr(self.processor, "image_processor"):
            proc = self.processor.image_processor(images=images, return_tensors="pt")
        elif hasattr(self.processor, "vision_processor"):
            proc = self.processor.vision_processor(images=images, return_tensors="pt")
        else:
            texts = [""] * len(images)
            proc = self.processor(text=texts, images=images, return_tensors="pt")
        pixel_values = proc["pixel_values"].to(self.device, dtype=self._dtype)
        return pixel_values

    def encode_image(self, pixel_values_or_images, image_sizes=None):
        if isinstance(pixel_values_or_images, (list, tuple)):
            pixel_values = self._processor_images(pixel_values_or_images)
        else:
            pixel_values = pixel_values_or_images
            pixel_values = pixel_values.to(self.device, dtype=self._dtype)
        if pixel_values.ndim == 6 and pixel_values.shape[1] == 1:
            pixel_values = pixel_values.squeeze(1)

        if hasattr(self.model, "get_image_features"):
            h, w = pixel_values.shape[-2], pixel_values.shape[-1]
            if pixel_values.ndim == 5:
                n = pixel_values.shape[0] * pixel_values.shape[1]
            else:
                n = pixel_values.shape[0]
            use_sizes = None
            if image_sizes is not None:
                use_sizes = image_sizes
            else:
                use_sizes = [(h, w)] * n
            try:
                img = self.model.get_image_features(pixel_values, image_sizes=use_sizes)
                return img.mean(dim=1) if img.ndim == 3 else img
            except Exception:
                # Some LLaVA video models expect different image_sizes semantics; fall back to vision tower.
                pass

        vision_tower = self._get_vision_tower()
        if vision_tower is None:
            raise AttributeError("LLaVA backend does not expose a vision tower or get_image_features.")
        
        print('pixel_values shape = ', pixel_values.shape)
        if pixel_values.ndim == 5:
            b, t, c, h, w = pixel_values.shape
            pixel_values = pixel_values.view(b * t, c, h, w)
        feats = vision_tower(pixel_values)
        if hasattr(feats, "last_hidden_state"):
            feats = feats.last_hidden_state
        if feats.ndim == 3:
            return feats.mean(dim=1)
        return feats

    def encode_text(self, texts):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.vl_max_text_len,
            return_tensors="pt",
        ).to(self.device)
        return self.encode_text_tokens(tokens["input_ids"], tokens["attention_mask"])

    def encode_text_tokens(self, input_ids, attention_mask):
        tokens = {"input_ids": input_ids.to(self.device), "attention_mask": attention_mask.to(self.device)}
        outputs = self._language_model(**tokens)
        hidden = outputs.last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled



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
        self.backbone = LLaVAVideoBackbone(cfg, device=device)
        if self.backbone.vision_hidden_size != cfg.d_model:
            self.vision_proj = nn.Linear(self.backbone.vision_hidden_size, cfg.d_model)
        else:
            self.vision_proj = nn.Identity()
        if self.backbone.text_hidden_size != cfg.d_model:
            self.text_raw_proj = nn.Linear(self.backbone.text_hidden_size, cfg.d_model)
        else:
            self.text_raw_proj = nn.Identity()

        self.video_temporal = TemporalTransformer(
            cfg.d_model, cfg.temporal_layers, cfg.temporal_heads, cfg.temporal_dropout
        )

        self.robot_enc = RobotEncoder(cfg.robot_obs_dim, cfg.d_model)
        # self.graph_enc = DenseGraphEncoder(cfg.d_model, cfg.gnn_layers)
        self.graph_temporal = TemporalTransformer(
            cfg.d_model, cfg.temporal_layers, cfg.temporal_heads, cfg.temporal_dropout
        )

        self.text_proj = nn.Linear(cfg.text_dim, cfg.d_model)

        fused_dim = cfg.d_model * 3
        if cfg.use_moe:
            self.fusion = MoEFeedForward(fused_dim, cfg.fusion_hidden, cfg.moe_experts, cfg.moe_top_k)
        else:
            self.fusion = nn.Sequential(
                nn.Linear(fused_dim, cfg.fusion_hidden),
                nn.GELU(),
                nn.Linear(cfg.fusion_hidden, fused_dim),
            )

        self.value_head = nn.Linear(fused_dim, 1)

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

        print('video shape = ', video.squeeze().shape)
        video_list = [
            video[i].permute(0, 2, 3, 1).cpu().numpy()
            for i in range(video.shape[0])
        ]
        
        video_inputs = self.backbone.processor(videos=video_list, return_tensors="pt")
        pixel_values_videos = video_inputs["pixel_values_videos"].to(self.backbone.device)

        inputs = {
            "input_ids": text_ids.to(self.backbone.device),
            "attention_mask": text_mask.to(self.backbone.device),
            "pixel_values_videos": pixel_values_videos,
        }

        with torch.no_grad():
            outputs = self.backbone(**inputs, max_new_tokens=128)

        print('outputs = ', outputs)

        # Decode
        texts = self.backbone.processor.batch_decode(outputs, skip_special_tokens=True)
        for i, t in enumerate(texts):
            print(f"[{i}] {t}")


        b, t = video.squeeze().shape[0], video.shape[1]
        if isinstance(self.backbone, LLaVAVideoBackbone):
            vid_tokens = self.backbone.encode_image(video, image_sizes=image_sizes)
        else:
            video_flat = video.view(b * t, *video.shape[2:])
            flat_sizes = None
            if image_sizes is not None:
                if torch.is_tensor(image_sizes):
                    if image_sizes.ndim == 3:
                        flat_sizes = image_sizes.view(b * t, -1)
                    else:
                        flat_sizes = image_sizes
                elif isinstance(image_sizes, (list, tuple)) and len(image_sizes) == b:
                    flat_sizes = []
                    for item in image_sizes:
                        if torch.is_tensor(item):
                            if item.ndim == 2:
                                flat_sizes.extend(item.tolist())
                            else:
                                flat_sizes.append(item.tolist())
                        elif isinstance(item, (list, tuple)):
                            flat_sizes.extend(list(item))
                        else:
                            flat_sizes.append(item)
                else:
                    flat_sizes = image_sizes
            vid_tokens = self.backbone.encode_image(video_flat, image_sizes=flat_sizes)

        if isinstance(self.backbone, LLaVAVideoBackbone):
            if vid_tokens.ndim == 3:
                vid_feat = vid_tokens.mean(dim=1)
            else:
                vid_feat = vid_tokens
            if vid_feat.ndim == 2 and vid_feat.shape[-1] != self.cfg.d_model:
                if isinstance(self.vision_proj, nn.Identity) or (
                    isinstance(self.vision_proj, nn.Linear) and self.vision_proj.in_features != vid_feat.shape[-1]
                ):
                    self.vision_proj = nn.Linear(vid_feat.shape[-1], self.cfg.d_model).to(
                        vid_feat.device, dtype=vid_feat.dtype
                    )
                proj_dtype = self.vision_proj.weight.dtype if hasattr(self.vision_proj, "weight") else vid_feat.dtype
                vid_feat = self.vision_proj(vid_feat.to(proj_dtype))
                print('vid_feat shape = ', vid_feat.shape)

        else:
            # Lazily fix vision projection if backbone dim differs from config
            if isinstance(self.vision_proj, nn.Identity):
                if vid_tokens.shape[-1] != self.cfg.d_model:
                    self.vision_proj = nn.Linear(vid_tokens.shape[-1], self.cfg.d_model).to(
                        vid_tokens.device, dtype=vid_tokens.dtype
                    )
            elif isinstance(self.vision_proj, nn.Linear):
                if self.vision_proj.in_features != vid_tokens.shape[-1]:
                    self.vision_proj = nn.Linear(vid_tokens.shape[-1], self.cfg.d_model).to(
                        vid_tokens.device, dtype=vid_tokens.dtype
                    )
            proj_dtype = self.vision_proj.weight.dtype if hasattr(self.vision_proj, "weight") else vid_tokens.dtype
            vid_tokens = vid_tokens.to(proj_dtype)
            vid_tokens = self.vision_proj(vid_tokens).view(b, t, -1)
            vid_tokens = self.video_temporal(vid_tokens)
            vid_feat = vid_tokens.mean(dim=1)

        print('vid_feat shape outside loop = ', vid_feat.shape)
        print('robot_obs shape = ', robot_obs.shape)

        robot_obs = robot_obs[:,:,:,:8].reshape(-1,40)
        print('robot_obs shape after = ', robot_obs.shape)

        robot_feats = self.robot_enc(robot_obs)
        # print('robot_feats shape = ', robot_feats.shape)
        # robot_feats = self.graph_enc(robot_feats, adj)
        # print('robot_feats shape after GNN = ', robot_feats.shape)

        print('robot_feats shape = ', robot_feats.shape)

        # team_tokens = robot_feats.mean(dim=2)  # [B, T, D]
        # team_tokens = self.graph_temporal(team_tokens)
        # team_feat = team_tokens.mean(dim=1)

        if text_emb is not None:
            text_feat = self.text_proj(text_emb)
        elif text_ids is not None and text_mask is not None:
            txt = self.backbone.encode_text_tokens(text_ids, text_mask)
            txt = txt.to(self.text_raw_proj.weight.dtype) if hasattr(self.text_raw_proj, "weight") else txt
            text_feat = self.text_raw_proj(txt)
        elif text_raw is not None:
            txt = self.backbone.encode_text(text_raw)
            txt = txt.to(self.text_raw_proj.weight.dtype) if hasattr(self.text_raw_proj, "weight") else txt
            text_feat = self.text_raw_proj(txt)
        else:
            raise ValueError("Provide either text_emb or text_raw.")

        print('text_feat shape = ', text_feat.shape)

        fused = torch.cat([vid_feat, robot_feats, text_feat], dim=-1)
        fused = self.fusion(fused)
        value = self.value_head(fused).squeeze(-1)
        return value
