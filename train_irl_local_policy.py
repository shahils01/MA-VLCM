import argparse
import os
import random
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import wandb

# ManyAgentGoToGoal rendering is matplotlib-based; Agg avoids black/empty frames on headless nodes.
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import ManyAgent_GoTOGoal  # noqa: F401
except Exception:
    ManyAgent_GoTOGoal = None

from data_loading import preprocess_vlm_video_inputs, webdataset_loader
from model import ModelConfig, MultimodalValueModel


def init_orthogonal(m: nn.Module, gain: float = 0.01, activate: bool = False) -> nn.Module:
    g = nn.init.calculate_gain("relu") if activate else gain
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=g)
        nn.init.constant_(m.bias, 0)
    return m


class LocalAgentPolicies(nn.Module):
    """Per-agent local actors matching ma_gnn_transformer_new.py decoder MLP shape."""

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        continuous: bool,
        action_low: Optional[torch.Tensor] = None,
        action_high: Optional[torch.Tensor] = None,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.action_low = action_low
        self.action_high = action_high

        self.mlp_ = nn.ModuleList()
        for _ in range(num_agents):
            actor = nn.Sequential(
                nn.LayerNorm(obs_dim),
                init_orthogonal(nn.Linear(obs_dim, hidden_dim), activate=True),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                init_orthogonal(nn.Linear(hidden_dim, hidden_dim), activate=True),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                init_orthogonal(nn.Linear(hidden_dim, action_dim)),
            )
            self.mlp_.append(actor)

        if continuous:
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.log_std = None

    def _forward_logits(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, N, obs_dim]
        outs = []
        for n in range(self.num_agents):
            outs.append(self.mlp_[n](obs[:, n, :]))
        return torch.stack(outs, dim=1)

    def _clip_to_action_space(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_low is None or self.action_high is None:
            return action
        low = self.action_low.view(1, 1, -1).to(action.device)
        high = self.action_high.view(1, 1, -1).to(action.device)
        return torch.max(torch.min(action, high), low)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self._forward_logits(obs)

        if self.continuous:
            std = self.log_std.exp().view(1, 1, -1).expand_as(logits)
            dist = torch.distributions.Normal(logits, std)
            action = logits if deterministic else dist.rsample()
            action_clipped = self._clip_to_action_space(action)
            log_prob = dist.log_prob(action).sum(dim=-1)  # [B, N]
            entropy = dist.entropy().sum(dim=-1)  # [B, N]
            return action_clipped, log_prob, entropy

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)  # [B, N]
        entropy = dist.entropy()  # [B, N]
        return action.unsqueeze(-1).float(), log_prob, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-agent log-prob/entropy for provided actions.
        obs: [B, N, obs_dim], actions: [B, N, act_dim or 1]
        """
        logits = self._forward_logits(obs)
        if self.continuous:
            std = self.log_std.exp().view(1, 1, -1).expand_as(logits)
            dist = torch.distributions.Normal(logits, std)
            act = actions
            if act.shape[-1] != logits.shape[-1]:
                act = act[..., : logits.shape[-1]]
            log_prob = dist.log_prob(act).sum(dim=-1)  # [B, N]
            entropy = dist.entropy().sum(dim=-1)  # [B, N]
            return log_prob, entropy

        dist = torch.distributions.Categorical(logits=logits)
        act_idx = actions.squeeze(-1).long()
        log_prob = dist.log_prob(act_idx)  # [B, N]
        entropy = dist.entropy()  # [B, N]
        return log_prob, entropy


def _normalize_scenario(scenario: str) -> str:
    aliases = {
        "ManyAgentGoToGoalEnv": "ManyAgentGoToGoal-v0",
        "ManyAgentGoToGoalEnv-v0": "ManyAgentGoToGoal-v0",
        "ManyAgentGoToGoal": "ManyAgentGoToGoal-v0",
    }
    return aliases.get(scenario, scenario)


def _unpack_reset_out(reset_out, num_agents: int):
    if isinstance(reset_out, tuple):
        if len(reset_out) >= 3:
            obs, share_obs, _ = reset_out[0], reset_out[1], reset_out[2]
            return obs, share_obs
        if len(reset_out) == 2:
            obs = reset_out[0]
            return obs, obs
        if len(reset_out) == 1:
            obs = reset_out[0]
            return obs, obs
    return reset_out, reset_out


def _unpack_step_out(step_out):
    if isinstance(step_out, tuple):
        if len(step_out) >= 6:
            obs, share_obs, rew, done, info, _ = step_out[:6]
            return obs, share_obs, rew, done, info
        if len(step_out) == 5:
            obs, rew, terminated, truncated, info = step_out
            done = np.logical_or(terminated, truncated)
            return obs, obs, rew, done, info
        if len(step_out) == 4:
            obs, rew, done, info = step_out
            return obs, obs, rew, done, info
    raise RuntimeError("Unsupported step output format from env.")


def _as_agent_array(x, num_agents: int, dtype=np.float32):
    arr = np.asarray(x)
    if arr.ndim == 0:
        return np.full((num_agents,), arr.item(), dtype=dtype)
    if arr.ndim == 1 and arr.shape[0] == num_agents:
        return arr.astype(dtype)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
        if arr.shape[0] >= num_agents:
            return arr[:num_agents].astype(dtype)
    raise RuntimeError(f"Cannot convert value with shape {np.asarray(x).shape} to ({num_agents},).")


def _edge_index_to_adj(edge_index, num_agents: int) -> np.ndarray:
    ei = np.asarray(edge_index)
    if ei.ndim == 2 and ei.shape == (num_agents, num_agents):
        return ei.astype(np.float32)
    if ei.ndim == 2 and ei.shape[0] == 2:
        src, dst = ei[0], ei[1]
        valid = (src >= 0) & (dst >= 0) & (src < num_agents) & (dst < num_agents)
        adj = np.zeros((num_agents, num_agents), dtype=np.float32)
        if valid.any():
            adj[src[valid].astype(np.int64), dst[valid].astype(np.int64)] = 1.0
        return adj
    raise RuntimeError(f"Unsupported edge_index shape {ei.shape}.")


def _extract_rgb_frame(render_out: Any):
    if isinstance(render_out, np.ndarray):
        if render_out.ndim == 3:
            return render_out
        if render_out.ndim == 4 and render_out.shape[0] > 0:
            return render_out[0]
        return None
    if isinstance(render_out, (list, tuple)):
        for x in render_out:
            fr = _extract_rgb_frame(x)
            if fr is not None:
                return fr
    return None


def _extract_frame_from_env_canvas(env: Any):
    # Fallback for envs that render via matplotlib and return None.
    cand = [env, getattr(env, "unwrapped", None)]
    for obj in cand:
        if obj is None:
            continue
        for name in ("fig", "figure", "_fig", "_render_fig"):
            fig = getattr(obj, name, None)
            if fig is None:
                continue
            canvas = getattr(fig, "canvas", None)
            if canvas is None:
                continue
            try:
                canvas.draw()
                w, h = canvas.get_width_height()
                if hasattr(canvas, "tostring_rgb"):
                    buf = canvas.tostring_rgb()
                    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
                    return arr
                if hasattr(canvas, "buffer_rgba"):
                    buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
                    if buf.ndim == 3 and buf.shape[-1] >= 3:
                        return buf[..., :3].copy()
            except Exception:
                continue
    return None


class ManyAgentVecEnv:
    """Minimal synchronous vector wrapper for ManyAgentGoToGoal-style env API."""

    def __init__(self, scenario: str, num_envs: int, seed: int):
        self.scenario = _normalize_scenario(scenario)
        self.num_envs = num_envs
        self.envs = []
        for i in range(num_envs):
            try:
                env = gym.make(self.scenario, disable_env_checker=True, render_mode="rgb_array")
            except TypeError:
                # Some env versions may not expose render_mode in constructor.
                env = gym.make(self.scenario, disable_env_checker=True)
            if hasattr(env, "seed"):
                env.seed(seed + i * 1000)
            self.envs.append(env)

        first = self.envs[0]
        self.n_agents = int(getattr(first, "n_agents", 0))
        if self.n_agents <= 0:
            obs0, _ = _unpack_reset_out(first.reset(), 0)
            self.n_agents = int(np.asarray(obs0).shape[0])

        self.action_space = getattr(first, "action_space", None)
        self.observation_space = getattr(first, "observation_space", None)

    def _get_adj_single(self, env) -> np.ndarray:
        if hasattr(env, "get_visibility_matrix"):
            return np.asarray(env.get_visibility_matrix(), dtype=np.float32)
        if hasattr(env, "get_edge_index_matrix"):
            return _edge_index_to_adj(env.get_edge_index_matrix(), self.n_agents)
        return np.eye(self.n_agents, dtype=np.float32)

    def get_adjacency(self) -> np.ndarray:
        mats = [self._get_adj_single(env) for env in self.envs]
        return np.stack(mats, axis=0)

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        obs_list, share_list = [], []
        for env in self.envs:
            obs, share_obs = _unpack_reset_out(env.reset(), self.n_agents)
            obs_list.append(np.asarray(obs, dtype=np.float32))
            share_list.append(np.asarray(share_obs, dtype=np.float32))
        return np.stack(obs_list, axis=0), np.stack(share_list, axis=0)

    def step(self, actions: np.ndarray):
        obs_list, share_list, rew_list, done_list, info_list = [], [], [], [], []
        for i, env in enumerate(self.envs):
            out = _unpack_step_out(env.step(actions[i]))
            obs, share_obs, rew, done, info = out

            rew_arr = _as_agent_array(rew, self.n_agents, dtype=np.float32)
            done_arr = _as_agent_array(done, self.n_agents, dtype=np.float32)

            if np.all(done_arr > 0.5):
                obs, share_obs = _unpack_reset_out(env.reset(), self.n_agents)

            obs_list.append(np.asarray(obs, dtype=np.float32))
            share_list.append(np.asarray(share_obs, dtype=np.float32))
            rew_list.append(rew_arr)
            done_list.append(done_arr)
            info_list.append(info)

        return (
            np.stack(obs_list, axis=0),
            np.stack(share_list, axis=0),
            np.stack(rew_list, axis=0),
            np.stack(done_list, axis=0),
            info_list,
        )

    def close(self):
        for env in self.envs:
            env.close()

    def render_rgb_array(self) -> np.ndarray:
        frames = []
        for env in self.envs:
            fr = None
            target = getattr(env, "unwrapped", env)
            try:
                fr = target.render(mode="rgb_array")
            except TypeError:
                try:
                    fr = target.render()
                except Exception:
                    fr = None
            except Exception:
                fr = None

            frame = _extract_rgb_frame(fr)
            if frame is None:
                frame = _extract_frame_from_env_canvas(target)
            if frame is not None:
                frames.append(frame.astype(np.uint8, copy=False))
                continue

            # Fallback black frame if render output unavailable.
            frames.append(np.zeros((84, 84, 3), dtype=np.uint8))
        return np.stack(frames, axis=0)


@dataclass
class RolloutStep:
    obs: torch.Tensor  # [E, N, obs_dim]
    actions: torch.Tensor  # [E, N, act_dim or 1]
    adj: torch.Tensor  # [E, N, N]
    frame: torch.Tensor  # [E, H, W, 3] uint8
    log_prob_sum: torch.Tensor  # [E]
    entropy_mean: torch.Tensor  # [E]
    reward_mean: torch.Tensor  # [E]
    done_env: torch.Tensor  # [E]


class FixedRolloutBuffer:
    """Fixed-size ring buffer that stores online actor rollouts."""

    def __init__(self, capacity_steps: int):
        self.capacity_steps = capacity_steps
        self.steps: deque[RolloutStep] = deque(maxlen=capacity_steps)

    def __len__(self):
        return len(self.steps)

    def push(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        adj: torch.Tensor,
        frame: torch.Tensor,
        log_prob_sum: torch.Tensor,
        entropy_mean: torch.Tensor,
        reward_mean: torch.Tensor,
        done_env: torch.Tensor,
    ):
        self.steps.append(
            RolloutStep(
                obs=obs.detach().cpu(),
                actions=actions.detach().cpu(),
                adj=adj.detach().cpu(),
                frame=frame.detach().cpu(),
                log_prob_sum=log_prob_sum.detach().cpu(),
                entropy_mean=entropy_mean.detach().cpu(),
                reward_mean=reward_mean.detach().cpu(),
                done_env=done_env.detach().cpu(),
            )
        )

    def sample_clips(self, batch_size: int, clip_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
        if len(self.steps) < clip_len:
            raise RuntimeError(f"Need at least {clip_len} steps in rollout buffer, found {len(self.steps)}.")

        max_start = len(self.steps) - clip_len
        first = self.steps[0]
        num_envs = int(first.obs.shape[0])

        robot_obs = []
        actions = []
        adj = []
        videos = []
        log_prob_sum = []
        entropy_mean = []
        reward_mean = []

        for _ in range(batch_size):
            start = random.randint(0, max_start)
            env_idx = random.randint(0, num_envs - 1)
            clip_obs = []
            clip_adj = []
            clip_act = []
            clip_vid = []
            clip_logp = []
            clip_ent = []
            clip_rew = []

            terminal_obs = None
            terminal_act = None
            terminal_adj = None
            terminal_vid = None
            done_triggered = False
            for t in range(clip_len):
                s = self.steps[start + t]
                if done_triggered:
                    clip_obs.append(terminal_obs)
                    clip_act.append(terminal_act)
                    clip_adj.append(terminal_adj)
                    clip_vid.append(terminal_vid)
                    clip_logp.append(torch.zeros((), dtype=torch.float32))
                    clip_ent.append(torch.zeros((), dtype=torch.float32))
                    clip_rew.append(torch.zeros((), dtype=torch.float32))
                    continue

                o = s.obs[env_idx]
                u = s.actions[env_idx]
                a = s.adj[env_idx]
                v = s.frame[env_idx]
                clip_obs.append(o)
                clip_act.append(u)
                clip_adj.append(a)
                clip_vid.append(v)
                clip_logp.append(s.log_prob_sum[env_idx])
                clip_ent.append(s.entropy_mean[env_idx])
                clip_rew.append(s.reward_mean[env_idx])

                if bool(s.done_env[env_idx].item() > 0.5):
                    terminal_obs = o
                    terminal_act = u
                    terminal_adj = a
                    terminal_vid = v
                    done_triggered = True

            robot_obs.append(torch.stack(clip_obs, dim=0))
            actions.append(torch.stack(clip_act, dim=0))
            adj.append(torch.stack(clip_adj, dim=0))
            videos.append(torch.stack(clip_vid, dim=0))
            log_prob_sum.append(torch.stack(clip_logp).sum())
            entropy_mean.append(torch.stack(clip_ent).mean())
            reward_mean.append(torch.stack(clip_rew).mean())

        return {
            "robot_obs": torch.stack(robot_obs, dim=0).to(device),  # [B, T, N, D]
            "actions": torch.stack(actions, dim=0).to(device),  # [B, T, N, A]
            "adj": torch.stack(adj, dim=0).to(device),  # [B, T, N, N]
            "video": torch.stack(videos, dim=0),  # [B, T, H, W, 3] uint8 on CPU
            "log_prob_sum": torch.stack(log_prob_sum).to(device),  # [B]
            "entropy_mean": torch.stack(entropy_mean).to(device),  # [B]
            "reward_mean": torch.stack(reward_mean).to(device),  # [B]
        }


class CachedBlankVideoInputs:
    def __init__(
        self,
        processor,
        prompt: str,
        clip_len: int,
        frame_size: int = 224,
        vlm_max_text_len: int = 256,
        vlm_truncation: bool = False,
        vlm_padding: str = "longest",
    ):
        self.processor = processor
        self.prompt = prompt
        self.clip_len = clip_len
        self.frame_size = frame_size
        self.vlm_max_text_len = vlm_max_text_len
        self.vlm_truncation = vlm_truncation
        self.vlm_padding = vlm_padding
        self.cache = {}

    def get(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if batch_size in self.cache:
            return {k: v.clone() for k, v in self.cache[batch_size].items()}

        black = Image.fromarray(np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))
        videos = [[black for _ in range(self.clip_len)] for _ in range(batch_size)]
        inputs = preprocess_vlm_video_inputs(
            vlm_processor=self.processor,
            frames=videos,
            text=self.prompt,
            text_prompt_template=self.prompt,
            vlm_max_text_len=self.vlm_max_text_len,
            vlm_truncation=self.vlm_truncation,
            vlm_padding=self.vlm_padding,
            squeeze_batch_dim=False,
        )
        self.cache[batch_size] = {k: v for k, v in inputs.items()}
        return {k: v.clone() for k, v in self.cache[batch_size].items()}


def _resize_frame_uint8(frame: np.ndarray, size: int) -> np.ndarray:
    if frame.shape[0] == size and frame.shape[1] == size:
        return frame
    img = Image.fromarray(frame)
    img = img.resize((size, size), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def build_video_inputs_from_batch(
    processor,
    prompt: str,
    videos_uint8: torch.Tensor,
    frame_size: int,
    vlm_max_text_len: int,
    vlm_truncation: bool,
    vlm_padding: str,
) -> Dict[str, torch.Tensor]:
    # videos_uint8: [B, T, H, W, 3] on CPU
    v = videos_uint8.detach().cpu().numpy().astype(np.uint8)
    batch_videos = []
    for b in range(v.shape[0]):
        frames = []
        for t in range(v.shape[1]):
            frames.append(Image.fromarray(v[b, t]))
        batch_videos.append(frames)
    _ = frame_size
    return preprocess_vlm_video_inputs(
        vlm_processor=processor,
        frames=batch_videos,
        text=prompt,
        text_prompt_template=prompt,
        vlm_max_text_len=vlm_max_text_len,
        vlm_truncation=vlm_truncation,
        vlm_padding=vlm_padding,
        squeeze_batch_dim=False,
    )


def build_critic(args, device: torch.device) -> MultimodalValueModel:
    cfg = ModelConfig(
        vl_backend=args.vl_backend,
        vl_model_name=args.vl_model_name,
        vl_dtype=args.vl_dtype,
        vl_max_text_len=args.vl_max_text_len,
        freeze_vl=args.freeze_vl,
        quantization_config=getattr(args, "quantization_config", None),
        video_channels=3,
        video_height=args.video_size,
        video_width=args.video_size,
        video_frames=args.clip_len,
        video_preprocessed=True,
        num_robots=args.num_robots,
        robot_obs_dim=args.robot_obs_dim,
        text_dim=512,
        d_model=args.d_model,
        temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        temporal_dropout=args.temporal_dropout,
        gnn_layers=args.gnn_layers,
        fusion_hidden=args.fusion_hidden,
        use_moe=False,
        moe_experts=4,
        moe_top_k=2,
        debug_save_video=False,
    )
    model = MultimodalValueModel(cfg, device=device)
    return model


def _parse_lora_targets(args):
    if args.lora_target_modules:
        return [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _apply_peft(model, args):
    if args.peft == "none":
        return model
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as e:
        raise RuntimeError("PEFT requested but 'peft' is not installed. `pip install peft`.") from e

    for p in model.backbone.model.parameters():
        p.requires_grad = False

    if args.peft == "qlora":
        model.backbone.model = prepare_model_for_kbit_training(model.backbone.model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=_parse_lora_targets(args),
        task_type="CAUSAL_LM",
    )
    model.backbone.model = get_peft_model(model.backbone.model, lora_cfg)
    return model


def _space_info(action_space, num_agents: int):
    if isinstance(action_space, (list, tuple)):
        space = next((s for s in action_space if s is not None), None)
    elif isinstance(action_space, dict):
        space = next((v for v in action_space.values() if v is not None), None)
    else:
        space = action_space

    if space is None:
        raise RuntimeError("Could not infer action space from environment (got None).")

    if hasattr(space, "n"):
        return False, int(space.n), None, None
    if hasattr(space, "shape"):
        action_dim = int(np.prod(space.shape))
        low = torch.tensor(np.asarray(space.low).reshape(-1), dtype=torch.float32)
        high = torch.tensor(np.asarray(space.high).reshape(-1), dtype=torch.float32)
        return True, action_dim, low, high

    raise RuntimeError("Unsupported action space type.")


def _space_info_with_overrides(envs: ManyAgentVecEnv, args, num_agents: int):
    if args.action_type != "auto":
        if args.action_type == "discrete":
            return False, int(args.action_dim), None, None
        return True, int(args.action_dim), None, None

    try:
        return _space_info(envs.action_space, num_agents)
    except Exception:
        if args.action_type == "auto":
            # Fallback for envs that do not expose gym-compatible action_space.
            if args.action_dim <= 0:
                raise
            return True, int(args.action_dim), None, None
        raise


def _to_device_inputs(inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def _get_expert_batch(loader_iter, loader):
    try:
        batch = next(loader_iter)
        return batch, loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
        return batch, loader_iter


def _extract_video_tensor_from_inputs(inputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    for key in ("pixel_values_videos", "pixel_values", "video_values", "videos", "video"):
        val = inputs.get(key, None)
        if torch.is_tensor(val):
            return val
    return None


def _expert_batch_sanity_report(batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    report = {
        "clips": 0.0,
        "mean_temporal_diff": 0.0,
        "near_static_clip_frac": 0.0,
        "done_rate": 0.0,
        "done_any_rate": 0.0,
        "done_all_rate": 0.0,
        "done_any_not_all_rate": 0.0,
    }

    if "done" in batch:
        done = batch["done"].detach().float()
        report["done_rate"] = float(done.mean().item())
    if "done_any" in batch:
        done_any = batch["done_any"].detach().float()
        report["done_any_rate"] = float(done_any.mean().item())
    if "done_all" in batch:
        done_all = batch["done_all"].detach().float()
        report["done_all_rate"] = float(done_all.mean().item())
    if "done_any" in batch and "done_all" in batch:
        done_any = batch["done_any"].detach().float()
        done_all = batch["done_all"].detach().float()
        report["done_any_not_all_rate"] = float(((done_any > 0.5) & (done_all < 0.5)).float().mean().item())

    video = _extract_video_tensor_from_inputs(batch.get("inputs", {}))
    if video is None or video.ndim < 4:
        return report

    x = video.detach().float().cpu()
    if x.ndim == 4:
        x = x.unsqueeze(0)
    if x.ndim < 5:
        return report

    bsz = float(x.shape[0])
    report["clips"] = bsz
    tdim = 1
    if x.shape[tdim] < 2:
        return report

    diffs = (x[:, 1:] - x[:, :-1]).abs().flatten(start_dim=2).mean(dim=2)
    clip_temporal = diffs.mean(dim=1)
    report["mean_temporal_diff"] = float(clip_temporal.mean().item())
    report["near_static_clip_frac"] = float((clip_temporal < 1e-3).float().mean().item())
    return report


def _run_expert_loader_sanity_check(loader, num_batches: int, accelerator) -> None:
    if num_batches <= 0:
        return
    it = iter(loader)
    reports = []
    for _ in range(num_batches):
        try:
            b = next(it)
        except StopIteration:
            break
        reports.append(_expert_batch_sanity_report(b))
    if not reports:
        accelerator.print("[expert_sanity] no batches available.")
        return

    keys = reports[0].keys()
    mean_report = {k: float(np.mean([r[k] for r in reports])) for k in keys}
    accelerator.print(
        "[expert_sanity] "
        f"batches={len(reports)} clips_per_batch={mean_report['clips']:.2f} "
        f"mean_temporal_diff={mean_report['mean_temporal_diff']:.6f} "
        f"near_static_clip_frac={mean_report['near_static_clip_frac']:.3f} "
        f"done_rate={mean_report['done_rate']:.3f} "
        f"done_any_rate={mean_report['done_any_rate']:.3f} "
        f"done_all_rate={mean_report['done_all_rate']:.3f} "
        f"done_any_not_all_rate={mean_report['done_any_not_all_rate']:.3f}"
    )


def _feature_compactness_loss(features: torch.Tensor) -> torch.Tensor:
    # Encourage expert features to stay close in latent space.
    if features.ndim != 2:
        features = features.view(features.shape[0], -1)
    if features.shape[0] < 2:
        return features.sum() * 0.0
    feats = F.normalize(features, dim=-1)
    dists = torch.cdist(feats, feats, p=2)
    mask = ~torch.eye(dists.shape[0], dtype=torch.bool, device=dists.device)
    return dists[mask].mean()


def _feature_separation_loss(policy_features: torch.Tensor, expert_center: torch.Tensor, margin: float) -> torch.Tensor:
    # Encourage policy features to stay farther than a margin from expert center.
    if policy_features.ndim != 2:
        policy_features = policy_features.view(policy_features.shape[0], -1)
    if policy_features.shape[0] == 0:
        return policy_features.sum() * 0.0
    pol = F.normalize(policy_features, dim=-1)
    center = F.normalize(expert_center, dim=-1)
    dists = torch.cdist(pol, center, p=2).squeeze(-1)
    return F.softplus(margin - dists).mean()


def parse_args():
    p = argparse.ArgumentParser()

    # Env / rollout
    p.add_argument("--scenario", type=str, default="ManyAgentGoToGoal-v0")
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rollout_steps", type=int, default=128)
    p.add_argument("--rollout_buffer_steps", type=int, default=4096)
    p.add_argument("--policy_video_source", type=str, default="env", choices=["env", "blank"])
    p.add_argument("--frame_store_size", type=int, default=84)

    # Expert data
    p.add_argument("--train_shards", type=str, required=True)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--expert_batch_size", type=int, default=2)
    p.add_argument("--expert_done_reduce", type=str, default="all", choices=["any", "all", "mean", "sum", "first"])
    p.add_argument("--expert_sanity_batches", type=int, default=3, help="Run loader sanity checks on N expert batches at startup.")
    p.add_argument("--text_prompt_template", type=str, default="You are a critic model. <video><obs> Evaluate team performance.")

    # Adversarial IRL
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--clip_len", type=int, default=10)
    p.add_argument("--critic_updates", type=int, default=2)
    p.add_argument("--actor_updates", type=int, default=1)
    p.add_argument("--policy_batch_size", type=int, default=8)
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--score_scale", type=float, default=1.0)
    p.add_argument("--disc_tanh_temp", type=float, default=100.0)
    p.add_argument("--raw_score_l2_coef", type=float, default=1e-4)
    p.add_argument(
        "--lambda_feat_contrastive",
        type=float,
        default=0.1,
        help="Weight for auxiliary critic feature-space contrastive shaping.",
    )
    p.add_argument(
        "--feat_contrastive_margin",
        type=float,
        default=1.0,
        help="Distance margin used by policy-vs-expert feature separation loss.",
    )

    # Optim
    p.add_argument("--critic_lr", type=float, default=3e-5)
    p.add_argument("--actor_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--critic_grad_clip", type=float, default=1.0)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--ddp_find_unused_parameters", action="store_true")
    p.add_argument("--grad_accum_steps", type=int, default=1)

    # Critic model
    p.add_argument("--vl_backend", type=str, default="llava_video")
    p.add_argument("--vl_model_name", type=str, default="llava-hf/LLaVA-NeXT-Video-7B-hf")
    p.add_argument("--vl_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--vl_max_text_len", type=int, default=256)
    p.add_argument("--freeze_vl", action="store_true")
    p.add_argument("--video_size", type=int, default=224)
    p.add_argument("--robot_obs_dim", type=int, default=40)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--temporal_layers", type=int, default=2)
    p.add_argument("--temporal_heads", type=int, default=4)
    p.add_argument("--temporal_dropout", type=float, default=0.1)
    p.add_argument("--gnn_layers", type=int, default=2)
    p.add_argument("--fusion_hidden", type=int, default=512)
    p.add_argument("--peft", type=str, default="none", choices=["none", "lora", "qlora"])
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="")
    p.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])

    # Policy
    p.add_argument("--policy_hidden_dim", type=int, default=256)
    p.add_argument("--action_type", type=str, default="auto", choices=["auto", "continuous", "discrete"])
    p.add_argument("--action_dim", type=int, default=2)

    # Logging / checkpoint
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--save_dir", type=str, default="checkpoints_irl_local")
    p.add_argument("--eval_interval", type=int, default=25)
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument("--eval_max_episode_steps", type=int, default=500)
    p.add_argument("--eval_num_envs", type=int, default=1)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="ma-vlcm-irl")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_run_name", type=str, default="")

    return p.parse_args()


def collect_rollout(
    envs: ManyAgentVecEnv,
    actors: LocalAgentPolicies,
    buffer: FixedRolloutBuffer,
    obs_np: np.ndarray,
    device: torch.device,
    args,
    rollout_steps: int,
) -> np.ndarray:
    obs = obs_np
    total_reward = 0.0
    for _ in range(rollout_steps):
        frame_np = envs.render_rgb_array()  # [E,H,W,3]
        if frame_np.shape[1] != args.frame_store_size or frame_np.shape[2] != args.frame_store_size:
            resized = [
                _resize_frame_uint8(frame_np[i], args.frame_store_size)
                for i in range(frame_np.shape[0])
            ]
            frame_np = np.stack(resized, axis=0)
        adj = envs.get_adjacency()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            act_t, logp_t, ent_t = actors.act(obs_t, deterministic=False)

        act_np = act_t.detach().cpu().numpy()
        if act_np.shape[-1] == 1:
            act_env = act_np.squeeze(-1).astype(np.int64)
        else:
            act_env = act_np

        next_obs, _, rewards, dones, _ = envs.step(act_env)
        reward_mean = torch.tensor(rewards.mean(axis=1), dtype=torch.float32)
        total_reward += reward_mean.sum().item()
        done_env = torch.tensor((dones.mean(axis=1) > 0.5).astype(np.float32), dtype=torch.float32)

        buffer.push(
            obs=torch.tensor(obs, dtype=torch.float32),
            actions=act_t.detach().cpu(),
            adj=torch.tensor(adj, dtype=torch.float32),
            frame=torch.tensor(frame_np, dtype=torch.uint8),
            log_prob_sum=logp_t.sum(dim=1).detach().cpu(),
            entropy_mean=ent_t.mean(dim=1).detach().cpu(),
            reward_mean=reward_mean,
            done_env=done_env,
        )
        obs = next_obs
    
    print(f"Collected rollout with total reward {total_reward:.2f}")
    return obs


@torch.no_grad()
def evaluate_policy(
    envs: ManyAgentVecEnv,
    actors: LocalAgentPolicies,
    device: torch.device,
    num_episodes: int,
    max_episode_steps: int,
) -> float:
    """Deterministic eval: average trajectory reward over agents."""
    was_training = actors.training
    actors.eval()
    ep_returns = []

    for _ in range(num_episodes):
        obs_all, _ = envs.reset()
        obs = np.asarray(obs_all[0], dtype=np.float32)
        ep_ret = 0.0

        for _ in range(max_episode_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action_t, _, _ = actors.act(obs_t, deterministic=True)
            action_np = action_t.squeeze(0).detach().cpu().numpy()
            if action_np.shape[-1] == 1:
                action_env = action_np.squeeze(-1).astype(np.int64)
            else:
                action_env = action_np

            next_obs, _, rewards, dones, _ = envs.step(np.expand_dims(action_env, axis=0))
            rew_vec = np.asarray(rewards[0]).reshape(-1)
            ep_ret += float(rew_vec.mean() if rew_vec.size > 0 else 0.0)

            obs = np.asarray(next_obs[0], dtype=np.float32)
            done_env = bool(np.asarray(dones[0]).reshape(-1).mean() > 0.5)
            if done_env:
                break

        ep_returns.append(ep_ret)

    if was_training:
        actors.train()
    return float(np.mean(ep_returns)) if ep_returns else 0.0


def main():
    args = parse_args()
    if args.peft == "qlora":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError("QLoRA requested but bitsandbytes/transformers are not available.") from e
        if args.vl_dtype == "float16":
            compute_dtype = torch.float16
        elif args.vl_dtype == "float32":
            compute_dtype = torch.float32
        else:
            compute_dtype = torch.bfloat16
        args.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        args.quantization_config = None

    os.makedirs(args.save_dir, exist_ok=True)
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(args.ddp_find_unused_parameters)
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=max(1, args.grad_accum_steps),
    )
    random.seed(args.seed + accelerator.process_index)
    np.random.seed(args.seed + accelerator.process_index)
    torch.manual_seed(args.seed + accelerator.process_index)

    device = accelerator.device
    accelerator.print(f"device={device}")
    accelerator.print(f"scenario={args.scenario}")
    if args.wandb and accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity if args.wandb_entity else None),
            name=(args.wandb_run_name if args.wandb_run_name else None),
            config=vars(args),
        )
    if ManyAgent_GoTOGoal is None and accelerator.is_main_process:
        print("[WARN] Could not import ManyAgent_GoTOGoal package in this environment.")

    envs = ManyAgentVecEnv(args.scenario, args.num_envs, args.seed)
    eval_envs = None
    if accelerator.is_main_process and args.eval_interval > 0 and args.eval_episodes > 0:
        eval_envs = ManyAgentVecEnv(args.scenario, args.eval_num_envs, args.seed + 100000)
    obs, _ = envs.reset()
    num_agents = envs.n_agents
    obs_dim = int(obs.shape[-1])
    args.num_robots = num_agents

    is_continuous, action_dim, action_low, action_high = _space_info_with_overrides(envs, args, num_agents)
    accelerator.print(f"num_agents={num_agents} obs_dim={obs_dim} action_dim={action_dim} continuous={is_continuous}")

    critic = build_critic(args, device)
    critic = _apply_peft(critic, args)
    critic.train()

    tokenizer = getattr(critic.backbone.processor, "tokenizer", None)
    if tokenizer is not None and "<obs>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<obs>"]})
        if hasattr(critic.backbone.model, "resize_token_embeddings"):
            critic.backbone.model.resize_token_embeddings(len(tokenizer))

    actors = LocalAgentPolicies(
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        continuous=is_continuous,
        action_low=action_low,
        action_high=action_high,
        hidden_dim=args.policy_hidden_dim,
    ).to(device)
    actors.train()

    critic_trainable = [p for p in critic.parameters() if p.requires_grad]
    if len(critic_trainable) == 0:
        raise RuntimeError("No trainable critic parameters found. Check PEFT configuration.")
    critic_opt = torch.optim.AdamW(critic_trainable, lr=args.critic_lr, weight_decay=args.weight_decay)
    actor_opt = torch.optim.AdamW(actors.parameters(), lr=args.actor_lr, weight_decay=0.0)

    expert_loader_args = SimpleNamespace(
        clip_len=args.clip_len,
        clip_stride=max(1, args.clip_len),
        text_mode="raw",
        robot_source="obs",
        reward_reduce="mean",
        done_reduce=args.expert_done_reduce,
        preprocess_in_loader=True,
        vl_model_name=args.vl_model_name,
        text_prompt_template=args.text_prompt_template,
        loss_type="contrastive",
        return_mode="nstep",
        vl_backend=args.vl_backend,
        vl_max_text_len=args.vl_max_text_len,
    )
    expert_loader = webdataset_loader(expert_loader_args, args.train_shards, args.expert_batch_size, args.num_workers)
    # Keep actors unwrapped to call custom methods (act/evaluate_actions) directly.
    # Critic is the heavy model and benefits most from DDP wrapping.
    critic, critic_opt, expert_loader = accelerator.prepare(
        critic, critic_opt, expert_loader
    )
    _run_expert_loader_sanity_check(expert_loader, args.expert_sanity_batches, accelerator)
    expert_iter = iter(expert_loader)
    critic_raw = accelerator.unwrap_model(critic)
    vlm_truncation = args.vl_backend != "llava_video"
    vlm_padding = "longest" if args.vl_backend == "llava_video" else "max_length"

    rollout_buffer = FixedRolloutBuffer(args.rollout_buffer_steps)
    blank_input_builder = CachedBlankVideoInputs(
        processor=critic_raw.backbone.processor,
        prompt=args.text_prompt_template,
        clip_len=args.clip_len,
        frame_size=args.video_size,
        vlm_max_text_len=args.vl_max_text_len,
        vlm_truncation=vlm_truncation,
        vlm_padding=vlm_padding,
    )

    obs = collect_rollout(
        envs, actors, rollout_buffer, obs, device, args, rollout_steps=max(args.rollout_steps, args.clip_len)
    )

    for it in range(1, args.iters + 1):
        obs = collect_rollout(envs, actors, rollout_buffer, obs, device, args, rollout_steps=args.rollout_steps)

        critic_losses = []
        actor_losses = []
        expert_scores_log = []
        policy_scores_log = []
        expert_scores_raw_log = []
        policy_scores_raw_log = []
        expert_feat_loss_log = []
        policy_feat_loss_log = []

        for _ in range(args.critic_updates):
            expert_batch, expert_iter = _get_expert_batch(expert_iter, expert_loader)
            expert_inputs = _to_device_inputs(expert_batch["inputs"], device)
            expert_robot_obs = expert_batch["robot_obs"].to(device)
            expert_adj = expert_batch["adj"].to(device)

            policy_batch = rollout_buffer.sample_clips(args.policy_batch_size, args.clip_len, device=device)
            if args.policy_video_source == "env":
                policy_inputs = build_video_inputs_from_batch(
                    processor=critic_raw.backbone.processor,
                    prompt=args.text_prompt_template,
                    videos_uint8=policy_batch["video"],
                    frame_size=args.video_size,
                    vlm_max_text_len=args.vl_max_text_len,
                    vlm_truncation=vlm_truncation,
                    vlm_padding=vlm_padding,
                )
                policy_inputs = _to_device_inputs(policy_inputs, device)
            else:
                policy_inputs = _to_device_inputs(blank_input_builder.get(args.policy_batch_size), device)

            with accelerator.accumulate(critic):
                # Do separate backward passes to avoid cross-forward autograd version conflicts
                # on integer mask/index tensors under DDP.
                use_feat_aux = args.lambda_feat_contrastive > 0.0
                expert_out = critic(
                    expert_inputs,
                    expert_robot_obs,
                    expert_adj,
                    return_features=use_feat_aux,
                    debug_video=False,
                )
                if isinstance(expert_out, dict):
                    expert_scores_raw = expert_out["value"]
                    expert_features = expert_out["vlm_feature"]
                else:
                    expert_scores_raw = expert_out
                    expert_features = None
                # expert_scores = torch.tanh(expert_scores_raw / args.disc_tanh_temp)
                expert_scores = expert_scores_raw
                expert_term = -expert_scores.mean() #+ args.raw_score_l2_coef * expert_scores_raw.pow(2).mean()
                # expert_feat_loss = expert_scores_raw.sum() * 0.0
                # expert_center = None
                # if use_feat_aux and expert_features is not None:
                #     expert_feat_loss = _feature_compactness_loss(expert_features)
                #     expert_center = expert_features.detach().mean(dim=0, keepdim=True)
                    # expert_term = expert_term + args.lambda_feat_contrastive * expert_feat_loss
                accelerator.backward(expert_term)

                policy_out = critic(
                    policy_inputs,
                    policy_batch["robot_obs"],
                    policy_batch["adj"],
                    return_features=use_feat_aux,
                )
                if isinstance(policy_out, dict):
                    policy_scores_raw = policy_out["value"]
                    policy_features = policy_out["vlm_feature"]
                else:
                    policy_scores_raw = policy_out
                    policy_features = None
                # policy_scores = torch.tanh(policy_scores_raw / args.disc_tanh_temp)
                policy_scores = policy_scores_raw
                policy_term = policy_scores.mean() #+ args.raw_score_l2_coef * policy_scores_raw.pow(2).mean()
                # policy_feat_loss = policy_scores_raw.sum() * 0.0
                # if use_feat_aux and policy_features is not None and expert_center is not None:
                #     policy_feat_loss = _feature_separation_loss(
                #         policy_features, expert_center, margin=args.feat_contrastive_margin
                #     )
                #     policy_term = policy_term + args.lambda_feat_contrastive * policy_feat_loss
                accelerator.backward(policy_term)

                if accelerator.sync_gradients and args.critic_grad_clip > 0:
                    accelerator.clip_grad_norm_(critic.parameters(), args.critic_grad_clip)
                if accelerator.sync_gradients:
                    critic_opt.step()
                    critic_opt.zero_grad(set_to_none=True)
                critic_loss = expert_term + policy_term

            critic_losses.append(float(critic_loss.item()))
            expert_scores_log.append(float(expert_scores.mean().item()))
            policy_scores_log.append(float(policy_scores.mean().item()))
            expert_scores_raw_log.append(float(expert_scores_raw.mean().item()))
            policy_scores_raw_log.append(float(policy_scores_raw.mean().item()))
            # expert_feat_loss_log.append(float(expert_feat_loss.item()))
            # policy_feat_loss_log.append(float(policy_feat_loss.item()))

        for _ in range(args.actor_updates):
            policy_batch = rollout_buffer.sample_clips(args.policy_batch_size, args.clip_len, device=device)
            if args.policy_video_source == "env":
                policy_inputs = build_video_inputs_from_batch(
                    processor=critic_raw.backbone.processor,
                    prompt=args.text_prompt_template,
                    videos_uint8=policy_batch["video"],
                    frame_size=args.video_size,
                    vlm_max_text_len=args.vl_max_text_len,
                    vlm_truncation=vlm_truncation,
                    vlm_padding=vlm_padding,
                )
                policy_inputs = _to_device_inputs(policy_inputs, device)
            else:
                policy_inputs = _to_device_inputs(blank_input_builder.get(args.policy_batch_size), device)

            with accelerator.accumulate(actors):
                with torch.no_grad():
                    score = critic(policy_inputs, policy_batch["robot_obs"], policy_batch["adj"])
                    # score = torch.tanh((score / args.disc_tanh_temp) * args.score_scale)
                    score = score * args.score_scale

                obs_seq = policy_batch["robot_obs"]   # [B, T, N, D]
                act_seq = policy_batch["actions"]     # [B, T, N, A]
                bsz, tlen = obs_seq.shape[0], obs_seq.shape[1]
                obs_flat = obs_seq.reshape(bsz * tlen, obs_seq.shape[2], obs_seq.shape[3])
                act_flat = act_seq.reshape(bsz * tlen, act_seq.shape[2], act_seq.shape[3])
                logp_step, ent_step = actors.evaluate_actions(obs_flat, act_flat)  # [B*T, N]
                logp = logp_step.sum(dim=1).view(bsz, tlen).sum(dim=1)  # [B]
                entropy = ent_step.mean(dim=1).view(bsz, tlen).mean(dim=1)  # [B]

                actor_loss = -(logp * score.detach()).mean() - args.entropy_coef * entropy.mean()

                accelerator.backward(actor_loss)
                if accelerator.sync_gradients:
                    actor_opt.step()
                    actor_opt.zero_grad(set_to_none=True)
                actor_losses.append(float(actor_loss.item()))

        if it % args.log_every == 0 and accelerator.is_main_process:
            rew_mean = float(torch.stack([s.reward_mean.mean() for s in rollout_buffer.steps]).mean().item())
            metrics = {
                "iter": it,
                "train/critic_loss": float(np.mean(critic_losses)),
                "train/actor_loss": float(np.mean(actor_losses)),
                "train/expert_score": float(np.mean(expert_scores_log)),
                "train/policy_score": float(np.mean(policy_scores_log)),
                "train/raw_expert_score": float(np.mean(expert_scores_raw_log)),
                "train/raw_policy_score": float(np.mean(policy_scores_raw_log)),
                "train/expert_feat_loss": float(np.mean(expert_feat_loss_log)),
                "train/policy_feat_loss": float(np.mean(policy_feat_loss_log)),
                "train/buffer_reward_mean": rew_mean,
            }
            print(
                f"iter={it} "
                f"critic_loss={metrics['train/critic_loss']:.4f} "
                f"actor_loss={metrics['train/actor_loss']:.4f} "
                f"expert_score={metrics['train/expert_score']:.4f} "
                f"policy_score={metrics['train/policy_score']:.4f} "
                f"raw_expert_score={metrics['train/raw_expert_score']:.4f} "
                f"raw_policy_score={metrics['train/raw_policy_score']:.4f} "
                f"expert_feat_loss={metrics['train/expert_feat_loss']:.4f} "
                f"policy_feat_loss={metrics['train/policy_feat_loss']:.4f} "
                f"buffer_reward_mean={metrics['train/buffer_reward_mean']:.4f}"
            )
            if args.wandb:
                wandb.log(metrics, step=it)

        if (
            accelerator.is_main_process
            and eval_envs is not None
            and args.eval_interval > 0
            and (it % args.eval_interval == 0)
        ):
            eval_avg_traj_reward = evaluate_policy(
                envs=eval_envs,
                actors=actors,
                device=device,
                num_episodes=args.eval_episodes,
                max_episode_steps=args.eval_max_episode_steps,
            )
            print(
                f"iter={it} "
                f"eval_avg_traj_reward={eval_avg_traj_reward:.4f} "
                f"(episodes={args.eval_episodes})"
            )
            if args.wandb:
                wandb.log({"eval/avg_traj_reward": eval_avg_traj_reward, "iter": it}, step=it)

        if it % args.save_every == 0 and accelerator.is_main_process:
            ckpt = {
                "iter": it,
                "critic": accelerator.unwrap_model(critic).state_dict(),
                "actors": accelerator.unwrap_model(actors).state_dict(),
                "critic_opt": critic_opt.state_dict(),
                "actor_opt": actor_opt.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, f"{args.save_dir}/irl_local_iter_{it}.pt")

    envs.close()
    if eval_envs is not None:
        eval_envs.close()
    if args.wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
