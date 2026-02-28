import argparse
import os
import random
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

try:
    import ManyAgent_GoTOGoal  # noqa: F401
except Exception:
    ManyAgent_GoTOGoal = None

from data_loading import webdataset_loader
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


class ManyAgentVecEnv:
    """Minimal synchronous vector wrapper for ManyAgentGoToGoal-style env API."""

    def __init__(self, scenario: str, num_envs: int, seed: int):
        self.scenario = _normalize_scenario(scenario)
        self.num_envs = num_envs
        self.envs = []
        for i in range(num_envs):
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


@dataclass
class RolloutStep:
    obs: torch.Tensor  # [E, N, obs_dim]
    actions: torch.Tensor  # [E, N, act_dim or 1]
    adj: torch.Tensor  # [E, N, N]
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
        log_prob_sum = []
        entropy_mean = []
        reward_mean = []

        for _ in range(batch_size):
            start = random.randint(0, max_start)
            env_idx = random.randint(0, num_envs - 1)
            clip_obs = []
            clip_adj = []
            clip_act = []
            clip_logp = []
            clip_ent = []
            clip_rew = []

            terminal_obs = None
            terminal_act = None
            terminal_adj = None
            done_triggered = False
            for t in range(clip_len):
                s = self.steps[start + t]
                if done_triggered:
                    clip_obs.append(terminal_obs)
                    clip_act.append(terminal_act)
                    clip_adj.append(terminal_adj)
                    clip_logp.append(torch.zeros((), dtype=torch.float32))
                    clip_ent.append(torch.zeros((), dtype=torch.float32))
                    clip_rew.append(torch.zeros((), dtype=torch.float32))
                    continue

                o = s.obs[env_idx]
                u = s.actions[env_idx]
                a = s.adj[env_idx]
                clip_obs.append(o)
                clip_act.append(u)
                clip_adj.append(a)
                clip_logp.append(s.log_prob_sum[env_idx])
                clip_ent.append(s.entropy_mean[env_idx])
                clip_rew.append(s.reward_mean[env_idx])

                if bool(s.done_env[env_idx].item() > 0.5):
                    terminal_obs = o
                    terminal_act = u
                    terminal_adj = a
                    done_triggered = True

            robot_obs.append(torch.stack(clip_obs, dim=0))
            actions.append(torch.stack(clip_act, dim=0))
            adj.append(torch.stack(clip_adj, dim=0))
            log_prob_sum.append(torch.stack(clip_logp).sum())
            entropy_mean.append(torch.stack(clip_ent).mean())
            reward_mean.append(torch.stack(clip_rew).mean())

        return {
            "robot_obs": torch.stack(robot_obs, dim=0).to(device),  # [B, T, N, D]
            "actions": torch.stack(actions, dim=0).to(device),  # [B, T, N, A]
            "adj": torch.stack(adj, dim=0).to(device),  # [B, T, N, N]
            "log_prob_sum": torch.stack(log_prob_sum).to(device),  # [B]
            "entropy_mean": torch.stack(entropy_mean).to(device),  # [B]
            "reward_mean": torch.stack(reward_mean).to(device),  # [B]
        }


class CachedBlankVideoInputs:
    def __init__(self, processor, prompt: str, clip_len: int, frame_size: int = 224):
        self.processor = processor
        self.prompt = prompt
        self.clip_len = clip_len
        self.frame_size = frame_size
        self.cache = {}

    def get(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if batch_size in self.cache:
            return {k: v.clone() for k, v in self.cache[batch_size].items()}

        black = Image.fromarray(np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))
        videos = [[black for _ in range(self.clip_len)] for _ in range(batch_size)]
        text = [self.prompt for _ in range(batch_size)]
        inputs = self.processor(text=text, videos=videos, return_tensors="pt", padding="longest", truncation=False)
        self.cache[batch_size] = {k: v for k, v in dict(inputs).items()}
        return {k: v.clone() for k, v in self.cache[batch_size].items()}


def build_critic(args, device: torch.device) -> MultimodalValueModel:
    cfg = ModelConfig(
        vl_backend=args.vl_backend,
        vl_model_name=args.vl_model_name,
        vl_dtype=args.vl_dtype,
        vl_max_text_len=args.vl_max_text_len,
        freeze_vl=args.freeze_vl,
        quantization_config=None,
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


def parse_args():
    p = argparse.ArgumentParser()

    # Env / rollout
    p.add_argument("--scenario", type=str, default="ManyAgentGoToGoal-v0")
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rollout_steps", type=int, default=128)
    p.add_argument("--rollout_buffer_steps", type=int, default=4096)

    # Expert data
    p.add_argument("--train_shards", type=str, required=True)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--expert_batch_size", type=int, default=2)
    p.add_argument("--text_prompt_template", type=str, default="You are a critic model. <video><obs> Evaluate team performance.")

    # Adversarial IRL
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--clip_len", type=int, default=10)
    p.add_argument("--critic_updates", type=int, default=2)
    p.add_argument("--actor_updates", type=int, default=1)
    p.add_argument("--policy_batch_size", type=int, default=8)
    p.add_argument("--entropy_coef", type=float, default=0.001)
    p.add_argument("--score_scale", type=float, default=1.0)
    p.add_argument("--disc_tanh_temp", type=float, default=100.0)
    p.add_argument("--raw_score_l2_coef", type=float, default=1e-4)

    # Optim
    p.add_argument("--critic_lr", type=float, default=3e-5)
    p.add_argument("--actor_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--critic_grad_clip", type=float, default=1.0)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--ddp_find_unused_parameters", action="store_true")

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

    return p.parse_args()


def collect_rollout(
    envs: ManyAgentVecEnv,
    actors: LocalAgentPolicies,
    buffer: FixedRolloutBuffer,
    obs_np: np.ndarray,
    device: torch.device,
    rollout_steps: int,
) -> np.ndarray:
    obs = obs_np
    for _ in range(rollout_steps):
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
        done_env = torch.tensor((dones.mean(axis=1) > 0.5).astype(np.float32), dtype=torch.float32)

        buffer.push(
            obs=torch.tensor(obs, dtype=torch.float32),
            actions=act_t.detach().cpu(),
            adj=torch.tensor(adj, dtype=torch.float32),
            log_prob_sum=logp_t.sum(dim=1).detach().cpu(),
            entropy_mean=ent_t.mean(dim=1).detach().cpu(),
            reward_mean=reward_mean,
            done_env=done_env,
        )
        obs = next_obs
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
    os.makedirs(args.save_dir, exist_ok=True)
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(args.ddp_find_unused_parameters)
    )
    accelerator = Accelerator(mixed_precision=args.mixed_precision, kwargs_handlers=[ddp_kwargs])
    random.seed(args.seed + accelerator.process_index)
    np.random.seed(args.seed + accelerator.process_index)
    torch.manual_seed(args.seed + accelerator.process_index)

    device = accelerator.device
    accelerator.print(f"device={device}")
    accelerator.print(f"scenario={args.scenario}")
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

    critic_opt = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay)
    actor_opt = torch.optim.AdamW(actors.parameters(), lr=args.actor_lr, weight_decay=0.0)

    expert_loader_args = SimpleNamespace(
        clip_len=args.clip_len,
        clip_stride=max(1, args.clip_len),
        text_mode="raw",
        robot_source="obs",
        reward_reduce="mean",
        done_reduce="any",
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
    expert_iter = iter(expert_loader)
    critic_raw = accelerator.unwrap_model(critic)

    rollout_buffer = FixedRolloutBuffer(args.rollout_buffer_steps)
    blank_input_builder = CachedBlankVideoInputs(
        processor=critic_raw.backbone.processor,
        prompt=args.text_prompt_template,
        clip_len=args.clip_len,
        frame_size=args.video_size,
    )

    obs = collect_rollout(envs, actors, rollout_buffer, obs, device, rollout_steps=max(args.rollout_steps, args.clip_len))

    for it in range(1, args.iters + 1):
        obs = collect_rollout(envs, actors, rollout_buffer, obs, device, rollout_steps=args.rollout_steps)

        critic_losses = []
        actor_losses = []
        expert_scores_log = []
        policy_scores_log = []
        expert_scores_raw_log = []
        policy_scores_raw_log = []

        for _ in range(args.critic_updates):
            expert_batch, expert_iter = _get_expert_batch(expert_iter, expert_loader)
            expert_inputs = _to_device_inputs(expert_batch["inputs"], device)
            expert_robot_obs = expert_batch["robot_obs"].to(device)
            expert_adj = expert_batch["adj"].to(device)

            policy_batch = rollout_buffer.sample_clips(args.policy_batch_size, args.clip_len, device=device)
            policy_inputs = _to_device_inputs(blank_input_builder.get(args.policy_batch_size), device)

            critic_opt.zero_grad(set_to_none=True)

            # Do separate backward passes to avoid cross-forward autograd version conflicts
            # on integer mask/index tensors under DDP.
            expert_scores_raw = critic(expert_inputs, expert_robot_obs, expert_adj)
            expert_scores = torch.tanh(expert_scores_raw / args.disc_tanh_temp)
            expert_term = -expert_scores.mean() + args.raw_score_l2_coef * expert_scores_raw.pow(2).mean()
            accelerator.backward(expert_term)

            policy_scores_raw = critic(policy_inputs, policy_batch["robot_obs"], policy_batch["adj"])
            policy_scores = torch.tanh(policy_scores_raw / args.disc_tanh_temp)
            policy_term = policy_scores.mean() + args.raw_score_l2_coef * policy_scores_raw.pow(2).mean()
            accelerator.backward(policy_term)

            if args.critic_grad_clip > 0:
                accelerator.clip_grad_norm_(critic.parameters(), args.critic_grad_clip)
            critic_opt.step()
            critic_loss = expert_term + policy_term

            critic_losses.append(float(critic_loss.item()))
            expert_scores_log.append(float(expert_scores.mean().item()))
            policy_scores_log.append(float(policy_scores.mean().item()))
            expert_scores_raw_log.append(float(expert_scores_raw.mean().item()))
            policy_scores_raw_log.append(float(policy_scores_raw.mean().item()))

        for _ in range(args.actor_updates):
            policy_batch = rollout_buffer.sample_clips(args.policy_batch_size, args.clip_len, device=device)
            policy_inputs = _to_device_inputs(blank_input_builder.get(args.policy_batch_size), device)

            with torch.no_grad():
                score = critic(policy_inputs, policy_batch["robot_obs"], policy_batch["adj"])
                score = torch.tanh((score / args.disc_tanh_temp) * args.score_scale)

            obs_seq = policy_batch["robot_obs"]   # [B, T, N, D]
            act_seq = policy_batch["actions"]     # [B, T, N, A]
            bsz, tlen = obs_seq.shape[0], obs_seq.shape[1]
            obs_flat = obs_seq.reshape(bsz * tlen, obs_seq.shape[2], obs_seq.shape[3])
            act_flat = act_seq.reshape(bsz * tlen, act_seq.shape[2], act_seq.shape[3])
            logp_step, ent_step = actors.evaluate_actions(obs_flat, act_flat)  # [B*T, N]
            logp = logp_step.sum(dim=1).view(bsz, tlen).sum(dim=1)  # [B]
            entropy = ent_step.mean(dim=1).view(bsz, tlen).mean(dim=1)  # [B]

            actor_loss = -(logp * score.detach()).mean() - args.entropy_coef * entropy.mean()

            actor_opt.zero_grad(set_to_none=True)
            accelerator.backward(actor_loss)
            actor_opt.step()
            actor_losses.append(float(actor_loss.item()))

        if it % args.log_every == 0 and accelerator.is_main_process:
            rew_mean = float(torch.stack([s.reward_mean.mean() for s in rollout_buffer.steps]).mean().item())
            print(
                f"iter={it} "
                f"critic_loss={np.mean(critic_losses):.4f} "
                f"actor_loss={np.mean(actor_losses):.4f} "
                f"expert_score={np.mean(expert_scores_log):.4f} "
                f"policy_score={np.mean(policy_scores_log):.4f} "
                f"raw_expert_score={np.mean(expert_scores_raw_log):.4f} "
                f"raw_policy_score={np.mean(policy_scores_raw_log):.4f} "
                f"buffer_reward_mean={rew_mean:.4f}"
            )

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


if __name__ == "__main__":
    main()
