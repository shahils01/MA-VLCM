import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from eval_critic import _load_checkpoint_state, _load_train_args
from train import _resolve_contrastive_depth_args, _resolve_vl_model_preset, build_model
from train_irl_local_policy import (
    CachedBlankVideoInputs,
    LocalAgentPolicies,
    _normalize_scenario,
    _resize_frame_uint8,
    _space_info_with_overrides,
    _unpack_reset_out,
    _unpack_step_out,
    build_video_inputs_from_batch,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train decentralized local actors with a frozen centralized MA-VLCM critic.")
    p.add_argument("--critic_checkpoint", type=str, required=True)
    p.add_argument("--env_repo", type=str, required=True, help="Path to Bayesian-Trust-Modeling repo.")
    p.add_argument("--scenario", type=str, default="ManyAgentGoToGoal-v0")
    p.add_argument("--env_kwargs", type=str, default="{}", help="JSON dict passed to gym.make.")
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--rollout_steps", type=int, default=128)
    p.add_argument("--clip_len", type=int, default=12)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--mini_batch_size", type=int, default=128)
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--clip_coef", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=1e-3)
    p.add_argument("--normalize_advantages", action="store_true")
    p.add_argument("--policy_hidden_dim", type=int, default=256)
    p.add_argument("--policy_video_source", type=str, default="env", choices=["env", "blank"])
    p.add_argument("--frame_store_size", type=int, default=224)
    p.add_argument("--critic_batch_size", type=int, default=8)
    p.add_argument("--text_prompt_template", type=str, default="")
    p.add_argument("--action_type", type=str, default="auto", choices=["auto", "continuous", "discrete"])
    p.add_argument("--action_dim", type=int, default=2)
    p.add_argument("--clip_actions_to_space", action="store_true")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=25)
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument("--eval_max_episode_steps", type=int, default=400)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--save_dir", type=str, default="checkpoints_marl_frozen_critic")
    return p.parse_args()


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _bootstrap_env_repo(env_repo: str):
    repo = Path(env_repo).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"Environment repo not found: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    init_py = repo / "__init__.py"
    if not init_py.exists():
        raise FileNotFoundError(f"Environment bootstrap file not found: {init_py}")

    spec = importlib.util.spec_from_file_location("btm_env_bootstrap", init_py)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)


def _parse_env_kwargs(raw: str):
    try:
        out = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"--env_kwargs must be valid JSON: {e}") from e
    if not isinstance(out, dict):
        raise ValueError("--env_kwargs must decode to a JSON object.")
    return out


class ManyAgentVecEnv:
    def __init__(self, scenario: str, num_envs: int, seed: int, env_kwargs: dict, render_mode: str | None):
        self.scenario = _normalize_scenario(scenario)
        self.num_envs = int(num_envs)
        self.envs = []
        for idx in range(self.num_envs):
            kwargs = dict(env_kwargs)
            if render_mode is not None:
                kwargs["render_mode"] = render_mode
            try:
                env = gym.make(self.scenario, disable_env_checker=True, **kwargs)
            except TypeError:
                kwargs.pop("render_mode", None)
                env = gym.make(self.scenario, disable_env_checker=True, **kwargs)
            if hasattr(env, "seed"):
                env.seed(seed + idx * 1000)
            self.envs.append(env)

        first = self.envs[0]
        self.action_space = getattr(first, "action_space", None)
        self.observation_space = getattr(first, "observation_space", None)
        self.n_agents = int(getattr(first, "n_agents", 0) or 0)
        if self.n_agents <= 0:
            obs0, _ = _unpack_reset_out(first.reset(), 0)
            self.n_agents = int(np.asarray(obs0).shape[0])

    def _get_adj_single(self, env):
        if hasattr(env, "get_visibility_matrix"):
            return np.asarray(env.get_visibility_matrix(), dtype=np.float32)
        if hasattr(env, "get_edge_index_matrix"):
            edge_index = np.asarray(env.get_edge_index_matrix())
            if edge_index.ndim == 2 and edge_index.shape == (self.n_agents, self.n_agents):
                return edge_index.astype(np.float32)
            if edge_index.ndim == 2 and edge_index.shape[0] == 2:
                adj = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
                src, dst = edge_index[0], edge_index[1]
                valid = (src >= 0) & (dst >= 0) & (src < self.n_agents) & (dst < self.n_agents)
                adj[src[valid].astype(np.int64), dst[valid].astype(np.int64)] = 1.0
                return adj
        return np.eye(self.n_agents, dtype=np.float32)

    def get_adjacency(self):
        return np.stack([self._get_adj_single(env) for env in self.envs], axis=0)

    def reset(self):
        obs_list, share_list = [], []
        for env in self.envs:
            obs, share_obs = _unpack_reset_out(env.reset(), self.n_agents)
            obs_list.append(np.asarray(obs, dtype=np.float32))
            share_list.append(np.asarray(share_obs, dtype=np.float32))
        return np.stack(obs_list, axis=0), np.stack(share_list, axis=0)

    def step(self, actions: np.ndarray):
        obs_list, share_list, reward_list, done_list, info_list = [], [], [], [], []
        for idx, env in enumerate(self.envs):
            obs, share_obs, rew, done, info = _unpack_step_out(env.step(actions[idx]))
            rew_arr = np.asarray(rew, dtype=np.float32).reshape(-1)
            done_arr = np.asarray(done, dtype=np.float32).reshape(-1)
            if rew_arr.size == 1:
                rew_arr = np.repeat(rew_arr, self.n_agents)
            if done_arr.size == 1:
                done_arr = np.repeat(done_arr, self.n_agents)
            if np.all(done_arr > 0.5):
                obs, share_obs = _unpack_reset_out(env.reset(), self.n_agents)
            obs_list.append(np.asarray(obs, dtype=np.float32))
            share_list.append(np.asarray(share_obs, dtype=np.float32))
            reward_list.append(rew_arr.astype(np.float32))
            done_list.append(done_arr.astype(np.float32))
            info_list.append(info)
        return (
            np.stack(obs_list, axis=0),
            np.stack(share_list, axis=0),
            np.stack(reward_list, axis=0),
            np.stack(done_list, axis=0),
            info_list,
        )

    def render_rgb_array(self):
        frames = []
        for env in self.envs:
            try:
                frame = env.render(mode="rgb_array")
            except TypeError:
                try:
                    frame = env.render()
                except Exception:
                    frame = None
            except Exception:
                frame = None

            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                frames.append(frame.astype(np.uint8, copy=False))
            elif isinstance(frame, np.ndarray) and frame.ndim == 4 and frame.shape[0] > 0:
                frames.append(frame[0].astype(np.uint8, copy=False))
            else:
                frames.append(np.zeros((84, 84, 3), dtype=np.uint8))
        return np.stack(frames, axis=0)

    def close(self):
        for env in self.envs:
            env.close()


def _load_frozen_critic(args, device):
    ckpt = torch.load(args.critic_checkpoint, map_location="cpu")
    train_args = _load_train_args(ckpt)
    _resolve_vl_model_preset(train_args)
    _resolve_contrastive_depth_args(train_args)
    train_args.quantization_config = None
    if args.text_prompt_template:
        train_args.text_prompt_template = args.text_prompt_template
    critic = build_model(train_args, device=device)
    state = ckpt.get("model", ckpt.get("critic"))
    if state is None:
        raise KeyError("Checkpoint must contain either 'model' or 'critic' weights.")
    _load_checkpoint_state(critic, state, getattr(train_args, "peft", "none"))
    critic.to(device)
    critic.eval()
    for param in critic.parameters():
        param.requires_grad = False
    return critic, train_args


def _normalize_critic_robot_inputs(obs_seq: torch.Tensor, adj_seq: torch.Tensor, target_agents: int):
    cur_agents = int(obs_seq.shape[2])
    if cur_agents == target_agents:
        return obs_seq, adj_seq

    obs_out = torch.zeros(
        obs_seq.shape[0],
        obs_seq.shape[1],
        target_agents,
        obs_seq.shape[3],
        dtype=obs_seq.dtype,
    )
    adj_out = torch.zeros(
        adj_seq.shape[0],
        adj_seq.shape[1],
        target_agents,
        target_agents,
        dtype=adj_seq.dtype,
    )
    copy_n = min(cur_agents, target_agents)
    obs_out[:, :, :copy_n, :] = obs_seq[:, :, :copy_n, :]
    adj_out[:, :, :copy_n, :copy_n] = adj_seq[:, :, :copy_n, :copy_n]
    return obs_out, adj_out


def _build_state_clips(states, clip_len: int):
    num_states = len(states)
    num_envs = states[0]["obs"].shape[0]
    robot_obs, adj, videos = [], [], []
    for env_idx in range(num_envs):
        for state_idx in range(num_states):
            clip_obs, clip_adj, clip_vid = [], [], []
            start = max(0, state_idx - clip_len + 1)
            window = states[start : state_idx + 1]
            while len(window) < clip_len:
                window = [states[0]] + window
            for item in window:
                clip_obs.append(item["obs"][env_idx])
                clip_adj.append(item["adj"][env_idx])
                clip_vid.append(item["frame"][env_idx])
            robot_obs.append(torch.from_numpy(np.stack(clip_obs, axis=0)).float())
            adj.append(torch.from_numpy(np.stack(clip_adj, axis=0)).float())
            videos.append(torch.from_numpy(np.stack(clip_vid, axis=0)).to(torch.uint8))
    return {
        "robot_obs": torch.stack(robot_obs, dim=0),
        "adj": torch.stack(adj, dim=0),
        "video": torch.stack(videos, dim=0),
    }


@torch.no_grad()
def _evaluate_critic_values(critic, critic_args, clip_batch, device, video_mode, blank_builder, batch_size: int):
    values = []
    total = int(clip_batch["robot_obs"].shape[0])
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        robot_obs = clip_batch["robot_obs"][start:end]
        adj = clip_batch["adj"][start:end]
        if int(robot_obs.shape[2]) != int(critic_args.num_robots):
            robot_obs, adj = _normalize_critic_robot_inputs(robot_obs, adj, int(critic_args.num_robots))

        if video_mode == "blank":
            video_inputs = blank_builder.get(end - start)
        else:
            video_inputs = build_video_inputs_from_batch(
                processor=critic.backbone.processor,
                prompt=critic_args.text_prompt_template,
                videos_uint8=clip_batch["video"][start:end],
                frame_size=int(getattr(critic_args, "video_height", 224)),
                vlm_max_text_len=int(getattr(critic_args, "vl_max_text_len", 256)),
                vlm_truncation=False,
                vlm_padding="longest",
                obs_token_repeats=int(getattr(critic_args, "obs_summary_tokens", 1)),
            )

        video_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in video_inputs.items()}
        value = critic(video_inputs, robot_obs.to(device), adj.to(device))
        values.append(value.detach().float().cpu())
    return torch.cat(values, dim=0)


def _collect_rollout(envs, actors, args, device):
    obs_np, _ = envs.reset()
    states = []
    transitions = []

    for step_idx in range(args.rollout_steps + 1):
        frame_np = envs.render_rgb_array()
        if frame_np.shape[1] != args.frame_store_size or frame_np.shape[2] != args.frame_store_size:
            frame_np = np.stack(
                [_resize_frame_uint8(frame_np[i], args.frame_store_size) for i in range(frame_np.shape[0])],
                axis=0,
            )
        states.append(
            {
                "obs": np.asarray(obs_np, dtype=np.float32),
                "adj": envs.get_adjacency().astype(np.float32),
                "frame": frame_np.astype(np.uint8),
            }
        )
        if step_idx == args.rollout_steps:
            break

        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
        action_t, log_prob_t, entropy_t = actors.act(obs_t, deterministic=False)
        action_np = action_t.detach().cpu().numpy()
        if action_np.shape[-1] == 1:
            env_action = action_np.squeeze(-1).astype(np.int64)
        else:
            env_action = action_np

        next_obs, _, rewards, dones, infos = envs.step(env_action)
        reward_mean = rewards.mean(axis=1).astype(np.float32)
        done_env = (dones.mean(axis=1) > 0.5).astype(np.float32)
        transitions.append(
            {
                "obs": obs_np.copy(),
                "actions": action_t.detach().cpu(),
                "log_prob_sum": log_prob_t.sum(dim=1).detach().cpu(),
                "entropy_mean": entropy_t.mean(dim=1).detach().cpu(),
                "reward": torch.from_numpy(reward_mean),
                "done": torch.from_numpy(done_env),
                "infos": infos,
            }
        )
        obs_np = next_obs
    return states, transitions


def _compute_gae(rewards, dones, values, gamma: float, gae_lambda: float):
    num_steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(num_envs, dtype=torch.float32)
    for t in reversed(range(num_steps)):
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * next_nonterminal - values[t]
        last_adv = delta + gamma * gae_lambda * next_nonterminal * last_adv
        advantages[t] = last_adv
    returns = advantages + values[:-1]
    return advantages, returns


@torch.no_grad()
def evaluate_policy(envs, actors, device, episodes: int, max_steps: int):
    was_training = actors.training
    actors.eval()
    returns = []
    for _ in range(episodes):
        obs_all, _ = envs.reset()
        obs = np.asarray(obs_all[0], dtype=np.float32)
        ep_return = 0.0
        for _ in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action_t, _, _ = actors.act(obs_t, deterministic=True)
            action_np = action_t.squeeze(0).detach().cpu().numpy()
            if action_np.shape[-1] == 1:
                env_action = action_np.squeeze(-1).astype(np.int64)
            else:
                env_action = action_np
            next_obs, _, rewards, dones, _ = envs.step(np.expand_dims(env_action, axis=0))
            ep_return += float(np.asarray(rewards[0], dtype=np.float32).mean())
            obs = np.asarray(next_obs[0], dtype=np.float32)
            if bool(np.asarray(dones[0], dtype=np.float32).mean() > 0.5):
                break
        returns.append(ep_return)
    if was_training:
        actors.train()
    return float(np.mean(returns)) if returns else 0.0


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    _seed_everything(args.seed)
    _bootstrap_env_repo(args.env_repo)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    env_kwargs = _parse_env_kwargs(args.env_kwargs)
    critic, critic_args = _load_frozen_critic(args, device)
    if not args.text_prompt_template:
        args.text_prompt_template = getattr(critic_args, "text_prompt_template", "")

    render_mode = "rgb_array" if args.policy_video_source == "env" else None
    envs = ManyAgentVecEnv(args.scenario, args.num_envs, args.seed, env_kwargs, render_mode=render_mode)
    eval_envs = ManyAgentVecEnv(args.scenario, 1, args.seed + 999, env_kwargs, render_mode=None)

    reset_obs, _ = envs.reset()
    num_agents = int(reset_obs.shape[1])
    obs_dim = int(reset_obs.shape[2])
    is_continuous, action_dim, action_low, action_high = _space_info_with_overrides(envs, args, num_agents)
    if is_continuous and not args.clip_actions_to_space:
        action_low, action_high = None, None

    actors = LocalAgentPolicies(
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        continuous=is_continuous,
        action_low=action_low,
        action_high=action_high,
        hidden_dim=args.policy_hidden_dim,
    ).to(device)
    actor_opt = torch.optim.AdamW(actors.parameters(), lr=args.actor_lr, weight_decay=args.weight_decay)

    blank_builder = None
    if args.policy_video_source == "blank":
        blank_builder = CachedBlankVideoInputs(
            processor=critic.backbone.processor,
            prompt=args.text_prompt_template,
            clip_len=args.clip_len,
            frame_size=int(getattr(critic_args, "video_height", args.frame_store_size)),
            vlm_max_text_len=int(getattr(critic_args, "vl_max_text_len", 256)),
            vlm_truncation=False,
            vlm_padding="longest",
            obs_token_repeats=int(getattr(critic_args, "obs_summary_tokens", 1)),
        )

    for it in range(1, args.iters + 1):
        states, transitions = _collect_rollout(envs, actors, args, device)
        state_clips = _build_state_clips(states, args.clip_len)
        values_flat = _evaluate_critic_values(
            critic,
            critic_args,
            state_clips,
            device,
            args.policy_video_source,
            blank_builder,
            args.critic_batch_size,
        )
        values = values_flat.view(args.num_envs, args.rollout_steps + 1).transpose(0, 1).contiguous()

        rewards = torch.stack([step["reward"] for step in transitions], dim=0)
        dones = torch.stack([step["done"] for step in transitions], dim=0)
        old_log_probs = torch.stack([step["log_prob_sum"] for step in transitions], dim=0)
        actions = torch.stack([step["actions"] for step in transitions], dim=0)
        obs = torch.stack([torch.from_numpy(step["obs"]).float() for step in transitions], dim=0)

        advantages, returns = _compute_gae(rewards, dones, values, args.gamma, args.gae_lambda)
        if args.normalize_advantages:
            advantages = (advantages - advantages.mean()) / advantages.std().clamp(min=1e-6)

        batch_obs = obs.reshape(-1, num_agents, obs_dim).to(device)
        batch_actions = actions.reshape(actions.shape[0] * actions.shape[1], num_agents, -1).to(device)
        batch_old_logp = old_log_probs.reshape(-1).to(device)
        batch_adv = advantages.reshape(-1).to(device)

        batch_size = batch_obs.shape[0]
        idxs = np.arange(batch_size)
        actor_losses = []
        approx_kls = []

        for _ in range(args.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, args.mini_batch_size):
                mb_idx = idxs[start : start + args.mini_batch_size]
                mb_obs = batch_obs[mb_idx]
                mb_actions = batch_actions[mb_idx]
                mb_old_logp = batch_old_logp[mb_idx]
                mb_adv = batch_adv[mb_idx]

                new_logp, entropy = actors.evaluate_actions(mb_obs, mb_actions)
                new_logp_sum = new_logp.sum(dim=1)
                entropy_mean = entropy.mean(dim=1)
                ratio = torch.exp(new_logp_sum - mb_old_logp)
                pg_loss_1 = -mb_adv * ratio
                pg_loss_2 = -mb_adv * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()
                loss = policy_loss - args.entropy_coef * entropy_mean.mean()

                actor_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actors.parameters(), args.max_grad_norm)
                actor_opt.step()

                actor_losses.append(float(loss.item()))
                approx_kls.append(float((mb_old_logp - new_logp_sum).mean().item()))

        mean_reward = float(rewards.mean().item())
        mean_value = float(values[:-1].mean().item())
        mean_adv = float(advantages.mean().item())
        if it % args.log_every == 0 or it == 1:
            print(
                f"iter={it} reward_mean={mean_reward:.4f} value_mean={mean_value:.4f} "
                f"adv_mean={mean_adv:.4f} loss={np.mean(actor_losses):.4f} "
                f"approx_kl={np.mean(approx_kls):.6f}"
            )

        if it % args.eval_interval == 0:
            eval_return = evaluate_policy(eval_envs, actors, device, args.eval_episodes, args.eval_max_episode_steps)
            print(f"eval iter={it} avg_return={eval_return:.4f}")

        if it % args.save_every == 0:
            torch.save(
                {
                    "actors": actors.state_dict(),
                    "actor_opt": actor_opt.state_dict(),
                    "iter": it,
                    "args": vars(args),
                    "critic_checkpoint": args.critic_checkpoint,
                },
                os.path.join(args.save_dir, f"marl_frozen_critic_iter_{it}.pt"),
            )

    envs.close()
    eval_envs.close()


if __name__ == "__main__":
    main()
