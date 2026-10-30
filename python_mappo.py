"""Causal Python-environment histories and a centralized MA-VLCM PPO critic.

Only the native MAPPO actor is needed at deployment. Images are encoded once
per 5 Hz camera frame; GAT/transformer/value parameters learn on every PPO run.
"""
from dataclasses import dataclass
from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn

from model import ModelConfig, MultimodalValueModel
from vlcm_mappo_timing import context_indices

MARL_ROOT = Path(__file__).resolve().parent.parent / "MARL_data_collection"
sys.path.insert(0, str(MARL_ROOT))
sys.path.insert(0, str(MARL_ROOT / "on-policy"))
from RL_bound_overwatch.envs import FormationControl
from onpolicy.config import get_config
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic


def actor_config(hidden_size=64, layer_n=1):
    args = get_config().parse_args([])
    args.algorithm_name = "mappo"
    args.use_recurrent_policy = args.use_naive_recurrent_policy = False
    args.use_popart = args.use_valuenorm = False
    args.hidden_size, args.layer_N = hidden_size, layer_n
    return args


def make_env(args, count):
    return FormationControl(
        num_agents=count, max_num_agents=6, pattern="polygon",
        formation_spacing=args.formation_spacing,
        formation_tolerance=args.formation_tolerance,
        initial_position_range=args.initial_position_range,
        min_start_separation=args.min_start_separation,
        observation_scale=args.observation_scale, dt=args.control_dt,
        max_steps=args.env_max_steps, formation_hold_steps=args.formation_hold_steps,
    )


def robot_state(env, previous_positions=None):
    """Match the ten MA-VLCM feature definitions, in world metres and m/s."""
    pos = env.pos
    rel = env._assigned_formation_config() + env._compute_swarm_centroid() - pos
    distance = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    velocity = np.zeros_like(pos) if previous_positions is None else (pos - previous_positions) / env.dt
    return np.column_stack((pos, np.sin(env.theta), np.cos(env.theta), velocity,
                            rel, np.linalg.norm(rel, axis=-1), distance.min(-1))).astype(np.float32)


@dataclass
class CameraFrame:
    image: object
    prompt: str
    features: object = None


class EpisodeHistory:
    def __init__(self, env, args, camera_queue):
        self.env, self.args = env, args
        self.states, self.extras, self.cameras, self.camera_steps = [], [], [], []
        self.append(camera_queue)

    def append(self, camera_queue, previous_positions=None):
        env, a = self.env, self.args
        self.states.append(robot_state(env, previous_positions))
        # These causal variables affect the Python reward/termination but were
        # absent in the Isaac offline ten-feature schema.
        self.extras.append(np.array([env.stable_team_steps / env.formation_hold_steps,
                                     env.last_settled_fraction], dtype=np.float32))
        if env.steps % a.camera_stride == 0:
            prompt = (f"Overhead view of {env.active_num_agents} black ground robots on a tiled floor. "
                      f"Form a regular {env.active_num_agents}-vertex polygon with neighbor spacing "
                      f"{env.d:g} metres. The camera covers {a.view_size_m:g} by {a.view_size_m:g} metres.")
            image = None
            if a.critic == "vlcm":
                image = Image.fromarray(env.render(size=a.image_size, view_size_m=a.view_size_m,
                                                  tile_floor=a.tile_floor))
            frame = CameraFrame(image, prompt)
            self.camera_steps.append(env.steps)
            self.cameras.append(frame)
            if a.critic == "vlcm":
                camera_queue.append(frame)

    def context(self):
        a = self.args
        index = context_indices(self.env.steps, 0, np.asarray(self.camera_steps),
                                clip_len=a.clip_len, state_stride=a.camera_stride,
                                control_dt=a.control_dt, max_image_age_steps=a.camera_stride - 1)
        return dict(robot_obs=np.stack([self.states[i] for i in index["state_indices"]]),
                    cameras=[self.cameras[i] for i in index["camera_indices"]],
                    frame_mask=index["frame_mask"], times=index["times"].astype(np.float32),
                    image_age=index["image_age"].astype(np.float32),
                    extras=np.stack([self.extras[i] for i in index["state_indices"]]),
                    shared_state=self.env.get_state()[0].copy())


def context_batch(contexts, device, feature_dim, graph_radius):
    """Pad active robot sets without allowing them to influence graph pooling."""
    b, t = len(contexts), len(contexts[0]["frame_mask"])
    obs = np.zeros((b, t, 6, 10), np.float32)
    mask = np.zeros((b, t, 6), bool)
    features = torch.zeros(b, t, feature_dim)
    for i, row in enumerate(contexts):
        n = row["robot_obs"].shape[1]
        obs[i, :, :n] = row["robot_obs"]
        mask[i, :, :n] = True
        for j, frame in enumerate(row["cameras"]):
            if frame.features is not None:
                features[i, j] = frame.features
            elif frame.image is not None:
                raise RuntimeError("Camera features must be encoded before PPO")
    distance = np.linalg.norm(obs[:, :, :, None, :2] - obs[:, :, None, :, :2], axis=-1)
    adj = (distance <= graph_radius) & mask[:, :, :, None] & mask[:, :, None, :]
    batch = dict(robot_obs=obs, robot_mask=mask, adj=adj, vl_features=features,
                 frame_mask=np.stack([r["frame_mask"] for r in contexts]),
                 times=np.stack([r["times"] for r in contexts]),
                 extras=np.stack([np.column_stack((r["image_age"], r["extras"])) for r in contexts]),
                 shared_state=np.stack([r["shared_state"] for r in contexts]))
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


class PythonValueCritic(nn.Module):
    def __init__(self, cfg, args, env, device, pretrained=None):
        super().__init__()
        self.kind, self.cfg = args.critic, cfg
        # A fixed invertible scale conditions the loss; rewards/GAE/value
        # predictions remain in original Python reward units.
        self.register_buffer("value_scale", torch.tensor(float(args.value_scale)))
        if self.kind == "mlp":
            self.core = R_Critic(actor_config(args.hidden_size, args.layer_n),
                                 env.shared_observation_spaces[0], device)
        else:
            self.core = MultimodalValueModel(cfg, device, load_backbone=False)
            python_checkpoint = pretrained is not None and pretrained.get("format") == "python_mappo_vlcm_v1"
            if pretrained is not None and not python_checkpoint:
                self.core.load_state_dict(pretrained["model_state"], strict=True)
            # Isaac's clipped reward targets have different units/termination.
            # Transfer its graph/fusion/temporal representation, reset regressor.
            self.core.target_mean.zero_()
            self.core.target_std.fill_(1)
            self.core.value_head[-1].reset_parameters()
            # Fixed physical-unit normalizers are shared by both graph variants.
            self.core.obs_mean.zero_()
            self.core.obs_std.copy_(torch.tensor([3., 3., 1., 1., .22, .22, 3., 3., 3., 3.]))
            if self.kind == "state-temporal":
                self.core.visual.requires_grad_(False)
            self.context_embedding = nn.Sequential(nn.Linear(3, cfg.d_model), nn.Tanh())
            nn.init.zeros_(self.context_embedding[0].weight)
            nn.init.zeros_(self.context_embedding[0].bias)
            if python_checkpoint:
                if pretrained["critic_kind"] != self.kind:
                    raise ValueError("Python critic checkpoint kind must match --critic")
                if float(pretrained["model_state"]["value_scale"]) != float(args.value_scale):
                    raise ValueError("--value-scale must match the saved Python critic")
                # Preserve the learned Python head, context and normalization.
                # This is critic warm-starting, independent of actor initialization.
                self.load_state_dict(pretrained["model_state"], strict=True)
        if getattr(args, "freeze_critic", False):
            self.requires_grad_(False)
        elif getattr(args, "freeze_critic_representation", False):
            if self.kind == "mlp":
                raise ValueError("Representation freezing requires a graph/temporal critic")
            self.requires_grad_(False)
            self.core.value_head.requires_grad_(True)
        self.to(device)

    def forward(self, batch):
        if self.kind == "mlp":
            n = len(batch["shared_state"])
            device = batch["shared_state"].device
            value, _ = self.core(batch["shared_state"], torch.zeros(n, 1, self.core.hidden_size, device=device),
                                 torch.ones(n, 1, device=device))
            return value.squeeze(-1) * self.value_scale
        core = self.core
        obs = (batch["robot_obs"] - core.obs_mean) / core.obs_std
        graph = core.graph(obs, batch["adj"], batch["robot_mask"])
        if self.kind == "state-temporal":
            visual = torch.zeros_like(graph)
        else:
            visual = core.visual(batch["vl_features"].float())
        fused = core.fuse(torch.cat((visual, graph), dim=-1))
        fused = fused + core.time_embedding(batch["times"].unsqueeze(-1))
        fused = fused + self.context_embedding(batch["extras"])
        temporal = core.temporal(fused, src_key_padding_mask=~batch["frame_mask"])
        return core.value_head(temporal[:, -1]).squeeze(-1) * self.value_scale


def load_python_critic(path, device=torch.device("cpu")):
    """Load a trained Python critic for raw-return inference on causal contexts."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "python_mappo_vlcm_v1":
        raise ValueError("Expected a trained Python MAPPO critic checkpoint")
    args = Namespace(**saved["args"])
    cfg = ModelConfig(**saved["model_config"])
    critic = PythonValueCritic(cfg, args, make_env(args, args.agent_counts[0]), device)
    critic.load_state_dict(saved["model_state"], strict=True)
    return critic.eval(), args


def compute_gae(rewards, values, next_values, terminated, done, gamma=.99, gae_lambda=.95):
    """Timeouts bootstrap final pre-reset observation and cut the GAE recursion."""
    advantage = np.zeros_like(rewards, dtype=np.float32)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (~terminated[t]) * next_values[t] - values[t]
        carry = delta + gamma * gae_lambda * (~done[t]) * carry
        advantage[t] = carry
    return advantage, advantage + values


class RolloutCollector:
    def __init__(self, args):
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.envs = [make_env(args, args.agent_counts[i % len(args.agent_counts)]) for i in range(args.num_envs)]
        self.histories, self.obs, self.camera_queue = [], [], []
        self.episode_returns = np.zeros(args.num_envs)
        self.episode_collisions = np.zeros(args.num_envs)
        for env in self.envs:
            obs, _, _ = env.reset(seed=int(self.rng.integers(0, 2**31)))
            self.obs.append(obs)
            self.histories.append(EpisodeHistory(env, args, self.camera_queue))

    def collect(self, actor, device, steps):
        a, e = self.args, len(self.envs)
        obs_rows, actions, logps, contexts, next_contexts = [], [], [], [], []
        rewards, done, terminated, active, completed = [], [], [], [], []
        outside = 0
        for _ in range(steps):
            obs = np.stack(self.obs)
            with torch.no_grad():
                action, logp, _ = actor(obs.reshape(-1, obs.shape[-1]),
                                       np.zeros((e * 6, 1, a.hidden_size), np.float32),
                                       np.ones((e * 6, 1), np.float32))
            action = action.cpu().numpy().reshape(e, 6, 2)
            if not np.isfinite(action).all():
                raise FloatingPointError("Non-finite actor actions")
            obs_rows.append(obs)
            actions.append(action)
            logps.append(logp.cpu().numpy().reshape(e, 6, 1))
            masks = np.array([np.arange(6) < env.active_num_agents for env in self.envs])
            active.append(masks)
            row_context, row_next, row_reward, row_done, row_terminal = [], [], [], [], []
            for i, env in enumerate(self.envs):
                row_context.append(self.histories[i].context())
                old_pos = env.pos.copy()
                next_obs, _, reward, dones, infos, _ = env.step(action[i])
                self.histories[i].append(self.camera_queue, old_pos)
                row_next.append(self.histories[i].context())
                # The team reward is repeated across ACTIVE agents only.
                team_reward = float(reward[:env.active_num_agents].mean())
                info = infos[0]
                row_reward.append(team_reward)
                row_done.append(bool(dones[0]))
                row_terminal.append(info["terminated"])
                outside += int(np.any(np.abs(env.pos) + env.rob_diam / 2 > a.view_size_m / 2, axis=-1).sum())
                self.episode_returns[i] += team_reward
                self.episode_collisions[i] += bool(info["collision_detected"])
                if dones[0]:
                    completed.append(dict(agents=env.active_num_agents, reward=float(self.episode_returns[i]),
                                          success=float(info["stable_formation_achieved"]),
                                          length=env.steps, collisions=float(self.episode_collisions[i]),
                                          formation_error=float(info["formation_error"])))
                    self.episode_returns[i] = self.episode_collisions[i] = 0
                    count = int(self.rng.choice(a.agent_counts))
                    env = self.envs[i] = make_env(a, count)
                    next_obs, _, _ = env.reset(seed=int(self.rng.integers(0, 2**31)))
                    self.histories[i] = EpisodeHistory(env, a, self.camera_queue)
                self.obs[i] = next_obs
            contexts.extend(row_context)
            next_contexts.extend(row_next)
            rewards.append(row_reward)
            done.append(row_done)
            terminated.append(row_terminal)
        return dict(obs=np.stack(obs_rows), actions=np.stack(actions), logps=np.stack(logps),
                    active=np.stack(active), contexts=contexts, next_contexts=next_contexts,
                    rewards=np.asarray(rewards, np.float32), done=np.asarray(done, bool),
                    terminated=np.asarray(terminated, bool), completed=completed,
                    outside_robot_frames=outside)

    def encode_cameras(self, backbone):
        queue, self.camera_queue = self.camera_queue, []
        # Bound feature memory on the GPU and release PIL images once encoded.
        chunk = backbone.cfg.vl_batch_size if backbone is not None else 1
        for start in range(0, len(queue), chunk):
            frames = queue[start:start + chunk]
            features = backbone([f.image for f in frames], [f.prompt for f in frames]).detach().cpu()
            for frame, feature in zip(frames, features):
                frame.features, frame.image = feature, None
        return len(queue)


def critic_values(critic, contexts, args, device):
    output = []
    with torch.no_grad():
        for start in range(0, len(contexts), args.critic_batch_size):
            batch = context_batch(contexts[start:start + args.critic_batch_size], device,
                                  critic.cfg.vl_hidden_dim, args.graph_radius)
            output.append(critic(batch).cpu().numpy())
    return np.concatenate(output)


def ppo_update(actor, critic, actor_optimizer, critic_optimizer, rollout, args, device, update):
    shape = rollout["rewards"].shape
    values = critic_values(critic, rollout["contexts"], args, device).reshape(shape)
    next_values = critic_values(critic, rollout["next_contexts"], args, device).reshape(shape)
    advantages, targets = compute_gae(rollout["rewards"], values, next_values,
                                      rollout["terminated"], rollout["done"], args.gamma, args.gae_lambda)
    mask = rollout["active"].reshape(-1)
    obs = rollout["obs"].reshape(-1, rollout["obs"].shape[-1])[mask]
    actions = rollout["actions"].reshape(-1, 2)[mask]
    logps = rollout["logps"].reshape(-1, 1)[mask]
    adv = np.repeat(advantages[:, :, None], 6, axis=2).reshape(-1)[mask]
    adv = (adv - adv.mean()) / max(float(adv.std()), 1e-5)
    actor_logs, critic_logs = [], []
    if critic_optimizer is None or update > args.critic_warmup_updates:
        for _ in range(args.ppo_epochs):
            for ids in np.array_split(np.random.permutation(len(obs)), min(args.num_minibatches, len(obs))):
                new_logp, entropy = actor.evaluate_actions(
                    obs[ids], np.zeros((len(ids), 1, args.hidden_size), np.float32), actions[ids],
                    np.ones((len(ids), 1), np.float32))
                old = torch.as_tensor(logps[ids], device=device)
                advantage = torch.as_tensor(adv[ids, None], device=device)
                ratio = (new_logp - old).exp()
                loss = -torch.minimum(ratio * advantage, ratio.clamp(1-args.clip, 1+args.clip) * advantage).mean()
                actor_optimizer.zero_grad(set_to_none=True)
                (loss - args.entropy_coef * entropy).backward()
                grad = nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm, error_if_nonfinite=True)
                actor_optimizer.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - (new_logp - old)).mean()
                actor_logs.append([loss.item(), entropy.item(), kl.item(),
                                   (torch.abs(ratio - 1) > args.clip).float().mean().item(), float(grad)])
    flat_targets, old_values = targets.reshape(-1), values.reshape(-1)
    for _ in range(args.critic_epochs if critic_optimizer is not None else 0):
        permutation = np.random.permutation(len(flat_targets))
        for start in range(0, len(permutation), args.critic_batch_size):
            ids = permutation[start:start + args.critic_batch_size]
            batch = context_batch([rollout["contexts"][i] for i in ids], device,
                                  critic.cfg.vl_hidden_dim, args.graph_radius)
            predicted = critic(batch) / args.value_scale
            target = torch.as_tensor(flat_targets[ids], device=device) / args.value_scale
            old = torch.as_tensor(old_values[ids], device=device) / args.value_scale
            clipped = old + (predicted - old).clamp(-args.clip, args.clip)
            loss = torch.maximum(nn.functional.huber_loss(predicted, target, reduction="none", delta=10),
                                 nn.functional.huber_loss(clipped, target, reduction="none", delta=10)).mean()
            critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            critic_optimizer.step()
            critic_logs.append([loss.item(), float(grad)])
    var = float(np.var(flat_targets))
    metrics = {"critic/explained_variance_before_update": 1-float(np.var(flat_targets-old_values))/var if var > 1e-8 else 0.,
               "critic/mse_raw_before_update": float(np.mean((flat_targets-old_values)**2)),
               "critic/value_mean": float(values.mean()), "critic/target_mean": float(targets.mean()),
               "rollout/reward_per_step": float(rollout["rewards"].mean()),
               "rollout/terminated": int(rollout["terminated"].sum()),
               "rollout/truncated": int((rollout["done"] & ~rollout["terminated"]).sum()),
               "rollout/outside_robot_frames": rollout["outside_robot_frames"],
               "rollout/outside_robot_frame_fraction": rollout["outside_robot_frames"] / int(rollout["active"].sum()),
               "vision/image_age_mean_seconds": float(np.mean([r["image_age"][-1] for r in rollout["contexts"]])),
               "vision/image_age_max_seconds": float(max(r["image_age"][-1] for r in rollout["contexts"])),
               "train/actor_updated": int(bool(actor_logs)),
               "train/critic_updated": int(bool(critic_logs))}
    if critic_logs:
        metrics.update({"critic/loss": float(np.mean(critic_logs, axis=0)[0]),
                        "critic/grad_norm": float(np.mean(critic_logs, axis=0)[1])})
    if actor_logs:
        metrics.update(dict(zip(["actor/loss", "actor/base_gaussian_entropy", "actor/approx_kl",
                                 "actor/clip_fraction", "actor/grad_norm"], np.mean(actor_logs, axis=0).tolist())))
    for n in args.agent_counts:
        episodes = [row for row in rollout["completed"] if row["agents"] == n]
        if episodes:
            metrics[f"episodes/{n}_agents/count"] = len(episodes)
            for key in ("reward", "success", "length", "collisions", "formation_error"):
                metrics[f"episodes/{n}_agents/{key}"] = float(np.mean([row[key] for row in episodes]))
    return metrics


def evaluate_actor(actor, args, device):
    """Fixed held-out reset seeds; no VLM or centralized critic needed."""
    metrics = {}
    for count in args.agent_counts:
        scores = []
        for episode in range(args.eval_episodes):
            env = make_env(args, count)
            obs, _, _ = env.reset(seed=args.eval_seed + 1000 * count + episode)
            total, collisions = 0., 0
            for _ in range(args.env_max_steps):
                with torch.no_grad():
                    action, _, _ = actor(obs, np.zeros((6, 1, args.hidden_size), np.float32),
                                         np.ones((6, 1), np.float32), deterministic=True)
                obs, _, reward, done, infos, _ = env.step(action.cpu().numpy())
                total += float(reward[:count].mean())
                collisions += bool(infos[0]["collision_detected"])
                if done[0]:
                    break
            scores.append([float(infos[0]["stable_formation_achieved"]), total,
                           env.steps, collisions, float(infos[0]["formation_error"])])
        for key, value in zip(("success", "reward", "length", "collision_steps", "formation_error"),
                              np.mean(scores, axis=0)):
            metrics[f"eval/{count}_agents/{key}"] = float(value)
    metrics["eval/success"] = float(np.mean([metrics[f"eval/{n}_agents/success"] for n in args.agent_counts]))
    return metrics
