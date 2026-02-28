import argparse
import os
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

try:
    import ManyAgent_GoTOGoal  # noqa: F401
except Exception:
    ManyAgent_GoTOGoal = None

from train_irl_local_policy import (
    LocalAgentPolicies,
    _normalize_scenario,
    _space_info_with_overrides,
    _unpack_reset_out,
    _unpack_step_out,
)


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


def _flatten_info_dicts(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, np.ndarray):
        out = []
        for x in obj.flat:
            out.extend(_flatten_info_dicts(x))
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for x in obj:
            out.extend(_flatten_info_dicts(x))
        return out
    return []


def _extract_team_flags(info: Any, num_agents: int) -> Tuple[int, int]:
    agent_infos = _flatten_info_dicts(info)
    reached = 0
    collision = 0
    max_agents = min(num_agents, len(agent_infos))
    for i in range(max_agents):
        x = agent_infos[i]
        if not isinstance(x, dict):
            continue
        if bool(x.get("reached_goal", False)):
            reached += 1
        if bool(x.get("collision", False)):
            collision += 1
    return reached, collision


def _to_env_actions(actions_t: torch.Tensor) -> np.ndarray:
    arr = actions_t.detach().cpu().numpy()
    if arr.shape[-1] == 1:
        return arr.squeeze(-1).astype(np.int64)
    return arr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--scenario", type=str, default="ManyAgentGoToGoal-v0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--device", type=str, default="cuda")

    # Match actor architecture args
    p.add_argument("--policy_hidden_dim", type=int, default=256)
    p.add_argument("--action_type", type=str, default="auto", choices=["auto", "continuous", "discrete"])
    p.add_argument("--action_dim", type=int, default=2)

    # Rendering / video
    p.add_argument("--render", action="store_true")
    p.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array"])
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--video_path", type=str, default="eval_policy.mp4")
    p.add_argument("--video_fps", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    scenario = _normalize_scenario(args.scenario)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if ManyAgent_GoTOGoal is None:
        print("[WARN] Could not import ManyAgent_GoTOGoal in this env.")

    render_mode = "rgb_array" if args.save_video else (args.render_mode if args.render else None)
    try:
        env = gym.make(scenario, disable_env_checker=True, render_mode=render_mode)
    except TypeError:
        # ManyAgentGoToGoal builds may not support gymnasium render_mode constructor kwarg.
        env = gym.make(scenario, disable_env_checker=True)
    if hasattr(env, "seed"):
        env.seed(args.seed)

    reset_out = env.reset()
    obs, _ = _unpack_reset_out(reset_out, 0)
    obs = np.asarray(obs, dtype=np.float32)
    num_agents = int(obs.shape[0])
    obs_dim = int(obs.shape[-1])

    class _Tmp:
        pass

    tmp = _Tmp()
    tmp.action_type = args.action_type
    tmp.action_dim = args.action_dim
    tmp.vl_backend = "llava_video"
    is_continuous, action_dim, action_low, action_high = _space_info_with_overrides(env, tmp, num_agents)

    actors = LocalAgentPolicies(
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        continuous=is_continuous,
        action_low=action_low,
        action_high=action_high,
        hidden_dim=args.policy_hidden_dim,
    ).to(device)
    actors.eval()

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "actors" in ckpt:
        actors.load_state_dict(ckpt["actors"], strict=False)
    else:
        actors.load_state_dict(ckpt, strict=False)

    all_ep_rewards = []
    all_reached = []
    all_collision = []
    frames = []

    for ep in range(args.eval_episodes):
        obs, _ = _unpack_reset_out(env.reset(), num_agents)
        obs = np.asarray(obs, dtype=np.float32)

        ep_reward = 0.0
        reached_count = 0
        collision_count = 0

        for _ in range(args.max_episode_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # [1,N,D]
            with torch.no_grad():
                action_t, _, _ = actors.act(obs_t, deterministic=True)

            action_np = _to_env_actions(action_t[0])
            out = _unpack_step_out(env.step(action_np))
            next_obs, _, reward, done, info = out

            rew_arr = np.asarray(reward).reshape(-1)
            ep_reward += float(rew_arr.mean() if rew_arr.size > 0 else float(reward))

            r, c = _extract_team_flags(info, num_agents)
            reached_count = max(reached_count, r)
            collision_count = max(collision_count, c)

            if args.render and not args.save_video:
                try:
                    env.render(mode=args.render_mode)
                except TypeError:
                    try:
                        env.render()
                    except Exception:
                        pass
                except Exception:
                    pass

            if args.save_video:
                try:
                    frame = env.render(mode="rgb_array")
                except TypeError:
                    frame = env.render()
                frame = _extract_rgb_frame(frame)
                if frame is not None:
                    frames.append(frame)

            done_arr = np.asarray(done).reshape(-1)
            done_env = bool(done_arr.mean() > 0.5)
            obs = np.asarray(next_obs, dtype=np.float32)
            if done_env:
                break

        all_ep_rewards.append(ep_reward)
        all_reached.append(reached_count)
        all_collision.append(collision_count)
        print(
            f"episode={ep + 1}/{args.eval_episodes} "
            f"reward={ep_reward:.3f} reached={reached_count}/{num_agents} collision={collision_count}/{num_agents}"
        )

    if args.save_video and imageio is not None and len(frames) > 0:
        out_dir = os.path.dirname(args.video_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        imageio.mimsave(args.video_path, frames, fps=args.video_fps)
        print(f"saved_video={os.path.abspath(args.video_path)} frames={len(frames)}")
    elif args.save_video and imageio is not None and len(frames) == 0:
        print("[WARN] save_video was enabled but 0 frames were captured from env.render().")
    elif args.save_video and imageio is None:
        print("[WARN] imageio not available, video not saved.")

    print("\n=== Eval Summary ===")
    print(f"episodes={args.eval_episodes}")
    print(f"avg_reward={float(np.mean(all_ep_rewards)):.4f}")
    print(f"avg_reached_goal_count={float(np.mean(all_reached)):.4f}/{num_agents}")
    print(f"avg_collision_count={float(np.mean(all_collision)):.4f}/{num_agents}")
    print(f"avg_reached_goal_rate={float(np.mean(all_reached) / max(num_agents, 1)):.4f}")
    print(f"avg_collision_rate={float(np.mean(all_collision) / max(num_agents, 1)):.4f}")

    env.close()


if __name__ == "__main__":
    main()
