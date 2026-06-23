#!/usr/bin/env python3
"""Live MA-VLCM inference for the TurtleBot3 lab runtime."""

import argparse
import ast
import csv
import json
import math
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

import torch

from .model import ModelConfig, MultimodalValueModel

try:
    import rclpy
    from geometry_msgs.msg import PointStamped, PoseStamped, Twist
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String

    _ROS_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    rclpy = None
    ExternalShutdownException = None
    Node = object
    PointStamped = None
    PoseStamped = None
    Twist = None
    CompressedImage = None
    String = None
    _ROS_IMPORT_ERROR = exc


ROBOT_COLORS = ("red", "blue", "green")


def parse_args():
    p = argparse.ArgumentParser(description="Run live MA-VLCM inference on TurtleBot3 lab topics.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--window_size", type=int, default=16)
    p.add_argument("--inference_rate_hz", type=float, default=1.0)
    p.add_argument("--robot_names", type=str, default="tb_1,tb_2,tb_3")
    p.add_argument("--adjacency_threshold", type=float, default=4.0)
    p.add_argument("--image_topic", type=str, default="/fleet_vlcm/overhead/compressed")
    p.add_argument("--prediction_topic", type=str, default="/fleet_vlcm/vlcm_prediction")
    p.add_argument("--output_csv", type=str, default="outputs/results/tb3_live_predictions.csv")
    p.add_argument("--goal_radius_m", type=float, default=0.12)
    p.add_argument("--proximity_penalty_distance_m", type=float, default=0.20)
    p.add_argument("--progress_scale", type=float, default=1.0)
    p.add_argument("--success_reward", type=float, default=5.0)
    p.add_argument("--proximity_penalty", type=float, default=-1.0)
    p.add_argument("--goal_change_epsilon_m", type=float, default=0.02)
    p.add_argument("--disable_lora", action="store_true")
    return p.parse_args()


def split_csv(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def yaw_from_quat_xyzw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def distance(a, b):
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _tuple_arg(value, default):
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return tuple(default)
        return tuple(parsed)
    return tuple(value)


def _load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    args = ckpt.get("args", {})
    if isinstance(args, dict):
        args = SimpleNamespace(**args)
    args.dataset_type = "tb3_lab"
    args.num_robots = 3
    args.robot_obs_dim = 8
    args.preprocess_in_loader = True
    args.video_preprocessed = True
    args.compile = getattr(args, "compile", False)
    if getattr(args, "peft", None) == "qlora":
        from transformers import BitsAndBytesConfig

        if getattr(args, "vl_dtype", "bfloat16") == "float16":
            compute_dtype = torch.float16
        elif getattr(args, "vl_dtype", "bfloat16") == "float32":
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
    return ckpt, args


def _arg(args, name, default):
    return getattr(args, name, default)


def build_model(args, device):
    cfg = ModelConfig(
        vl_backend=_arg(args, "vl_backend", "llava_onevision"),
        vl_model_name=_arg(args, "vl_model_name", "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"),
        vl_dtype=_arg(args, "vl_dtype", "bfloat16"),
        vl_max_text_len=_arg(args, "vl_max_text_len", 256),
        freeze_vl=_arg(args, "freeze_vl", False),
        freeze_vision_tower=_arg(args, "freeze_vision_tower", True),
        quantization_config=_arg(args, "quantization_config", None),
        video_channels=_arg(args, "video_channels", 3),
        video_frames=_arg(args, "video_frames", 16),
        video_preprocessed=_arg(args, "video_preprocessed", True),
        video_mean=_tuple_arg(_arg(args, "video_mean", None), (0.5, 0.5, 0.5)),
        video_std=_tuple_arg(_arg(args, "video_std", None), (0.5, 0.5, 0.5)),
        num_robots=_arg(args, "num_robots", 3),
        robot_obs_dim=_arg(args, "robot_obs_dim", 8),
        text_dim=_arg(args, "text_dim", 512),
        d_model=_arg(args, "d_model", 256),
        temporal_layers=_arg(args, "temporal_layers", 2),
        temporal_heads=_arg(args, "temporal_heads", 4),
        temporal_dropout=_arg(args, "temporal_dropout", 0.1),
        gnn_layers=_arg(args, "gnn_layers", 4),
        fusion_hidden=_arg(args, "fusion_hidden", 512),
        use_moe=_arg(args, "use_moe", False),
        moe_experts=_arg(args, "moe_experts", 4),
        moe_top_k=_arg(args, "moe_top_k", 2),
        debug_save_video=_arg(args, "debug_save_video", False),
        contrastive_multidepth=_arg(args, "contrastive_multidepth", False),
        contrastive_depth_offsets=tuple(_arg(args, "contrastive_depth_offsets_list", [0])),
    )
    return MultimodalValueModel(cfg, device=device)


def _parse_lora_targets(args):
    targets = _arg(args, "lora_target_modules", None)
    if targets:
        return [t.strip() for t in targets.split(",") if t.strip()]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _apply_peft(model, args):
    if _arg(args, "peft", "none") == "none":
        return model
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:
        raise RuntimeError(
            "PEFT/LoRA checkpoint loading requires the 'peft' package in the ROS Python."
        ) from exc

    for param in model.backbone.model.parameters():
        param.requires_grad = False

    if hasattr(model.backbone.model, "gradient_checkpointing_enable"):
        model.backbone.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if _arg(args, "peft", "none") == "qlora":
        model.backbone.model = prepare_model_for_kbit_training(
            model.backbone.model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    lora_cfg = LoraConfig(
        r=_arg(args, "lora_r", 16),
        lora_alpha=_arg(args, "lora_alpha", 32),
        lora_dropout=_arg(args, "lora_dropout", 0.05),
        bias=_arg(args, "lora_bias", "none"),
        target_modules=_parse_lora_targets(args),
        task_type="CAUSAL_LM",
    )
    model.backbone.model = get_peft_model(model.backbone.model, lora_cfg)
    return model


def _build_loaded_model(ckpt, args, device, disable_lora=False):
    model = build_model(args, device=device)
    model = _apply_peft(model, args)

    cleaned_sd = {}
    for k, v in ckpt["model"].items():
        cleaned_sd[k.replace("module.", "") if k.startswith("module.") else k] = v
    missing, unexpected = model.load_state_dict(cleaned_sd, strict=False)
    if missing:
        print(f"[tb3_live] Missing keys on load (first 5): {missing[:5]}")
    if unexpected:
        print(f"[tb3_live] Unexpected keys on load (first 5): {unexpected[:5]}")

    if disable_lora:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("--disable_lora requires the peft package") from exc
        for _, module in model.named_modules():
            if isinstance(module, PeftModel):
                module.disable_adapter_layers()

    mp = getattr(args, "mixed_precision", "no")
    if mp == "bf16":
        model_dtype = torch.bfloat16
    elif mp == "fp16":
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    model = model.to(device=device, dtype=model_dtype)
    model.eval()
    return model, model_dtype


class Tb3LiveInferenceNode(Node):
    def __init__(self, cli_args, model, model_dtype, model_args, device):
        super().__init__("tb3_live_inference")
        self.cli_args = cli_args
        self.model = model
        self.model_dtype = model_dtype
        self.model_args = model_args
        self.device = device
        self.robot_names = split_csv(cli_args.robot_names)
        self.robot_count = len(self.robot_names)

        self.latest_frame = None
        self.latest_frame_time = 0.0
        self.frame_buffer = deque(maxlen=cli_args.window_size)
        self.obs_buffer = deque(maxlen=cli_args.window_size)
        self.adj_buffer = deque(maxlen=cli_args.window_size)
        self.prompt_buffer = deque(maxlen=cli_args.window_size)

        self.poses = {name: None for name in self.robot_names}
        self.commands = {name: (0.0, 0.0) for name in self.robot_names}
        self.measured_velocities = {name: (0.0, 0.0) for name in self.robot_names}
        self.measured_speeds = {name: 0.0 for name in self.robot_names}
        self.goals = {name: None for name in self.robot_names}

        self.episode_index = 0
        self.episode_id = ""
        self.step_index = 0
        self.cumulative_reward = 0.0
        self.previous_goal_distances = None
        self.reached_once = [False for _ in self.robot_names]
        self.goal_signature = None
        self.last_inference_time = 0.0

        output_path = Path(cli_args.output_csv).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = output_path.open("w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "episode_id",
                "step",
                "prediction",
                "reward",
                "cumulative_reward",
                "target",
                "abs_error",
            ]
        )
        self.csv_file.flush()

        self.prediction_pub = self.create_publisher(String, cli_args.prediction_topic, 10)
        self.create_subscription(CompressedImage, cli_args.image_topic, self.on_image, 10)
        for name in self.robot_names:
            self.create_subscription(
                PoseStamped,
                f"/{name}/cv_pose",
                lambda msg, robot=name: self.on_pose(robot, msg),
                10,
            )
            self.create_subscription(
                Twist,
                f"/{name}/cmd_vel",
                lambda msg, robot=name: self.on_cmd(robot, msg),
                10,
            )
            self.create_subscription(
                Twist,
                f"/{name}/cv_measured_velocity",
                lambda msg, robot=name: self.on_measured_velocity(robot, msg),
                10,
            )
            self.create_subscription(
                PointStamped,
                f"/{name}/mppi_goal",
                lambda msg, robot=name: self.on_goal(robot, msg),
                10,
            )

        timer_period = 1.0 / max(0.1, float(cli_args.inference_rate_hz))
        self.timer = self.create_timer(timer_period, self.on_timer)
        self.get_logger().info(
            "MA-VLCM live inference waiting for frames, poses, and goals on fleet topics."
        )

    def destroy_node(self):
        try:
            self.csv_file.close()
        finally:
            super().destroy_node()

    def on_image(self, msg):
        data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning("Failed to decode compressed overhead frame.")
            return
        self.latest_frame = frame
        self.latest_frame_time = time.monotonic()

    def on_pose(self, robot, msg):
        q = msg.pose.orientation
        self.poses[robot] = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            yaw_from_quat_xyzw(q.x, q.y, q.z, q.w),
        )

    def on_cmd(self, robot, msg):
        self.commands[robot] = (float(msg.linear.x), float(msg.angular.z))

    def on_measured_velocity(self, robot, msg):
        self.measured_velocities[robot] = (float(msg.linear.x), float(msg.angular.z))
        self.measured_speeds[robot] = float(msg.linear.z)

    def on_goal(self, robot, msg):
        self.goals[robot] = (float(msg.point.x), float(msg.point.y))

    def on_timer(self):
        if not self._ready():
            return

        current_signature = self._goal_signature()
        if self._goal_signature_changed(current_signature):
            self._start_episode(current_signature)

        current_step = self.step_index
        state_json, reward = self._build_state_and_reward()
        self.frame_buffer.append(self.latest_frame.copy())
        self.obs_buffer.append(self._state_to_robot_obs(state_json))
        self.adj_buffer.append(self._state_to_adj(state_json))
        self.prompt_buffer.append(self._build_prompt(state_json))

        if len(self.frame_buffer) < self.cli_args.window_size:
            self.step_index += 1
            return

        try:
            prediction = self.run_inference()
        except Exception as exc:
            self.get_logger().error(f"MA-VLCM inference failed: {exc}")
            return

        target = float(self.cumulative_reward)
        abs_error = abs(float(prediction) - target)
        payload = {
            "episode_id": self.episode_id,
            "step": int(current_step),
            "prediction": float(prediction),
            "reward": float(reward),
            "cumulative_reward": target,
            "target": target,
            "target_label": "cumulative_reward",
            "abs_error": float(abs_error),
            "window_size": int(self.cli_args.window_size),
        }
        out_msg = String()
        out_msg.data = json.dumps(payload)
        self.prediction_pub.publish(out_msg)
        self.csv_writer.writerow(
            [
                self.episode_id,
                int(current_step),
                float(prediction),
                float(reward),
                target,
                target,
                float(abs_error),
            ]
        )
        self.csv_file.flush()
        self.step_index += 1

    def _ready(self):
        return (
            self.latest_frame is not None
            and all(self.poses[name] is not None for name in self.robot_names)
            and all(self.goals[name] is not None for name in self.robot_names)
        )

    def _goal_signature(self):
        return tuple(
            (round(self.goals[name][0], 3), round(self.goals[name][1], 3))
            for name in self.robot_names
        )

    def _goal_signature_changed(self, signature):
        if self.goal_signature is None:
            return True
        eps = float(self.cli_args.goal_change_epsilon_m)
        for old, new in zip(self.goal_signature, signature):
            if distance(old, new) > eps:
                return True
        return False

    def _start_episode(self, signature):
        self.episode_index += 1
        self.episode_id = f"tb3_live_{self.episode_index:04d}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.step_index = 0
        self.cumulative_reward = 0.0
        self.previous_goal_distances = self._goal_distances()
        self.reached_once = [
            d <= self.cli_args.goal_radius_m for d in self.previous_goal_distances
        ]
        self.goal_signature = signature
        self.frame_buffer.clear()
        self.obs_buffer.clear()
        self.adj_buffer.clear()
        self.prompt_buffer.clear()
        self.get_logger().info(f"Started MA-VLCM live episode {self.episode_id}")

    def _positions(self):
        return [(self.poses[name][0], self.poses[name][1]) for name in self.robot_names]

    def _distance_matrix(self):
        positions = self._positions()
        out = np.zeros((self.robot_count, self.robot_count), dtype=np.float32)
        for i in range(self.robot_count):
            for j in range(i + 1, self.robot_count):
                d = distance(positions[i], positions[j])
                out[i, j] = d
                out[j, i] = d
        return out

    def _goal_distances(self):
        distances = []
        for name in self.robot_names:
            pose = self.poses[name]
            goal = self.goals[name]
            distances.append(distance((pose[0], pose[1]), goal))
        return distances

    def _build_state_and_reward(self):
        dist_mat = self._distance_matrix()
        goal_distances = self._goal_distances()
        reached_now = [d <= self.cli_args.goal_radius_m for d in goal_distances]
        agent_rewards, scalar_reward = self._compute_step_rewards(
            goal_distances,
            reached_now,
            dist_mat,
        )
        self.cumulative_reward += scalar_reward

        agents = []
        for i, name in enumerate(self.robot_names):
            pose = self.poses[name]
            goal = self.goals[name]
            cmd = self.commands[name]
            measured = self.measured_velocities[name]
            row = dist_mat[i].copy()
            if len(row) > i:
                row[i] = np.inf
            min_neighbor = float(np.min(row)) if len(row) > 1 else 0.0
            collision = min_neighbor < self.cli_args.proximity_penalty_distance_m
            agents.append(
                {
                    "id": i,
                    "domain_id": i + 1,
                    "robot": name,
                    "color": ROBOT_COLORS[i] if i < len(ROBOT_COLORS) else "unknown",
                    "goal_label": chr(ord("A") + i),
                    "goal_pos": [float(goal[0]), float(goal[1])],
                    "pos": [float(pose[0]), float(pose[1])],
                    "yaw": float(pose[2]),
                    "vel": [float(cmd[0]), float(cmd[1])],
                    "measured_vel": [float(measured[0]), float(measured[1])],
                    "measured_speed": float(self.measured_speeds[name]),
                    "dist_to_goal": float(goal_distances[i]),
                    "min_neighbor_dist": min_neighbor,
                    "reached": bool(reached_now[i]),
                    "collision": bool(collision),
                    "failure": False,
                    "action": "STOP" if reached_now[i] else "FORWARD",
                    "reward": float(agent_rewards[i]),
                }
            )

        self.previous_goal_distances = goal_distances
        self.reached_once = [old or new for old, new in zip(self.reached_once, reached_now)]
        done = all(reached_now)
        return (
            {
                "episode_meta": {
                    "episode_id": self.episode_id,
                    "episode_index": self.episode_index,
                    "step": self.step_index,
                    "done": bool(done),
                    "success": bool(done),
                    "failure": False,
                    "outcome": "success" if done else "running",
                    "termination_reason": "all_reached" if done else "",
                },
                "agents": agents,
                "reward": float(np.mean(agent_rewards) if agent_rewards else 0.0),
                "cumulative_reward": float(self.cumulative_reward),
            },
            scalar_reward,
        )

    def _compute_step_rewards(self, goal_distances, reached_now, dist_mat):
        if self.previous_goal_distances is None:
            self.previous_goal_distances = goal_distances
        rewards = []
        for prev, current, was_reached, is_reached in zip(
            self.previous_goal_distances,
            goal_distances,
            self.reached_once,
            reached_now,
        ):
            reward = self.cli_args.progress_scale * (float(prev) - float(current))
            if is_reached and not was_reached:
                reward += self.cli_args.success_reward
            rewards.append(float(reward))

        team_penalty = 0.0
        for i in range(dist_mat.shape[0]):
            for j in range(i + 1, dist_mat.shape[1]):
                if float(dist_mat[i, j]) < self.cli_args.proximity_penalty_distance_m:
                    team_penalty += self.cli_args.proximity_penalty
        scalar = float(np.mean(rewards) if rewards else 0.0) + team_penalty
        return rewards, scalar

    def _state_to_robot_obs(self, state_json):
        rows = []
        for ag in state_json["agents"]:
            yaw = float(ag.get("yaw", 0.0))
            vel = ag.get("vel", [0.0, 0.0])
            pos = ag.get("pos", [0.0, 0.0])
            rows.append(
                [
                    float(pos[0]),
                    float(pos[1]),
                    float(np.cos(yaw)),
                    float(np.sin(yaw)),
                    float(vel[0]),
                    float(vel[1]),
                    float(ag.get("dist_to_goal", 0.0)),
                    float(ag.get("min_neighbor_dist", 0.0)),
                ]
            )
        return np.asarray(rows, dtype=np.float32)

    def _state_to_adj(self, state_json):
        positions = np.asarray([ag["pos"] for ag in state_json["agents"]], dtype=np.float32)
        adj = np.zeros((len(positions), len(positions)), dtype=np.float32)
        if len(positions) == 0:
            return adj
        diffs = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        adj[dists < self.cli_args.adjacency_threshold] = 1.0
        return adj

    def _build_prompt(self, state_json):
        agents = state_json.get("agents", [])
        meta = state_json.get("episode_meta", {})
        obs_lines = []
        for ag in agents:
            pos = ag.get("pos", [0.0, 0.0])
            goal = ag.get("goal_pos", [0.0, 0.0])
            vel = ag.get("vel", [0.0, 0.0])
            reached = "yes" if ag.get("reached", False) else "no"
            collision = "yes" if ag.get("collision", False) else "no"
            obs_lines.append(
                f"Agent {ag.get('id', 0)} ({ag.get('color', 'unknown')}): "
                f"position ({pos[0]:.2f}, {pos[1]:.2f}), "
                f"heading {ag.get('yaw', 0.0):.2f} rad, "
                f"linear_vel {float(vel[0]):.2f} m/s, angular_vel {float(vel[1]):.2f} rad/s, "
                f"goal {ag.get('goal_label', '?')} at ({goal[0]:.2f}, {goal[1]:.2f}), "
                f"dist_to_goal {ag.get('dist_to_goal', 0.0):.2f}m, "
                f"min_neighbor_dist {ag.get('min_neighbor_dist', 0.0):.2f}m, "
                f"reached: {reached}, collision: {collision}."
            )
        header = (
            "You are an expert vision language critic model for multi-agent teams able to critize given trajectories of data for their n-step returns, thus critizing the policy. "
            f"This is a real indoor TurtleBot3 lab environment with {len(agents)} agents observed from a bird's-eye webcam view. "
            "The visual input is a native overhead camera view of multiple TurtleBot3 robots moving on the floor; the robots may not have visible IDs, labels, or color markers in the image. "
            "Robot identity, goal assignment, goal coordinates, and distance-to-goal are provided by the structured observations rather than by visual labels. "
            f"The reward is the mean progress toward the assigned goals, plus +{self.cli_args.success_reward:g} when an agent reaches its goal for the first time, "
            f"and {self.cli_args.proximity_penalty:g} if any pair of agents comes within {self.cli_args.proximity_penalty_distance_m:.2f}m. "
            "Traversability information is unavailable in this environment. "
            "Predict the expected infinite horizon return of the current policy based on these observations: "
            f"Timestep: {meta.get('step', self.step_index)}. "
            f"Episode outcome token: {meta.get('outcome', 'running')}. "
            f"Termination reason: {meta.get('termination_reason', '') or 'none'}. "
        )
        return header + " ".join(obs_lines)

    def run_inference(self):
        frames = [
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for frame in list(self.frame_buffer)
        ]
        prompt = self.prompt_buffer[-1]
        processor = self.model.backbone.processor
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            vocab = tokenizer.get_vocab()
            if "<video>" in vocab and "<video>" not in prompt and "<image>" not in prompt:
                prompt = f"<video>\n{prompt}"
            if "<obs>" in vocab and "<obs>" not in prompt:
                if "<video>" in prompt:
                    prompt = prompt.replace("<video>\n", "<video><obs>\n", 1)
                else:
                    prompt = f"<obs>\n{prompt}"

        inputs = processor(
            text=prompt,
            videos=frames,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=getattr(self.model_args, "vl_max_text_len", 256),
        )
        robot_obs = torch.tensor(
            np.stack(list(self.obs_buffer), axis=0),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        adj = torch.tensor(
            np.stack(list(self.adj_buffer), axis=0),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        inputs = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in dict(inputs).items()
        }

        with torch.no_grad():
            pred = self.model(video=inputs, robot_obs=robot_obs, adj=adj)
        return float(pred.view(-1)[0].detach().cpu().item())


def main():
    if _ROS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "tb3_live_inference requires a ROS 2 Python environment with rclpy, "
            "geometry_msgs, sensor_msgs, and std_msgs available."
        ) from _ROS_IMPORT_ERROR

    cli_args = parse_args()
    ckpt, model_args = _load_checkpoint(cli_args.checkpoint)

    if cli_args.device:
        device = torch.device(cli_args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model, model_dtype = _build_loaded_model(
        ckpt, model_args, device=device, disable_lora=cli_args.disable_lora
    )
    print(f"[tb3_live] Loaded checkpoint {cli_args.checkpoint} on {device} ({model_dtype})")

    rclpy.init()
    node = Tb3LiveInferenceNode(cli_args, model, model_dtype, model_args, device)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
