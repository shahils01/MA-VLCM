#!/usr/bin/env python3
"""Live MA-VLCM inference for the TurtleBot3 lab runtime."""

import argparse
import csv
import json
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
    from cv_bridge import CvBridge
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from sensor_msgs.msg import Image as RosImage
    from std_msgs.msg import String
    from tb3_lab_msgs.msg import TeamState

    _ROS_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    rclpy = None
    CvBridge = None
    ExternalShutdownException = None
    Node = object
    RosImage = None
    String = None
    TeamState = None
    _ROS_IMPORT_ERROR = exc


def parse_args():
    p = argparse.ArgumentParser(description="Run live VLCM inference on TurtleBot3 lab topics.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--window_size", type=int, default=16)
    p.add_argument("--adjacency_threshold", type=float, default=1.0)
    p.add_argument("--state_topic", type=str, default="/tb3_lab/team_state")
    p.add_argument("--image_topic", type=str, default="/tb3_lab/overhead_warped")
    p.add_argument("--prediction_topic", type=str, default="/tb3_lab/vlcm_prediction")
    p.add_argument("--output_csv", type=str, default="outputs/results/tb3_live_predictions.csv")
    p.add_argument("--disable_lora", action="store_true")
    return p.parse_args()


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
        video_mean=tuple(_arg(args, "video_mean", (0.5, 0.5, 0.5))),
        video_std=tuple(_arg(args, "video_std", (0.5, 0.5, 0.5))),
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
            "PEFT/LoRA checkpoint loading requires the 'peft' package. "
            "Install it in the ROS Python with `/usr/bin/python3 -m pip install --user peft`."
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


def _build_prompt(state_msg: TeamState) -> str:
    obs_lines = []
    for ag in state_msg.agents:
        reached = "yes" if ag.reached else "no"
        collision = "yes" if ag.collision else "no"
        obs_lines.append(
            f"Agent {ag.robot_id} ({ag.color}): position ({ag.x:.2f}, {ag.y:.2f}), "
            f"heading {ag.yaw:.2f} rad, linear_vel {ag.v_lin:.2f} m/s, angular_vel {ag.v_ang:.2f} rad/s, "
            f"goal {ag.goal_label} at ({ag.goal_x:.2f}, {ag.goal_y:.2f}), "
            f"dist_to_goal {ag.dist_to_goal:.2f}m, min_neighbor_dist {ag.min_neighbor_dist:.2f}m, "
            f"reached: {reached}, collision: {collision}."
        )
    header = (
        "You are an expert vision language critic model for multi-agent teams able to critize given trajectories of data for their n-step returns, thus critizing the policy. "
        f"This is a real indoor TurtleBot3 lab environment with {len(state_msg.agents)} agents observed from a bird's-eye webcam view. "
        "The agents are color-coded red, blue, and green, and must navigate to fixed floor goals labeled A, B, and C. "
        "The reward is the mean progress toward the assigned goals, plus +5 when an agent reaches its goal for the first time, and -1 if any pair of agents comes within 0.25m. "
        "Traversability information is unavailable in this environment. "
        "Predict the expected infinite horizon return of the current policy based on these observations: "
        f"Timestep: {state_msg.step}. "
    )
    return header + " ".join(obs_lines)


def _teamstate_to_robot_obs(state_msg: TeamState) -> np.ndarray:
    rows = []
    for ag in state_msg.agents:
        rows.append(
            [
                ag.x,
                ag.y,
                np.cos(ag.yaw),
                np.sin(ag.yaw),
                ag.v_lin,
                ag.v_ang,
                ag.dist_to_goal,
                ag.min_neighbor_dist,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _teamstate_to_adj(state_msg: TeamState, threshold: float) -> np.ndarray:
    n = len(state_msg.agents)
    adj = np.zeros((n, n), dtype=np.float32)
    positions = np.asarray([[ag.x, ag.y] for ag in state_msg.agents], dtype=np.float32)
    if n == 0:
        return adj
    diffs = positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    adj[dists < threshold] = 1.0
    return adj


class Tb3LiveInferenceNode(Node):
    def __init__(self, cli_args, model, model_dtype, model_args, device):
        super().__init__("tb3_live_inference")
        self.cli_args = cli_args
        self.model = model
        self.model_dtype = model_dtype
        self.model_args = model_args
        self.device = device
        self.bridge = CvBridge()

        self.latest_frame = None
        self.latest_frame_stamp = None
        self.frame_buffer = deque(maxlen=cli_args.window_size)
        self.obs_buffer = deque(maxlen=cli_args.window_size)
        self.adj_buffer = deque(maxlen=cli_args.window_size)
        self.prompt_buffer = deque(maxlen=cli_args.window_size)

        output_path = Path(cli_args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = output_path.open("w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["episode_id", "step", "prediction", "reward", "cumulative_reward"])
        self.csv_file.flush()

        self.prediction_pub = self.create_publisher(String, cli_args.prediction_topic, 10)
        self.image_sub = self.create_subscription(
            RosImage, cli_args.image_topic, self.on_image, 10
        )
        self.state_sub = self.create_subscription(
            TeamState, cli_args.state_topic, self.on_team_state, 10
        )

    def destroy_node(self):
        try:
            self.csv_file.close()
        finally:
            super().destroy_node()

    def on_image(self, msg: RosImage):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.latest_frame = frame
        self.latest_frame_stamp = self.get_clock().now()

    def on_team_state(self, msg: TeamState):
        if self.latest_frame is None:
            return
        if self.latest_frame_stamp is None:
            return

        self.frame_buffer.append(self.latest_frame.copy())
        self.obs_buffer.append(_teamstate_to_robot_obs(msg))
        self.adj_buffer.append(_teamstate_to_adj(msg, self.cli_args.adjacency_threshold))
        self.prompt_buffer.append(_build_prompt(msg))

        if len(self.frame_buffer) < self.cli_args.window_size:
            return

        prediction = self.run_inference()
        payload = {
            "episode_id": msg.episode_id,
            "step": int(msg.step),
            "prediction": float(prediction),
            "reward": float(msg.reward),
            "cumulative_reward": float(msg.cumulative_reward),
        }
        out_msg = String()
        out_msg.data = json.dumps(payload)
        self.prediction_pub.publish(out_msg)
        self.csv_writer.writerow(
            [
                msg.episode_id,
                int(msg.step),
                float(prediction),
                float(msg.reward),
                float(msg.cumulative_reward),
            ]
        )
        self.csv_file.flush()

    def run_inference(self) -> float:
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
            np.stack(list(self.obs_buffer), axis=0), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        adj = torch.tensor(
            np.stack(list(self.adj_buffer), axis=0), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in dict(inputs).items()}

        with torch.no_grad():
            pred = self.model(video=inputs, robot_obs=robot_obs, adj=adj)
        return float(pred.view(-1)[0].detach().cpu().item())


def main():
    if _ROS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "tb3_live_inference requires a ROS 2 Python environment with rclpy, "
            "cv_bridge, sensor_msgs, and tb3_lab_msgs available."
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
