"""Choose a safe vision inference microbatch on the allocated A100."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import torch
from PIL import Image

from model import ModelConfig, SmolVLMBackbone
from python_mappo import make_env, python_collection_instruction
from train_mappo import parser


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        p.error("Run this preflight inside srun on a compute node")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU")
    if "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This experiment requires one A100")
    expected = "e6a9975efab55b1c8b284627cfb5efd1b998f017638cd1766f8c8b5ce4fd85a8"
    actual = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Supervised formation checkpoint hash mismatch: {actual}")
    source = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**source["model_config"])
    if cfg.backbone_mode != "frozen":
        raise RuntimeError("Expected a frozen supervised backbone")
    torch.set_num_threads(1)
    device = torch.device("cuda")
    backbone = SmolVLMBackbone(cfg, device).eval()
    args = parser().parse_args([])
    args.formation_spacing, args.formation_tolerance = .5, .075
    args.initial_position_range, args.min_start_separation = 1.25, .4
    args.env_max_steps, args.control_dt = 400, .05
    images, prompts = [], []
    for count in (3, 4, 5, 6):
        env = make_env(args, count)
        env.reset(seed=1 + count)
        images.append(Image.fromarray(env.render(size=256, view_size_m=3.048, tile_floor=False)))
        prompts.append(python_collection_instruction(.5))
    rows = []
    total_memory = torch.cuda.get_device_properties(0).total_memory
    for batch in (4, 8, 16, 32, 64):
        cfg.vl_batch_size = batch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            output = backbone((images * (batch // 4)), prompts * (batch // 4))
            torch.cuda.synchronize()
            del output
            start = time.monotonic()
            for _ in range(2):
                output = backbone(images * (batch // 4), prompts * (batch // 4))
                torch.cuda.synchronize()
                del output
            seconds = time.monotonic() - start
            reserved = torch.cuda.max_memory_reserved()
            row = dict(batch=batch, images_per_second=2 * batch / seconds,
                       peak_reserved_gib=reserved / 2**30, safe=reserved < .85 * total_memory)
            rows.append(row)
            print(json.dumps(row), flush=True)
            if not row["safe"]:
                break
        except torch.cuda.OutOfMemoryError:
            rows.append(dict(batch=batch, safe=False, error="CUDA out of memory"))
            torch.cuda.empty_cache()
            break
    safe = [row for row in rows if row["safe"]]
    if not safe:
        raise RuntimeError("No safe vision batch size found")
    selected = max(safe, key=lambda row: row["images_per_second"])["batch"]
    result = dict(host=socket.gethostname(), slurm_job_id=os.environ["SLURM_JOB_ID"],
                  gpu=torch.cuda.get_device_name(0), checkpoint_sha256=actual,
                  selected_vl_batch_size=selected, candidates=rows, model_config=asdict(cfg))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Selected vision microbatch: {selected}", flush=True)


if __name__ == "__main__":
    main()
