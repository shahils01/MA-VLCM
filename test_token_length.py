"""
Sanity-check script for VLCM training inputs/outputs.
Loads a real batch from the dataloader and prints:
  - Token lengths (input_ids shape, non-padding tokens)
  - Decoded text prompt sent to the VLM
  - Video frame info (shape, dtype)
  - Robot observations and adjacency matrix
  - Reward / done / returns (training targets)
"""

import sys
import os
import torch
import argparse
from torch.utils.data import DataLoader
from train import SequenceWebDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch/aparame/Research/" "VLCM_Data_Collection/data_scratch",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--clip_len", type=int, default=5)
    parser.add_argument("--num_robots", type=int, default=4)
    parser.add_argument("--robot_obs_dim", type=int, default=6)
    parser.add_argument(
        "--vl_model_name",
        type=str,
        default="llava-hf/LLaVA-NeXT-Video-7B-32K-hf",
    )
    parser.add_argument(
        "--vl_max_text_len",
        type=int,
        default=1024,
    )
    args = parser.parse_args()

    print(f"Loading processor: {args.vl_model_name}")
    print(f"Data dir: {args.data_dir}")
    print(f"Batch size: {args.batch_size}, " f"clip_len: {args.clip_len}\n")

    ds = SequenceWebDataset(
        shards=args.data_dir,
        clip_len=args.clip_len,
        clip_stride=1,
        text_mode="raw",
        robot_source="obs",
        reward_reduce="mean",
        done_reduce="any",
        vlm_processor=None,
        vl_model_name=args.vl_model_name,
        robot_obs_dim=args.robot_obs_dim,
        num_robots=args.num_robots,
        max_num_robots=args.num_robots,
        shuffle_shards=False,
        text_prompt_template=None,
        dataset_type="rware",
        rware_config="mixed-rware",
        return_mode="nstep",
        n_step=50,
        gamma=0.99,
        keep_raw_video=False,
        include_next=False,
        vlm_max_text_len=args.vl_max_text_len,
        vlm_truncation=False,
        vlm_padding="longest",
        resize_width=672,
        resize_height=336,
    )

    # Collate function (same as train.py)
    def _collate(batch):
        def _stack_inputs(items):
            out = {}
            if not items:
                return out
            keys = items[0].keys()
            for k in keys:
                vals = [d[k] for d in items]
                if torch.is_tensor(vals[0]):
                    if k in ["input_ids", "attention_mask"]:
                        out[k] = torch.nn.utils.rnn.pad_sequence(
                            vals,
                            batch_first=True,
                            padding_value=0,
                        )
                    elif k == "labels":
                        out[k] = torch.nn.utils.rnn.pad_sequence(
                            vals,
                            batch_first=True,
                            padding_value=-100,
                        )
                    else:
                        out[k] = torch.stack(vals, dim=0)
                else:
                    out[k] = vals
            return out

        out = {
            "inputs": _stack_inputs([b["inputs"] for b in batch]),
            "robot_obs": torch.stack([b["robot_obs"] for b in batch], dim=0),
            "adj": torch.stack([b["adj"] for b in batch], dim=0),
            "reward": torch.stack([b["reward"] for b in batch], dim=0).view(-1),
            "done": torch.stack([b["done"] for b in batch], dim=0).view(-1),
        }
        if "returns" in batch[0]:
            out["returns"] = torch.stack([b["returns"] for b in batch], dim=0).view(-1)
        return out

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=_collate,
    )

    # Load the tokenizer for decoding
    from transformers import LlavaNextVideoProcessor

    processor = LlavaNextVideoProcessor.from_pretrained(args.vl_model_name)
    tokenizer = processor.tokenizer

    print("Fetching first batch...\n")
    for batch in loader:
        sep = "=" * 60
        print(sep)
        print("BATCH OVERVIEW")
        print(sep)

        inputs = batch["inputs"]
        bs = batch["reward"].shape[0]
        print(f"Batch size: {bs}")

        # --- Input IDs / Token Info ---
        print(f"\n--- TOKEN INFO ---")
        input_ids = inputs["input_ids"]
        attn_mask = inputs.get("attention_mask")
        print(f"input_ids shape: {tuple(input_ids.shape)}")
        if attn_mask is not None:
            print(f"attention_mask shape: " f"{tuple(attn_mask.shape)}")

        for i in range(bs):
            ids = input_ids[i]
            non_pad = (ids != 0).sum().item()
            print(
                f"\n  Sample {i}: "
                f"seq_len={ids.shape[0]}, "
                f"non_pad_tokens={non_pad}"
            )

            # Decode and print the text prompt
            # Filter out padding
            valid_ids = ids[ids != 0]
            decoded = tokenizer.decode(
                valid_ids,
                skip_special_tokens=False,
            )
            # Truncate very long decoded text for display
            if len(decoded) > 800:
                decoded = decoded[:400] + "\n  ... [TRUNCATED] ...\n  " + decoded[-400:]
            print(f"  DECODED TEXT PROMPT:")
            print(f"  {decoded}")

        # --- Video / Pixel Values ---
        print(f"\n--- VIDEO / IMAGE INFO ---")
        for k in sorted(inputs.keys()):
            v = inputs[k]
            if torch.is_tensor(v):
                print(f"  {k}: shape={tuple(v.shape)}, " f"dtype={v.dtype}")
                if "pixel" in k:
                    print(
                        f"    min={v.min().item():.4f}, "
                        f"max={v.max().item():.4f}, "
                        f"mean={v.mean().item():.4f}"
                    )

        # --- Robot Observations ---
        print(f"\n--- ROBOT OBSERVATIONS ---")
        rob = batch["robot_obs"]
        print(f"  robot_obs shape: {tuple(rob.shape)}")
        print(f"  (B, T, N_robots, obs_dim)")
        for i in range(bs):
            last_step = rob[i, -1]  # last timestep
            print(f"  Sample {i} " f"(last timestep, all robots):")
            print(f"    {last_step}")

        # --- Adjacency Matrix ---
        print(f"\n--- ADJACENCY MATRIX ---")
        adj = batch["adj"]
        print(f"  adj shape: {tuple(adj.shape)}")
        print(f"  (B, T, N_robots, N_robots)")
        for i in range(bs):
            print(f"  Sample {i} (last timestep):")
            print(f"    {adj[i, -1]}")

        # --- Training Targets ---
        print(f"\n--- TRAINING TARGETS ---")
        print(f"  reward: {batch['reward']}")
        print(f"  done:   {batch['done']}")
        if "returns" in batch:
            print(f"  returns: {batch['returns']}")

        # --- Token Length Stats ---
        print(f"\n--- TOKEN LENGTH SUMMARY ---")
        lengths = [(input_ids[i] != 0).sum().item() for i in range(bs)]
        print(f"  Token lengths: {lengths}")
        print(f"  Max: {max(lengths)}, " f"Mean: {sum(lengths)/len(lengths):.1f}")
        print(f"  Current vl_max_text_len: " f"{args.vl_max_text_len}")
        print(f"  Recommended: " f"{max(lengths) + 64} (max + 64 headroom)")

        print(f"\n{sep}")
        print("END OF BATCH")
        print(sep)
        break  # Only print one batch


if __name__ == "__main__":
    main()
