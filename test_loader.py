import sys
import os
import torch
import resource
from types import SimpleNamespace
from train import SequenceWebDataset
import argparse


def print_mem():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux returns maxrss in KB
    print(f"Mem usage: {usage / 1024:.2f} MB", flush=True)


class MockTokenizer:
    def get_vocab(self):
        return {"<video>": 1, "<obs>": 2, "<image>": 3}

    def add_special_tokens(self, tokens):
        pass

    def convert_tokens_to_ids(self, token):
        return 1


class MockProcessor:
    def __init__(self):
        self.tokenizer = MockTokenizer()

    def __call__(
        self, text=None, videos=None, images=None, return_tensors="pt", **kwargs
    ):
        # Return dummy tensors
        return {
            "pixel_values_videos": torch.randn(1, 3, 8, 224, 224),
            "input_ids": torch.randint(0, 100, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        }

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


def test(dir_path):

    shards = dir_path + "/*.tar"
    print(f"Testing shards: {shards}", flush=True)

    # Use our mock processor
    mock_proc = MockProcessor()

    ds = SequenceWebDataset(
        shards=shards,
        clip_len=8,
        clip_stride=1,
        text_mode="raw",
        robot_source="obs",
        reward_reduce="mean",
        done_reduce="any",
        vlm_processor=mock_proc,
        vl_model_name=None,
        robot_obs_dim=6,
        text_prompt_template="Test prompt",
        dataset_type="rware",
        rware_config="tiny-2ag-easy-v2",
    )

    print("Iterating dataset...", flush=True)
    count = 0
    for i, sample in enumerate(ds):
        count += 1
        if i % 10 == 0:
            print(f"Step {i}", flush=True)
            print_mem()

        # Verify sample structure
        if i == 0:
            print("Sample keys:", sample.keys(), flush=True)
            if "inputs" in sample:
                print("Inputs keys:", sample["inputs"].keys(), flush=True)

        if i >= 50:
            break

    print(f"Successfully iterated {count} samples.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, required=True)
    args = parser.parse_args()
    test(args.dir_path)
