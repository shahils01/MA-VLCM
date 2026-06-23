#!/usr/bin/env python3
"""Create/update the Hugging Face TB3 lab VLCM dataset repository."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="data/tb3_lab",
        help="Directory containing TB3 lab WebDataset .tar shards.",
    )
    parser.add_argument(
        "--repo-id",
        default="adi2440/tb3-lab",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create the dataset as public instead of private.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch/revision to upload to.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")
    shard_count = len(list(dataset_dir.rglob("*.tar")))
    if shard_count == 0:
        raise SystemExit(f"No .tar shards found under: {dataset_dir}")

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(dataset_dir),
        path_in_repo=".",
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        commit_message=f"Upload {shard_count} TB3 lab VLCM shards",
    )
    print(f"Uploaded {shard_count} shards from {dataset_dir} to {args.repo_id}")


if __name__ == "__main__":
    main()
