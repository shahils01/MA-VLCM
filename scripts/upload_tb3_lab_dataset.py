#!/usr/bin/env python3
"""Create/update the Hugging Face TB3 lab VLCM dataset repository."""

import argparse
import re
from pathlib import Path

from huggingface_hub import HfApi
try:
    from huggingface_hub.utils import HfHubHTTPError
except Exception:  # pragma: no cover - older huggingface_hub versions
    HfHubHTTPError = Exception


SEQUENCED_SHARD_RE = re.compile(r"^(\d+)\.tar$")


def _unique_paths(paths):
    out = []
    seen = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    return out


def resolve_dataset_dir(dataset_dir: str) -> Path:
    """Resolve dataset_dir, accepting both tb3_lab and tb3-lab spellings."""

    requested = Path(dataset_dir).expanduser()
    script_root = Path(__file__).resolve().parents[1]
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(script_root / requested)

    text = str(requested)
    aliases = []
    if "tb3-lab" in text:
        aliases.append(Path(text.replace("tb3-lab", "tb3_lab")))
    if "tb3_lab" in text:
        aliases.append(Path(text.replace("tb3_lab", "tb3-lab")))
    for alias in aliases:
        candidates.append(alias)
        if not alias.is_absolute():
            candidates.append(script_root / alias)

    for candidate in _unique_paths(candidates):
        if candidate.is_dir():
            return candidate

    checked = "\n  ".join(str(path) for path in _unique_paths(candidates))
    raise SystemExit(f"Dataset directory does not exist. Checked:\n  {checked}")


def shard_sort_key(path: Path):
    """Sort numeric shards numerically, then timestamped/new shards stably."""

    match = SEQUENCED_SHARD_RE.match(path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    return (1, path.stat().st_mtime, path.name)


def get_remote_shard_numbers(api: HfApi, repo_id: str, revision: str):
    """Return numeric .tar shard ids already present in the HF dataset repo."""

    try:
        remote_files = api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )
    except HfHubHTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 404:
            return set()
        raise

    numbers = set()
    for remote_file in remote_files:
        name = Path(remote_file).name
        match = SEQUENCED_SHARD_RE.match(name)
        if match:
            numbers.add(int(match.group(1)))
    return numbers


def normalize_shard_filenames(
    dataset_dir: Path,
    dry_run: bool = False,
    reserved_numbers=None,
    renumber_reserved_conflicts: bool = False,
):
    """Rename non-sequential .tar shards to fill the numeric shard sequence."""

    reserved_numbers = set(reserved_numbers or [])
    shards = sorted(dataset_dir.rglob("*.tar"), key=shard_sort_key)
    if not shards:
        raise SystemExit(f"No .tar shards found under: {dataset_dir}")

    existing_numbers = {}
    to_rename = []
    for shard in shards:
        match = SEQUENCED_SHARD_RE.match(shard.name)
        if match:
            number = int(match.group(1))
            if (
                number in existing_numbers
                or (renumber_reserved_conflicts and number in reserved_numbers)
            ):
                to_rename.append(shard)
            else:
                existing_numbers[number] = shard
        else:
            to_rename.append(shard)

    if not to_rename:
        return shards, []

    used_numbers = set(existing_numbers) | reserved_numbers
    next_number = 0
    plan = []
    for shard in sorted(to_rename, key=shard_sort_key):
        while next_number in used_numbers:
            next_number += 1
        used_numbers.add(next_number)
        target = shard.with_name(f"{next_number}.tar")
        plan.append((shard, target))

    if dry_run:
        return shards, plan

    temp_plan = []
    for index, (src, target) in enumerate(plan):
        temp = src.with_name(f".renaming-{index:06d}-{src.name}")
        if temp.exists():
            raise SystemExit(f"Temporary rename path already exists: {temp}")
        src.rename(temp)
        temp_plan.append((temp, target))

    for temp, target in temp_plan:
        if target.exists():
            raise SystemExit(f"Refusing to overwrite existing shard: {target}")
        temp.rename(target)

    normalized = sorted(dataset_dir.rglob("*.tar"), key=shard_sort_key)
    return normalized, plan


def local_shard_numbers(shards):
    numbers = set()
    for shard in shards:
        match = SEQUENCED_SHARD_RE.match(shard.name)
        if match:
            numbers.add(int(match.group(1)))
    return numbers


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
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Upload shards as-is without renaming non-sequential files.",
    )
    parser.add_argument(
        "--no-remote-numbering",
        action="store_true",
        help=(
            "Do not inspect Hugging Face for existing numeric shards before "
            "renaming local files."
        ),
    )
    parser.add_argument(
        "--allow-overwrite-existing-remote",
        action="store_true",
        help=(
            "Allow local numeric shard names that already exist on Hugging Face. "
            "Without this, the script aborts to avoid accidental overwrites."
        ),
    )
    parser.add_argument(
        "--renumber-remote-conflicts",
        action="store_true",
        help=(
            "If local numeric shards conflict with Hugging Face filenames, "
            "rename them to the next available remote-safe numbers."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned renames and upload summary without changing files or uploading.",
    )
    args = parser.parse_args()

    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    api = HfApi()
    remote_numbers = set()
    if not args.no_remote_numbering:
        remote_numbers = get_remote_shard_numbers(api, args.repo_id, args.revision)
        if remote_numbers:
            print(
                f"Found {len(remote_numbers)} numeric shards on Hugging Face; "
                f"highest is {max(remote_numbers)}. New local shards will start "
                "after the remote sequence."
            )
        else:
            print("No existing numeric shards found on Hugging Face.")

    if args.no_normalize:
        shards = sorted(dataset_dir.rglob("*.tar"), key=shard_sort_key)
        rename_plan = []
        if not shards:
            raise SystemExit(f"No .tar shards found under: {dataset_dir}")
    else:
        shards, rename_plan = normalize_shard_filenames(
            dataset_dir,
            dry_run=args.dry_run,
            reserved_numbers=remote_numbers,
            renumber_reserved_conflicts=args.renumber_remote_conflicts,
        )

    if rename_plan:
        action = "Would rename" if args.dry_run else "Renamed"
        for src, target in rename_plan:
            print(f"{action}: {src.relative_to(dataset_dir)} -> {target.relative_to(dataset_dir)}")
    else:
        print("Shard filenames are already sequential.")

    remote_collisions = local_shard_numbers(shards) & remote_numbers
    if (
        remote_collisions
        and not args.allow_overwrite_existing_remote
        and not args.renumber_remote_conflicts
    ):
        examples = ", ".join(f"{n}.tar" for n in sorted(remote_collisions)[:10])
        raise SystemExit(
            "Local numeric shard names already exist on Hugging Face: "
            f"{examples}. Use --allow-overwrite-existing-remote to replace them, "
            "or --renumber-remote-conflicts to rename local conflicts to new "
            "remote-safe numbers."
        )

    shard_count = len(shards)
    if args.dry_run:
        print(
            f"Dry run complete: {shard_count} shards under {dataset_dir} "
            f"would be uploaded to {args.repo_id}"
        )
        return

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
