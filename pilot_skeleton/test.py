"""Inspect one local or Hugging Face episode using the pilot data adapter."""

import argparse

from data.episode_adapter_pseudocode import EpisodeAdapter


parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset",
    nargs="?",
    default="shulnak09/formation-vlcm-polygon-3to6",
    help="HF dataset repo ID or local episode archive/dataset directory",
)
parser.add_argument("--revision", default="main")
args = parser.parse_args()

adapter = EpisodeAdapter()
episodes = adapter.discover_episodes(args.dataset, revision=args.revision)
episode = adapter.load_episode(episodes[0])
print("episode:", episode["episode_id"])
print("frames:", len(episode["images"]))
print("capture/control Hz:", episode["capture_hz"], episode["control_hz"])
print("20 Hz reward stream:", episode["control_rewards_available"])
print("robots:", episode["num_agents"])
print("robot_obs:", episode["robot_obs"].shape)
print("adjacency:", episode["adj"].shape)
print("success:", episode["success"])
print("valid:", adapter.validate_episode(episode))
print("first prompt:", episode["prompt"])
print("first frame size:", episode["images"][0].size)
print("first clip:", {key: value for key, value in adapter.make_clips(
    episode, {"clip_length": 8, "stride": 8, "horizon": 4}
)[0].items() if key not in ("images", "robot_obs", "adj")})
