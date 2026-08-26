from collections import Counter

from ma_vlcm.sampling import (
    balanced_tb3_shards,
    split_tb3_shards_by_stratum,
    tb3_shard_stratum,
)
from ma_vlcm.train import _expand_hf_dataset_shards


source = (
    "hf://datasets/adi2440/tb3-isaac-vlcm/**/*.tar;"
    "/scratch/aparame/VLCM/Only_VLCM_Data/VLCM_Data_Collection/"
    "TURTLEBOT/data/**/*.tar"
)
files = _expand_hf_dataset_shards(source)
domains = ("goal_to_goal", "static_obstacles")
agent_counts = (3, 4, 5, 6)
train, validation = split_tb3_shards_by_stratum(
    files,
    agent_counts,
    domains,
    val_split=0.2,
    seed=42,
    require_all=False,
)
balanced = balanced_tb3_shards(
    train, agent_counts, domains, balance_mode="domain"
)
print(
    {
        "all": len(files),
        "train": len(train),
        "validation": len(validation),
        "balanced_train": len(balanced),
    }
)
print(
    Counter(
        tb3_shard_stratum(
            path,
            (3, 4, 5, 6),
            ("goal_to_goal", "static_obstacles"),
        )
        for path in files
    )
)
