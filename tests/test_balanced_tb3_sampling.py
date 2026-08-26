from collections import Counter

import pytest

from ma_vlcm.sampling import (
    balanced_tb3_shards,
    split_tb3_shards_by_stratum,
    tb3_shard_stratum,
)


def test_zero_padded_cardinality_and_hf_repository_aliases_are_recognized():
    old = "/cache/datasets--adi2440--tb3-isaac-vlcm/agents_03/worker_00/a.tar"
    obstacle = (
        "/cache/datasets--adi2440--tb3-isaac-avoid-obstacles-vlcm/"
        "static_obstacles/agents_03/worker_00/b.tar"
    )
    domains = ("goal_to_goal", "static_obstacles")
    assert tb3_shard_stratum(old, (3,), domains) == ("goal_to_goal", 3)
    assert tb3_shard_stratum(obstacle, (3,), domains) == ("static_obstacles", 3)


def test_balanced_sampling_equalizes_domain_cardinality_strata():
    paths = [
        "goal_to_goal/agents_3/a.tar",
        "goal_to_goal/agents_3/b.tar",
        "goal_to_goal/agents_4/a.tar",
        "goal_to_goal/agents_5/a.tar",
        "static_obstacles/agents_3/a.tar",
        "static_obstacles/agents_4/a.tar",
        "static_obstacles/agents_5/a.tar",
    ]
    balanced = balanced_tb3_shards(
        paths, (3, 4, 5), ("goal_to_goal", "static_obstacles")
    )
    strata = Counter(
        (
            "static_obstacles" if "static_obstacles" in path else "goal_to_goal",
            next(count for count in (3, 4, 5) if f"agents_{count}" in path),
        )
        for path in balanced
    )
    assert set(strata.values()) == {2}


def test_balanced_sampling_fails_if_a_required_stratum_is_missing():
    with pytest.raises(ValueError, match="strata"):
        balanced_tb3_shards(
            ["goal_to_goal/agents_3/a.tar"],
            (3, 4, 5),
            ("goal_to_goal", "static_obstacles"),
        )


def test_domain_balancing_allows_different_cardinalities_per_domain():
    paths = [
        "goal_to_goal/agents_03/a.tar",
        "goal_to_goal/agents_04/a.tar",
        "goal_to_goal/agents_05/a.tar",
        "goal_to_goal/agents_06/a.tar",
        "static_obstacles/agents_03/a.tar",
        "static_obstacles/agents_03/b.tar",
    ]
    balanced = balanced_tb3_shards(
        paths,
        (3, 4, 5, 6),
        ("goal_to_goal", "static_obstacles"),
        balance_mode="domain",
    )
    domains = Counter(
        "static_obstacles" if "static_obstacles" in path else "goal_to_goal"
        for path in balanced
    )
    assert domains["goal_to_goal"] == domains["static_obstacles"] == 4

    train, validation = split_tb3_shards_by_stratum(
        paths,
        (3, 4, 5, 6),
        ("goal_to_goal", "static_obstacles"),
        val_split=0.2,
        seed=3,
        require_all=False,
    )
    assert set(train).isdisjoint(validation)


def test_stratified_split_preserves_train_coverage_without_leakage():
    paths = [
        f"{domain}/agents_{count}/shard_{index}.tar"
        for domain in ("goal_to_goal", "static_obstacles")
        for count in (3, 4, 5)
        for index in range(3)
    ]
    train, validation = split_tb3_shards_by_stratum(
        paths, (3, 4, 5), ("goal_to_goal", "static_obstacles"),
        val_split=0.33, seed=7,
    )
    assert set(train).isdisjoint(validation)
    assert len(train) == 12
    assert len(validation) == 6
    for domain in ("goal_to_goal", "static_obstacles"):
        for count in (3, 4, 5):
            assert any(domain in path and f"agents_{count}" in path for path in train)
