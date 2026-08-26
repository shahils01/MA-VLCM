"""Balanced dataset-source utilities."""

from itertools import cycle, islice
import random
import re


_DOMAIN_ALIASES = (
    ("tb3_isaac_avoid_obstacles_vlcm", "static_obstacles"),
    ("avoid_obstacles", "static_obstacles"),
    ("tb3_isaac_vlcm", "goal_to_goal"),
)


def tb3_shard_stratum(path, agent_counts, task_domains):
    """Infer a TB3 shard's task domain and cardinality from its path.

    Hugging Face cache paths retain the repository name.  The aliases keep the
    legacy ``tb3-isaac-vlcm`` repository usable even though its shard paths do
    not contain an explicit ``goal_to_goal`` directory.
    """

    normalized = str(path).lower().replace("-", "_")
    domains = [str(value).strip().lower() for value in task_domains]
    counts = [int(value) for value in agent_counts]
    domain = next((name for name in domains if name in normalized), None)
    if domain is None:
        for marker, candidate in _DOMAIN_ALIASES:
            if marker in normalized and candidate in domains:
                domain = candidate
                break

    count = next(
        (
            value
            for value in counts
            if re.search(rf"(?:^|[/_])agents[_-]?0*{value}(?:[/_]|$)", normalized)
            or re.search(rf"(?:^|[/_])0*{value}[_-]?agents(?:[/_]|$)", normalized)
        ),
        None,
    )
    return domain, count


def balanced_tb3_shards(
    shards, agent_counts, task_domains, balance_mode="domain_cardinality"
):
    """Repeat paths so requested TB3 source groups have equal probability."""

    paths = [str(path) for path in shards]
    counts = [int(value) for value in agent_counts]
    domains = [str(value).strip().lower() for value in task_domains]
    if balance_mode == "domain":
        strata = {domain: [] for domain in domains}
    elif balance_mode == "domain_cardinality":
        strata = {(domain, count): [] for domain in domains for count in counts}
    else:
        raise ValueError(f"unsupported TB3 balance mode: {balance_mode}")
    for path in paths:
        domain, count = tb3_shard_stratum(path, counts, domains)
        if domain is not None and count is not None:
            key = domain if balance_mode == "domain" else (domain, count)
            strata[key].append(path)

    missing = [key for key, values in strata.items() if not values]
    if missing:
        raise ValueError(
            "balanced TB3 sampling could not identify shards for strata "
            f"{missing}; organize paths by domain and agents_N or disable balancing"
        )
    target = max(len(values) for values in strata.values())
    balanced = []
    for key in sorted(strata):
        balanced.extend(islice(cycle(sorted(strata[key])), target))
    return balanced


def split_tb3_shards_by_stratum(
    shards,
    agent_counts,
    task_domains,
    val_split=0.2,
    seed=42,
    require_all=True,
):
    """Split each domain/cardinality stratum without cross-split leakage."""

    if not 0.0 <= float(val_split) < 1.0:
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")
    paths = [str(path) for path in shards]
    counts = [int(value) for value in agent_counts]
    domains = [str(value).strip().lower() for value in task_domains]
    strata = {(domain, count): [] for domain in domains for count in counts}
    for path in paths:
        domain, count = tb3_shard_stratum(path, counts, domains)
        if domain is not None and count is not None:
            strata[(domain, count)].append(path)
    missing = [key for key, values in strata.items() if not values]
    if missing and require_all:
        raise ValueError(f"cannot stratify TB3 shards; missing strata {missing}")
    strata = {key: values for key, values in strata.items() if values}

    train, validation = [], []
    rng = random.Random(seed)
    for key in sorted(strata):
        values = sorted(strata[key])
        rng.shuffle(values)
        if val_split <= 0.0 or len(values) == 1:
            train.extend(values)
            continue
        validation_count = min(
            len(values) - 1, max(1, int(round(len(values) * float(val_split))))
        )
        validation.extend(values[:validation_count])
        train.extend(values[validation_count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation or None
