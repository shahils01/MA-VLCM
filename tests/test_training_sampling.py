import pytest
import torch
from accelerate.utils.operations import concatenate


from ma_vlcm.train import (
    _compute_epoch_batch_counts,
    _episode_id_to_int64,
    split_shards,
)


def test_epoch_batch_counts_do_not_divide_samples_by_accumulation():
    microbatches, optimizer_steps = _compute_epoch_batch_counts(
        samples_per_epoch=5000,
        batch_size=1,
        grad_accum_steps=16,
        num_processes=1,
    )
    assert microbatches == 5000
    assert optimizer_steps == 313

    microbatches, optimizer_steps = _compute_epoch_batch_counts(
        samples_per_epoch=5000,
        batch_size=2,
        grad_accum_steps=8,
        num_processes=1,
    )
    assert microbatches == 2500
    assert optimizer_steps == 313


def test_split_shards_is_deterministic_and_disjoint():
    shards = [f"episode_{index:04d}.tar" for index in range(10)]
    train_a, val_a = split_shards(shards, val_split=0.2, seed=7)
    train_b, val_b = split_shards(shards, val_split=0.2, seed=7)

    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) == 8

    assert len(val_a) == 2
    assert set(train_a).isdisjoint(val_a)
    assert set(train_a + val_a) == set(shards)

def test_split_shards_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        split_shards(["episode.tar"], val_split=1.0)



def test_episode_ids_are_deterministic_accelerate_safe_tensors():
    episode_a = _episode_id_to_int64("episode_0001")
    episode_b = _episode_id_to_int64("episode_0002")

    assert episode_a == _episode_id_to_int64("episode_0001")
    assert episode_a != episode_b
    assert 0 <= episode_a < 2**63

    batches = [
        {"episode_id": torch.tensor([episode_a], dtype=torch.long)},
        {"episode_id": torch.tensor([episode_b], dtype=torch.long)},
    ]
    combined = concatenate(batches)
