import pytest
import torch

from ma_vlcm.reward_refinement import (
    TrajectoryOutcome,
    bradley_terry_loss,
    preferred_trajectory,
    temporal_order_accuracy,
)
from ma_vlcm.train import _parse_tb3_lab_state


def test_tb3_progress_v2_row_appends_initial_goal_distance():
    state = {
        "agents": [
            {
                "id": 0,
                "pos": [1.0, 2.0],
                "yaw": 0.0,
                "vel": [0.1, -0.2],
                "dist_to_goal": 0.7,
                "min_neighbor_dist": 0.4,
            }
        ]
    }
    row = _parse_tb3_lab_state(
        state, num_robots=1, robot_obs_dim=9, initial_goal_distances=[2.5]
    )
    assert row.shape == (1, 9)
    assert row[0, 8].item() == pytest.approx(2.5)


def test_legacy_eight_dimensional_rows_remain_compatible():
    row = _parse_tb3_lab_state(
        {"agents": [{"id": 0, "dist_to_goal": 1.0}]},
        num_robots=1,
        robot_obs_dim=8,
    )
    assert row.shape == (1, 8)


def test_trajectory_order_prefers_success_progress_then_speed():
    success = TrajectoryOutcome(True, 0.8, 20.0)
    failure = TrajectoryOutcome(False, 1.0, 1.0)
    assert preferred_trajectory(success, failure) == 1
    assert preferred_trajectory(
        TrajectoryOutcome(True, 0.9, 30.0), success
    ) == 1
    assert preferred_trajectory(
        TrajectoryOutcome(True, 0.8, 10.0), success
    ) == 1


def test_temporal_ordering_and_bradley_terry_losses():
    assert temporal_order_accuracy(torch.tensor([[0.1, 0.2, 0.9]])).item() == 1.0
    good = bradley_terry_loss(torch.tensor([2.0]), torch.tensor([-1.0]))
    bad = bradley_terry_loss(torch.tensor([-1.0]), torch.tensor([2.0]))
    assert good < bad
