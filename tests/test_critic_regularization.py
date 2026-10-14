"""Numerical and training-path checks for conditional temporal regularization."""
import io
import json
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from accelerate import Accelerator
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ma_vlcm.critic_regularization import (
    temporal_delta_loss, conditional_lipschitz_loss, fixed_tb3_clip_distance,
    tb3_condition_key, conditional_lipschitz_diagnostics,
)
from ma_vlcm.potential import PotentialConfig
from ma_vlcm.train import SequenceWebDataset, parse_args, run_epoch, webdataset_loader


def test_temporal_loss_preserves_positive_and_negative_changes():
    prediction = torch.tensor([.2, .5], requires_grad=True)
    following = torch.tensor([.3, .3], requires_grad=True)
    assert temporal_delta_loss(prediction, following, torch.tensor([0., .4]), torch.tensor([.1, .2])).item() == pytest.approx(0., abs=1e-7)
    loss = temporal_delta_loss(prediction, following, torch.tensor([0., .4]), torch.tensor([-.1, .5]))
    loss.backward()
    assert following.grad[0] > 0 and following.grad[1] < 0


def test_one_sided_bound_allows_flat_outputs_and_bounds_both_signs():
    pred = torch.zeros(3, requires_grad=True)
    nxt = torch.tensor([.05, -.3, .3], requires_grad=True)
    distance = torch.tensor([.1, .1, .1], requires_grad=True)
    loss = conditional_lipschitz_loss(pred, nxt, distance, torch.ones(3, dtype=torch.bool))
    assert loss.item() == pytest.approx(2 * .2 ** 2 / 3)
    loss.backward()
    assert nxt.grad[0] == 0 and nxt.grad[1] < 0 and nxt.grad[2] > 0
    assert distance.grad is None
    assert conditional_lipschitz_loss(pred, pred, distance, torch.ones(3, dtype=torch.bool)).item() == 0


def test_zero_distance_and_empty_valid_pairs_are_finite_and_differentiable():
    pred = torch.tensor([.2], requires_grad=True)
    nxt = torch.tensor([.8], requires_grad=True)
    assert conditional_lipschitz_loss(pred, nxt, torch.zeros(1), torch.ones(1, dtype=torch.bool)).item() == pytest.approx(.36)
    loss = conditional_lipschitz_loss(pred, nxt, torch.zeros(1), torch.zeros(1, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0 and pred.grad.item() == 0


@pytest.mark.parametrize('bad', [-.1, float('nan'), float('inf')])
def test_invalid_distances_fail(bad):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        conditional_lipschitz_loss(torch.zeros(1), torch.ones(1), torch.tensor([bad]), torch.ones(1, dtype=torch.bool))


def frame(x=0., color='black', nodes=3):
    obs = torch.zeros(nodes, 9)
    obs[:, 0] = x
    return {'robot_obs': obs, 'adj': torch.eye(nodes), 'image': Image.new('RGB', (64, 64), color)}


def test_fixed_input_metric_has_declared_units_and_no_team_size_bias():
    assert fixed_tb3_clip_distance([frame()], [frame()]) == 0
    assert fixed_tb3_clip_distance([frame()], [frame(x=.1)]) == pytest.approx(1.)
    assert fixed_tb3_clip_distance([frame(nodes=6)], [frame(x=.1, nodes=6)]) == pytest.approx(1.)
    assert fixed_tb3_clip_distance([frame()] * 16, [frame(x=.1)] * 16) == pytest.approx(1.)
    assert fixed_tb3_clip_distance([frame()], [frame(color='white')]) == pytest.approx(10.)
    # Historical graph entries are not consumed by the model; the full video is.
    assert fixed_tb3_clip_distance([frame(x=100), frame()], [frame(), frame()]) == 0


def test_condition_key_ignores_motion_and_changes_with_goal_layout_and_domain():
    state = {'task_domain': 'goal_to_goal', 'agents': [{'id': 0, 'goal': [1., 0.], 'pos': [0., 0.]}]}
    original = tb3_condition_key(state, 'ep')
    state['agents'][0]['pos'] = [.4, 0.]
    assert tb3_condition_key(state, 'ep') == original
    assert tb3_condition_key(state, 'another-episode') != original
    state['agents'][0]['goal'] = [2., 0.]
    assert tb3_condition_key(state, 'ep') != original
    assert tb3_condition_key(dict(state, task_domain='static_obstacles'), 'ep') != tb3_condition_key(state, 'ep')


def test_diagnostics_detect_target_metric_conflict():
    values = conditional_lipschitz_diagnostics(torch.zeros(2), torch.tensor([.1, .2]), torch.zeros(2), torch.tensor([.3, .3]), torch.tensor([.2, .1]), torch.tensor([True, False]))
    assert values == {'lipschitz_valid_pairs': 1, 'lipschitz_excluded_pairs': 1,
                      'lipschitz_prediction_violations': 0, 'lipschitz_target_violations': 1}


class FakeProcessor:
    def __call__(self, videos, return_tensors):
        return {'pixel_values_videos': torch.zeros(1, len(videos), 3, 8, 8)}


def write_shard(path, *, collision=False, changed_goal=False):
    config = asdict(PotentialConfig())
    with tarfile.open(path, 'w') as archive:
        for step in range(3):
            state = {'task_domain': 'goal_to_goal', 'agents': [{'id': 0, 'pos': [.01 * step, 0.],
                      'goal': [2. if changed_goal and step == 2 else 1., 0.],
                      'dist_to_goal': 1., 'initial_goal_distance': 1.}],
                     'episode_meta': {'episode_id': 'ep', 'step': step, 'done': False}}
            label = {'schema': 'tb3_potential_v1', 'target': -1. if collision and step == 2 else -.1 * step,
                     'collision': collision and step == 2, 'initial_distances': [1.],
                     'potential_config': config, 'episode_success': False}
            image = io.BytesIO()
            Image.new('RGB', (8, 8)).save(image, format='PNG')
            for suffix, payload in [('state.json', json.dumps(state).encode()), ('progress.json', json.dumps(label).encode()), ('overhead.png', image.getvalue())]:
                info = tarfile.TarInfo(f'ep_step{step:05d}.{suffix}')
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


@pytest.mark.parametrize('collision,changed_goal,valid', [(False, False, True), (True, False, False), (False, True, False)])
def test_loader_builds_fixed_condition_pairs_and_preserves_collision_labels(tmp_path, collision, changed_goal, valid):
    path = tmp_path / 'episode.tar'
    write_shard(path, collision=collision, changed_goal=changed_goal)
    dataset = SequenceWebDataset(str(path), 2, 1, 'raw', 'obs', 'mean', 'any',
        vlm_processor=FakeProcessor(), dataset_type='tb3_potential_v1', target_mode='progress',
        target_schema='tb3_potential_v1', potential_config=asdict(PotentialConfig()), vl_backend='vjepa2',
        robot_obs_dim=9, num_robots=None, include_next=True, include_lipschitz=True,
        resize_width=8, resize_height=8)
    sample = next(iter(dataset))
    assert sample['lipschitz_valid'].item() is valid
    assert sample['next_progress'].item() == pytest.approx(-1. if collision else -.2)
    if valid:
        assert sample['lipschitz_distance'].item() == pytest.approx(.1)


def training_args(**overrides):
    return SimpleNamespace(target_mode='progress', temporal_consistency_loss_weight=1.,
        conditional_lipschitz_loss_weight=1., loss_type='mse', success_loss_weight=0.,
        value_output_activation='identity', debug_save_video=False, max_grad_norm=0., **overrides)


class TinyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
    def forward(self, inputs, robot_obs, adj, return_features=False):
        return robot_obs[:, -1, 0, 0] * self.weight


def test_real_epoch_path_supervises_both_endpoints_and_logs_validation_components():
    model = TinyCritic()
    current = torch.zeros(1, 2, 1, 9)
    following = current.clone()
    following[:, -1, 0, 0] = .2
    batch = {'inputs': {}, 'next_inputs': {}, 'robot_obs': current, 'next_robot_obs': following,
             'adj': torch.ones(1, 2, 1, 1), 'next_adj': torch.ones(1, 2, 1, 1),
             'reward': torch.zeros(1), 'done': torch.zeros(1), 'progress': torch.zeros(1),
             'next_progress': torch.tensor([-.2]), 'lipschitz_distance': torch.tensor([.1]),
             'lipschitz_valid': torch.tensor([True])}
    accelerator = Accelerator(cpu=True)
    args = training_args()
    # Endpoint MSE=.08; temporal Smooth-L1=.39; Lipschitz hinge squared=.01.
    expected = .08 + .39 + .01
    loss = run_epoch(model, [batch], None, accelerator, 1, .95, args, train=False)
    assert loss == pytest.approx(expected, abs=1e-6)
    assert run_epoch.last_metrics['val/epoch_progress_mse_loss'] == pytest.approx(.08)
    assert run_epoch.last_metrics['val/lipschitz_target_violation_rate'] == 1.
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    run_epoch(model, [batch], optimizer, accelerator, 1, .95, args, train=True)
    assert model.weight.item() < 1.


def test_lipschitz_alone_requests_next_clip_and_collates_metadata(tmp_path):
    write_shard(tmp_path / 'episode.tar')
    with patch.object(sys, 'argv', ['train', '--vl_backend', 'vjepa2', '--dataset_type', 'tb3_progress_v2',
            '--robot_obs_dim', '9', '--target_mode', 'progress', '--loss_type', 'mse',
            '--conditional_lipschitz_loss_weight', '1', '--clip_len', '2', '--clip_stride', '1',
            '--resize_width', '8', '--resize_height', '8']):
        args = parse_args()
    # Legacy labels are interpreted geometrically by tb3_progress_v2; this
    # test exercises collate and next-input creation with temporal weight zero.
    loader = webdataset_loader(args, str(tmp_path / 'episode.tar'), 1, 0, dataset_type='tb3_progress_v2')
    loader.dataset.vlm_processor = FakeProcessor()
    sample = next(iter(loader))
    assert sample['lipschitz_distance'].shape == (1,)
    assert sample['lipschitz_valid'].dtype == torch.bool
    assert 'next_progress' in sample and 'next_inputs' in sample


def test_repeated_shard_never_pairs_episode_end_with_its_start(tmp_path):
    path = tmp_path / 'episode.tar'
    write_shard(path)
    dataset = SequenceWebDataset(str(path), 2, 1, 'raw', 'obs', 'mean', 'any',
        vlm_processor=FakeProcessor(), dataset_type='tb3_potential_v1', target_mode='progress',
        target_schema='tb3_potential_v1', potential_config=asdict(PotentialConfig()), vl_backend='vjepa2',
        robot_obs_dim=9, num_robots=None, include_next=True, include_lipschitz=True,
        resize_width=8, resize_height=8)
    iterator = iter(dataset)
    for _ in range(3):
        sample = next(iterator)
        assert sample['progress'].item() == pytest.approx(-.1)
        assert sample['next_progress'].item() == pytest.approx(-.2)
