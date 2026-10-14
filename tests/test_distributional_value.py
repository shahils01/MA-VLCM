import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from accelerate import Accelerator

from ma_vlcm.distributional_value import value_support, project_scalar, categorical_value, categorical_value_loss
from ma_vlcm.model import MultimodalValueModel
from ma_vlcm.train import run_epoch, parse_args


def test_two_hot_targets_recover_signed_scalar_and_endpoints():
    support = value_support(21)
    targets = torch.tensor([-1., -.35, 0., .27, 1.])
    projected = project_scalar(targets, support)
    assert torch.allclose(projected.sum(-1), torch.ones(5))
    assert torch.allclose((projected * support).sum(-1), targets, atol=1e-6)
    assert projected[1, 6].item() == pytest.approx(.5, abs=1e-6)
    assert projected[1, 7].item() == pytest.approx(.5, abs=1e-6)
    assert projected[0, 0] == 1 and projected[-1, -1] == 1


@pytest.mark.parametrize('value', [-1.1, 1.1, float('nan')])
def test_invalid_labels_rejected(value):
    with pytest.raises(ValueError):
        project_scalar(torch.tensor([value]), value_support(21))


def test_model_value_expectation_and_saved_head_contract():
    model = MultimodalValueModel.__new__(MultimodalValueModel)
    torch.nn.Module.__init__(model)
    model.cfg = SimpleNamespace(value_distribution_bins=21, value_support_min=-1., value_support_max=1.)
    model.value_head = torch.nn.Linear(3, 21)
    features = torch.randn(4, 3)
    value, logits = model._value_prediction(features)
    assert value.shape == (4,) and logits.shape == (4, 21)
    assert torch.allclose(value, categorical_value(logits))
    assert torch.all(value >= -1) and torch.all(value <= 1)
    buffer = io.BytesIO()
    torch.save({'state': model.state_dict(), 'args': vars(model.cfg)}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    clone = MultimodalValueModel.__new__(MultimodalValueModel)
    torch.nn.Module.__init__(clone)
    clone.cfg = SimpleNamespace(**saved['args'])
    clone.value_head = torch.nn.Linear(3, 21)
    clone.load_state_dict(saved['state'])
    assert torch.equal(clone._value_prediction(features)[0], value)
    clone.cfg.value_distribution_bins = 0
    clone.value_head = torch.nn.Linear(3, 1)
    legacy, logits = clone._value_prediction(features)
    assert legacy.shape == (4,) and logits is None


class TinyCategorical(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(1, 21)
    def forward(self, inputs, robot_obs, adj, return_features=False):
        logits = self.head(robot_obs[:, -1, 0, :1])
        value = categorical_value(logits)
        return {'value': value, 'value_logits': logits} if return_features else value


def test_epoch_ce_and_regularizers_supervise_both_endpoints():
    model = TinyCategorical()
    current = torch.zeros(1, 2, 1, 9)
    following = current.clone(); following[:, -1, 0, 0] = .2
    batch = {'inputs': {}, 'next_inputs': {}, 'robot_obs': current, 'next_robot_obs': following,
        'adj': torch.ones(1, 2, 1, 1), 'next_adj': torch.ones(1, 2, 1, 1),
        'reward': torch.zeros(1), 'done': torch.zeros(1), 'progress': torch.tensor([.2]),
        'next_progress': torch.tensor([-.2]), 'lipschitz_distance': torch.tensor([.01]),
        'lipschitz_valid': torch.tensor([True])}
    args = SimpleNamespace(target_mode='progress', temporal_consistency_loss_weight=1.,
        conditional_lipschitz_loss_weight=1., loss_type='categorical_ce', success_loss_weight=0.,
        value_output_activation='identity', value_support_min=-1., value_support_max=1.,
        debug_save_video=False, max_grad_norm=0.)
    from ma_vlcm.critic_regularization import temporal_delta_loss, conditional_lipschitz_loss
    a = model({}, current, batch['adj'], True); b = model({}, following, batch['next_adj'], True)
    ce = .5 * (categorical_value_loss(a['value_logits'], batch['progress']) + categorical_value_loss(b['value_logits'], batch['next_progress']))
    temporal = temporal_delta_loss(a['value'], b['value'], batch['progress'], batch['next_progress'])
    lip = conditional_lipschitz_loss(a['value'], b['value'], batch['lipschitz_distance'], batch['lipschitz_valid'])
    accelerator = Accelerator(cpu=True)
    actual = run_epoch(model, [batch], None, accelerator, 1, .95, args, train=False)
    assert actual == pytest.approx((ce + temporal + lip).item(), abs=1e-6)
    assert run_epoch.last_metrics['val/epoch_categorical_ce_loss'] == pytest.approx(ce.item())
    previous = model.head.weight.detach().clone()
    run_epoch(model, [batch], torch.optim.SGD(model.parameters(), lr=.1), accelerator, 1, .95, args, train=True)
    assert not torch.equal(model.head.weight, previous)


def test_cli_requires_signed_categorical_contract():
    args = ['train', '--loss_type', 'categorical_ce', '--value_distribution_bins', '21',
        '--value_output_activation', 'identity', '--target_schema', 'tb3_potential_v1',
        '--target_mode', 'progress', '--dataset_type', 'tb3_potential_v1', '--robot_obs_dim', '9',
        '--progress_distance_mode', 'euclidean', '--temporal_consistency_loss_weight', '1',
        '--conditional_lipschitz_loss_weight', '1']
    with patch('sys.argv', args):
        parsed = parse_args()
    assert parsed.value_distribution_bins == 21
    with patch('sys.argv', args + ['--value_output_activation', 'tanh']), pytest.raises(SystemExit):
        parse_args()
