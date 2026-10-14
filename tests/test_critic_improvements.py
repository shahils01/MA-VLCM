"""Accuracy changes: padding-safe query, hazard gradients, signed events/tails."""
import io
import json
import math
import tarfile
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from accelerate import Accelerator
from ma_vlcm.model import ModelConfig, MultimodalValueModel
from ma_vlcm.critic_improvements import (potential_component_target, potential_auxiliary_loss,
    event_weights, weighted_mean, change_metrics)
from ma_vlcm.potential import PotentialConfig
from ma_vlcm.distributional_value import categorical_value
from ma_vlcm.train import SequenceWebDataset, run_epoch


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(20, 8)
        self.config = SimpleNamespace(image_token_id=17, video_token_id=18)
    def forward(self, inputs_embeds, attention_mask=None, **kwargs):
        hidden = inputs_embeds * attention_mask.unsqueeze(-1) if attention_mask is not None else inputs_embeds
        return SimpleNamespace(hidden_states=(hidden.cumsum(dim=1),))


class TinyBackbone(torch.nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.model = TinyLM()
        self.tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda name: 19)
    def get_input_embeddings(self): return self.model.embedding
    def _move_inputs_to_device(self, inputs): return inputs


class TinyGNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(kwargs['in_channels'], kwargs['out_channels'])
    def forward(self, nodes, edges): return self.linear(nodes)


def tiny_model(**kwargs):
    cfg = ModelConfig(d_model=4, robot_obs_dim=9, num_robots=3, value_distribution_bins=21, **kwargs)
    with patch('ma_vlcm.model.VisionLanguageBackbone', TinyBackbone), patch('ma_vlcm.gat.GNN_Model', TinyGNN):
        return MultimodalValueModel(cfg, torch.device('cpu'))


def test_query_and_attention_pool_ignore_padding_and_have_trainable_gradients():
    torch.manual_seed(4)
    model = tiny_model(critic_readout='query', agent_pooling='attention_max', potential_aux_heads=True)
    obs = torch.randn(1, 2, 3, 9); adj = torch.eye(3).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
    a = {'input_ids': torch.tensor([[1, 19, 2]]), 'attention_mask': torch.ones(1, 3, dtype=torch.long)}
    expected = model(a, obs, adj, return_features=True)
    for ids, mask in [([1, 19, 2, 0, 0], [1, 1, 1, 0, 0]), ([0, 0, 1, 19, 2], [0, 0, 1, 1, 1])]:
        padded_obs = torch.cat((obs, torch.full((1, 2, 2, 9), 999.)), dim=2)
        padded_adj = torch.nn.functional.pad(adj, (0, 2, 0, 2))
        actual = model({'input_ids': torch.tensor([ids]), 'attention_mask': torch.tensor([mask])}, padded_obs, padded_adj, return_features=True)
        assert torch.allclose(actual['value'], expected['value'], atol=1e-6)
        assert torch.allclose(actual['potential_components'], expected['potential_components'], atol=1e-6)
    target = torch.tensor([[.3, .9, .8, 1.]])
    aux, _ = potential_auxiliary_loss(expected, target)
    (expected['value'].sum() + aux).backward()
    for name in ['critic_query', 'agent_attention.weight', 'agent_pool_fusion.weight', 'potential_head.weight']:
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and gradient.abs().sum() > 0
    clone = tiny_model(critic_readout='query', agent_pooling='attention_max', potential_aux_heads=True)
    clone.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(clone(a, obs, adj), model(a, obs, adj))


def test_query_rejects_visual_placeholder_and_empty_prompt():
    model = tiny_model(critic_readout='query')
    obs = torch.zeros(1, 2, 3, 9); adj = torch.eye(3).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
    with pytest.raises(ValueError, match='visual token'):
        model({'input_ids': torch.tensor([[1, 18]]), 'attention_mask': torch.ones(1, 2)}, obs, adj)
    with pytest.raises(ValueError, match='nonempty'):
        model({'input_ids': torch.tensor([[0, 0]]), 'attention_mask': torch.zeros(1, 2)}, obs, adj)
    legacy = tiny_model()
    assert not any(name.startswith(('critic_query', 'agent_attention', 'agent_pool_fusion', 'potential_head')) for name in legacy.state_dict())


def test_component_targets_validate_ranges_and_do_not_change_scalar_targets():
    data = {'goal_cost': 3., 'agent_risk': .4, 'obstacle_risk': .8, 'collision': True, 'target': -1.}
    target = potential_component_target(data)
    assert target.tolist() == pytest.approx([math.log(4), .4, .8, 1])
    assert data['target'] == -1
    for bad in [dict(data, goal_cost=-1), dict(data, agent_risk=1.1), dict(data, goal_cost=float('nan')), {}]:
        with pytest.raises(ValueError): potential_component_target(bad)


@pytest.mark.parametrize('collision', [0., 1.])
def test_auxiliary_bce_is_finite_in_single_class_batches(collision):
    predicted = torch.full((3, 3), .5, requires_grad=True)
    logits = torch.zeros(3, requires_grad=True)
    loss, components = potential_auxiliary_loss({'potential_components': predicted, 'collision_logit': logits}, torch.tensor([[.2, 0., .8, collision]] * 3))
    assert torch.isfinite(loss) and len(components) == 4
    loss.backward()
    assert (logits.grad < 0).all() if collision else (logits.grad > 0).all()


def test_event_weights_emphasize_both_signs_and_collision_plateaus():
    target = torch.tensor([.2, .2, -1., .2]); following = torch.tensor([-.2, .6, -1., .21])
    assert event_weights(target, following, boost=3).tolist() == [3., 3., 3., 1.]
    assert weighted_mean(torch.tensor([1., 2., 3., 4.]), torch.tensor([3., 3., 3., 1.])).item() == pytest.approx(2.2)
    metrics = change_metrics(torch.zeros(4), torch.tensor([-.1, .1, -.1, .0]), target, following)
    assert metrics['drop_direction_hits'] == metrics['drop_pairs'] == 1
    assert metrics['recovery_direction_hits'] == metrics['recovery_pairs'] == 1
    assert metrics['false_drop_pairs'] == 1


class FakeProcessor:
    def __call__(self, videos, return_tensors):
        return {'pixel_values_videos': torch.zeros(1, len(videos), 3, 8, 8)}


def write_episode(path):
    with tarfile.open(path, 'w') as archive:
        for step in range(9):
            collision = step == 8
            state = {'task_domain': 'goal_to_goal', 'agents': [{'id': 0, 'pos': [.01 * step, 0.], 'goal': [1., 0.], 'dist_to_goal': 1., 'initial_goal_distance': 1.}], 'episode_meta': {'episode_id': 'ep', 'step': step, 'done': collision}}
            label = {'schema': 'tb3_potential_v1', 'target': -1. if collision else .05 * step, 'goal_cost': 1., 'agent_risk': float(collision), 'obstacle_risk': 0., 'collision': collision, 'initial_distances': [1.], 'potential_config': asdict(PotentialConfig()), 'episode_success': False}
            image = io.BytesIO(); Image.new('RGB', (8, 8)).save(image, format='PNG')
            for suffix, payload in [('state.json', json.dumps(state).encode()), ('progress.json', json.dumps(label).encode()), ('overhead.png', image.getvalue())]:
                info = tarfile.TarInfo(f'ep_step{step:05d}.{suffix}'); info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


def dataset(path, **kwargs):
    return SequenceWebDataset(str(path), 3, 1, 'raw', 'obs', 'mean', 'any', vlm_processor=FakeProcessor(), dataset_type='tb3_potential_v1', target_mode='progress', target_schema='tb3_potential_v1', potential_config=asdict(PotentialConfig()), vl_backend='vjepa2', robot_obs_dim=9, num_robots=None, resize_width=8, resize_height=8, **kwargs)


def test_multigap_pairs_supervise_terminal_collision_and_do_not_wrap(tmp_path):
    path = tmp_path/'episode.tar'; write_episode(path)
    ds = dataset(path, include_next=True, include_lipschitz=True, include_potential_components=True, temporal_pair_offsets=(1,2,4), shuffle_shards=True)
    with patch('ma_vlcm.train.random.choice', lambda values: max(values)):
        iterator = iter(ds); samples = [next(iterator) for _ in range(6)]
    assert {sample['pair_offset'].item() for sample in samples} == {1,2,4}
    assert samples[-1]['next_progress'].item() == -1
    assert not samples[-1]['lipschitz_valid'].item()
    assert samples[-1]['next_potential_components'][3].item() == 1
    assert all(sample['next_robot_obs'][-1, 0, 0] > sample['robot_obs'][-1, 0, 0] for sample in samples)


def test_validation_and_inference_disable_event_filter_and_extra_offsets(tmp_path):
    path = tmp_path/'episode.tar'; write_episode(path)
    ds = dataset(path, include_next=False, temporal_pair_offsets=(1,2,4), event_sampling_boost=100, shuffle_shards=False)
    iterator = iter(ds); samples = [next(iterator) for _ in range(7)]
    assert samples[-1]['progress'].item() == -1
    assert all('next_inputs' not in sample and 'potential_components' not in sample for sample in samples)
    assert ds.event_sampling_boost == 1 and ds.temporal_pair_offsets == (1,)


class TinyAuxCritic(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.head = torch.nn.Linear(1,21); self.aux = torch.nn.Linear(1,4)
    def forward(self, inputs, robot_obs, adj, return_features=False):
        features = robot_obs[:, -1, 0, :1]; logits = self.head(features); raw = self.aux(features)
        return {'value': categorical_value(logits), 'value_logits': logits, 'potential_components': torch.cat((torch.nn.functional.softplus(raw[:,:1]), raw[:,1:3].sigmoid()), -1), 'collision_logit': raw[:,3]}


def test_full_epoch_path_trains_aux_heads_and_logs_event_metrics():
    model = TinyAuxCritic(); current = torch.zeros(2,2,1,9); following = current.clone(); following[:, -1,0,0] = .1
    batch = {'inputs': {}, 'next_inputs': {}, 'robot_obs': current, 'next_robot_obs': following, 'adj': torch.ones(2,2,1,1), 'next_adj': torch.ones(2,2,1,1), 'reward': torch.zeros(2), 'done': torch.zeros(2), 'progress': torch.tensor([.3, .3]), 'next_progress': torch.tensor([-1., .7]), 'lipschitz_distance': torch.ones(2), 'lipschitz_valid': torch.tensor([False,True]), 'potential_components': torch.tensor([[.3,0.,0.,0.]] * 2), 'next_potential_components': torch.tensor([[.3,1.,0.,1.], [.2,0.,0.,0.]])}
    args = SimpleNamespace(target_mode='progress', temporal_consistency_loss_weight=1., conditional_lipschitz_loss_weight=1., loss_type='categorical_ce', success_loss_weight=0., value_output_activation='identity', value_support_min=-1., value_support_max=1., debug_save_video=False, max_grad_norm=0., potential_aux_loss_weight=.5, event_loss_boost=3, event_delta_threshold=.03)
    old = model.aux.weight.detach().clone(); accelerator = Accelerator(cpu=True)
    loss = run_epoch(model, [batch], torch.optim.SGD(model.parameters(), lr=.1), accelerator, 1,.95,args,train=True)
    assert math.isfinite(loss) and not torch.equal(model.aux.weight, old)
    assert 'train/epoch_aux_collision_loss' in run_epoch.last_metrics
    assert run_epoch.last_metrics['train/drop_pairs'] == 1 and run_epoch.last_metrics['train/recovery_pairs'] == 1


class FakeQwenProcessor:
    def __init__(self): self.calls = []
    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return {'input_ids': torch.tensor([[1,19,2]]), 'attention_mask': torch.ones(1,3,dtype=torch.long), 'video_grid_thw': torch.tensor([[2,2,2]]), 'pixel_values_videos': torch.zeros(8,1536)}


@pytest.mark.parametrize('query', [False, True])
def test_qwen_query_loader_appends_assistant_prompt_and_keeps_all_frames(tmp_path, query):
    path = tmp_path/'episode.tar'; write_episode(path)
    processor = FakeQwenProcessor()
    ds = SequenceWebDataset(str(path), 3,1,'raw','obs','mean','any',vlm_processor=processor,
        dataset_type='tb3_potential_v1',target_mode='progress',target_schema='tb3_potential_v1',
        potential_config=asdict(PotentialConfig()),vl_backend='qwen3_vl',robot_obs_dim=9,
        num_robots=None,resize_width=224,resize_height=224,qwen_critic_query=query)
    next(iter(ds))
    assert processor.calls[0]['add_generation_prompt'] is query
    assert processor.calls[0]['do_sample_frames'] is False
    assert processor.calls[0]['video_metadata'][0]['total_num_frames'] == 3
