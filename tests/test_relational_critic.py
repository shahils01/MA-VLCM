"""Causality, agent grounding, padding and checkpoint compatibility contracts."""
import torch
import pytest
from ma_vlcm.relational_critic import CausalAgentHistory, AgentVisualGrounding, inject_agent_tokens
from test_critic_improvements import tiny_model


def test_history_cannot_see_future_frames_and_uses_earlier_observations():
    torch.manual_seed(7)
    model = CausalAgentHistory(8, 2, 2, 8, 0).eval()
    x = torch.randn(2, 8, 3, 8, requires_grad=True)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    y = model(x, mask)
    changed = x.detach().clone(); changed[:, 4:] += torch.randn_like(changed[:, 4:]) * 10
    assert torch.allclose(y[:, :4], model(changed, mask)[:, :4], atol=1e-6)
    y[:, -1, :2].square().sum().backward()
    assert x.grad[:, :7, :2].abs().sum() > 0
    assert torch.isfinite(x.grad).all()
    assert y[1, :, 2].eq(0).all()


def test_per_agent_injection_preserves_individual_features_and_rejects_missing_tokens():
    emb = torch.zeros(2, 6, 4, requires_grad=True)
    ids = torch.tensor([[1, 9, 9, 2, 0, 0], [1, 9, 9, 9, 2, 0]])
    agents = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    actual = inject_agent_tokens(emb, ids, 9, agents, mask)
    assert torch.equal(actual[0, 1:3], agents[0, :2])
    actual.square().sum().backward()
    assert agents.grad[0, :2].abs().sum() > 0 and agents.grad[0, 2].eq(0).all()
    with pytest.raises(ValueError, match='placeholder count'):
        inject_agent_tokens(emb, torch.zeros_like(ids), 9, agents, mask)


def test_visual_grounding_ignores_visual_and_agent_padding_and_has_gradients():
    torch.manual_seed(5)
    model = AgentVisualGrounding(8, 4, 2, 0).eval()
    agents = torch.randn(1, 3, 4, requires_grad=True)
    hidden = torch.randn(1, 6, 8, requires_grad=True)
    visual = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    nodes = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    context = torch.randn(1, 8, requires_grad=True)
    expected = model(agents, hidden, visual, context, nodes)
    changed_hidden = hidden.detach().clone();changed_hidden[:, 3:] = 999
    changed_agents = agents.detach().clone();changed_agents[:, 2] = 999
    actual = model(changed_agents, changed_hidden, visual, context, nodes)
    assert torch.allclose(expected, actual, atol=1e-6)
    expected.square().sum().backward()
    for x in (agents, hidden, context):
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='visual tokens'):
        model(agents, hidden, torch.zeros_like(visual), context, nodes)


def test_full_relational_critic_backward_history_padding_and_strict_reload():
    torch.manual_seed(11)
    options=dict(critic_readout='query',agent_pooling='attention_max',relational_critic=True,
                 temporal_dropout=0.,potential_aux_heads=True,trajectory_preference_head=True)
    model=tiny_model(**options).eval()
    obs=torch.randn(1,8,3,9,requires_grad=True)
    adj=torch.eye(3).view(1,1,3,3).repeat(1,8,1,1)
    inputs={'input_ids':torch.tensor([[18,18,1,19,19,19,2]]),
            'attention_mask':torch.ones(1,7,dtype=torch.long)}
    result=model(inputs,obs,adj,return_features=True)
    padded_obs=torch.nn.functional.pad(obs,(0,0,0,2),value=999)
    padded_adj=torch.nn.functional.pad(adj,(0,2,0,2))
    actual=model(inputs,padded_obs,padded_adj,return_features=True)
    assert torch.allclose(result['value'],actual['value'],atol=1e-6)
    loss=result['value'].square().sum()+result['potential_components'].square().sum()+result['preference_score'].square().sum()
    loss.backward()
    assert obs.grad[:,:7].abs().sum()>0
    for prefix in ('agent_history','agent_visual_grounding','critic_query','obs_to_lm'):
        grads=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0
    clone=tiny_model(**options).eval();clone.load_state_dict(model.state_dict(),strict=True)
    assert torch.equal(model(inputs,obs.detach(),adj),clone(inputs,obs.detach(),adj))


def test_legacy_configuration_does_not_add_relational_weights():
    model=tiny_model()
    assert not any(n.startswith(('agent_history.','agent_visual_grounding.')) for n in model.state_dict())


@pytest.mark.parametrize("pad_agents", [False, True])
def test_relational_bfloat16_autocast_forward_and_backward(pad_agents):
    torch.manual_seed(22)
    model=tiny_model(critic_readout="query",agent_pooling="attention_max",
        relational_critic=True,temporal_dropout=0.,potential_aux_heads=True,
        trajectory_preference_head=True).eval()
    obs=torch.randn(2,8,3,9)
    adj=torch.eye(3).repeat(2,8,1,1)
    if pad_agents:
        obs=torch.nn.functional.pad(obs,(0,0,0,2),value=999.)
        adj=torch.nn.functional.pad(adj,(0,2,0,2))
    inputs={"input_ids":torch.tensor([[18,18,1,19,19,19,2]]*2),
            "attention_mask":torch.ones(2,7,dtype=torch.long)}
    with torch.autocast("cpu",dtype=torch.bfloat16):
        result=model(inputs,obs,adj,return_features=True)
        loss=result["value_logits"].float().square().mean()+result["preference_score"].float().square().mean()+result["potential_components"].float().square().mean()
    assert torch.isfinite(result["value"]).all() and torch.isfinite(loss)
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.agent_attention.weight.grad.abs().sum()>0
