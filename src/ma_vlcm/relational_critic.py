"""Causal agent histories and instruction-conditioned visual grounding."""
import torch
from torch import nn


class CausalAgentHistory(nn.Module):
    def __init__(self, dim, heads, layers, frames, dropout):
        super().__init__()
        self.position = nn.Parameter(torch.randn(frames, dim) * .02)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout,
                                          batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, features, node_mask):
        b, t, n, d = features.shape
        if t > self.position.shape[0]:
            raise ValueError("agent history exceeds configured causal window")
        features = features.masked_fill(~node_mask[:, None, :, None], 0)
        x = features.permute(0, 2, 1, 3).reshape(b * n, t, d)
        x = x + self.position[:t]
        causal = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        x = self.encoder(x, mask=causal)
        return x.reshape(b, n, t, d).permute(0, 2, 1, 3).masked_fill(
            ~node_mask[:, None, :, None], 0)


class AgentVisualGrounding(nn.Module):
    def __init__(self, hidden, dim, heads, dropout):
        super().__init__()
        self.visual_projection = nn.Linear(hidden, dim)
        self.context_film = nn.Linear(hidden, 2 * dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.agent_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.agent_norm = nn.LayerNorm(dim)

    def forward(self, agents, hidden, visual_mask, context, node_mask):
        if not visual_mask.any(dim=1).all():
            raise ValueError("relational critic requires visual tokens in every sample")
        rows = [hidden[i, visual_mask[i]] for i in range(hidden.shape[0])]
        lengths = torch.tensor([row.shape[0] for row in rows], device=hidden.device)
        visual = nn.utils.rnn.pad_sequence(rows, batch_first=True)
        padding = torch.arange(visual.shape[1], device=hidden.device)[None] >= lengths[:, None]
        visual = self.visual_projection(visual.to(self.visual_projection.weight.dtype))
        scale, bias = self.context_film(context).chunk(2, dim=-1)
        queries = agents * (1 + scale.tanh()[:, None]) + bias[:, None]
        grounded, _ = self.cross_attention(queries, visual, visual, key_padding_mask=padding, need_weights=False)
        grounded = self.cross_norm(queries + grounded)
        related, _ = self.agent_attention(grounded, grounded, grounded, key_padding_mask=~node_mask, need_weights=False)
        return self.agent_norm(grounded + related).masked_fill(~node_mask[:, :, None], 0)


def inject_agent_tokens(inputs_embeds, input_ids, token_id, agent_tokens, node_mask):
    """Assign one observation placeholder to each real agent without pooling."""
    rows = []
    for i in range(input_ids.shape[0]):
        positions = input_ids[i].eq(token_id).nonzero(as_tuple=False).flatten()
        agents = agent_tokens[i, node_mask[i]]
        if positions.numel() != agents.shape[0]:
            raise ValueError("observation placeholder count differs from real agent count")
        rows.append(inputs_embeds[i].index_copy(0, positions, agents.to(inputs_embeds.dtype)))
    return torch.stack(rows)
