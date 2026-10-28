"""Masked graph attention for small, variable-size robot teams (pure PyTorch)."""

import torch
from torch import nn
from torch.nn import functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, width, heads=4, dropout=0.1):
        super().__init__()
        if width % heads:
            raise ValueError("Graph width must be divisible by the number of attention heads")
        self.heads = heads
        self.head_width = width // heads
        self.source = nn.Linear(width, width, bias=False)
        self.neighbor = nn.Linear(width, width, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, self.head_width))
        nn.init.xavier_uniform_(self.attention)
        self.output = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, adjacency, robot_mask):
        # Leading dimensions can be [batch, time]; the last dimensions are [N, D].
        n = nodes.shape[-2]
        shape = (*nodes.shape[:-1], self.heads, self.head_width)
        source = self.source(nodes).reshape(shape)
        neighbor = self.neighbor(nodes).reshape(shape)
        pairs = F.leaky_relu(source.unsqueeze(-3) + neighbor.unsqueeze(-4), 0.2)
        logits = (pairs * self.attention).sum(-1).movedim(-1, -3)
        valid_pair = robot_mask.unsqueeze(-1) & robot_mask.unsqueeze(-2)
        allowed = ((adjacency > 0) | torch.eye(n, device=nodes.device, dtype=torch.bool)) & valid_pair
        weights = logits.masked_fill(~allowed.unsqueeze(-3), -1e4).softmax(-1)
        # Padded rows have no neighbors; explicitly zero them rather than softmax(-inf).
        weights = weights * allowed.unsqueeze(-3)
        weights = self.dropout(weights)
        values = neighbor.movedim(-2, -3)
        messages = torch.matmul(weights, values).movedim(-3, -2).flatten(-2)
        updated = self.norm(nodes + self.dropout(self.output(messages)))
        return updated * robot_mask.unsqueeze(-1)


class GraphAttentionEncoder(nn.Module):
    def __init__(self, input_dim=10, width=256, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, width), nn.GELU())
        self.layers = nn.ModuleList([
            GraphAttentionLayer(width, heads, dropout) for _ in range(layers)
        ])
        self.pool = nn.Linear(2 * width + 1, width)

    def forward(self, observations, adjacency, robot_mask):
        nodes = self.input(observations) * robot_mask.unsqueeze(-1)
        for layer in self.layers:
            nodes = layer(nodes, adjacency, robot_mask)
        count = robot_mask.sum(-1, keepdim=True).clamp_min(1)
        mean = nodes.sum(-2) / count
        maximum = nodes.masked_fill(~robot_mask.unsqueeze(-1), -1e4).amax(-2)
        valid_frame = robot_mask.any(-1, keepdim=True)
        maximum = maximum * valid_frame
        team = torch.cat((mean, maximum, count.float().log()), dim=-1)
        return self.pool(team) * valid_frame
