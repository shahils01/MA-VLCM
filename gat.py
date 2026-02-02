import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, ones
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_add
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data


class GNN_Model(MessagePassing):
    def __init__(self, args, in_channels, hid_channels, out_channels, num_agents):
        super().__init__(node_dim=0, aggr='add')

        self.args = args
        self.num_nodes = num_agents
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = int(getattr(self.args, "num_heads", 1))
        self.hid_channels = hid_channels
        self.hid_channels_ = self.heads * self.hid_channels
        self.K = int(getattr(self.args, "iterations", 1))
        self.num_layers = int(getattr(self.args, "num_layers", 1))
        self.add_dropout = bool(getattr(self.args, "add_dropout", False))
        self.dropout_p = float(getattr(self.args, "dropout", 0.0))
        self.lambd_gnn = float(getattr(self.args, "lambd_gnn", 1.0))
                
        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):
        self.dropout = nn.Dropout(self.dropout_p)
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()

        self.dense_lins = nn.ModuleList()
        self.decay_weights = [
            np.log((self.lambd_gnn / (k + 1)) + (1 + 1e-6))
            for k in range(max(self.K, 1))
        ]

        # Shared node encoder for all robots.
        enc_layers = [
            Linear(self.in_channels, self.hid_channels_, bias=True, weight_initializer='glorot'),
            nn.ELU(),
            nn.Dropout(self.dropout_p),
        ]
        for _ in range(self.num_layers - 1):
            enc_layers.extend(
                [
                    Linear(self.hid_channels_, self.hid_channels_, bias=True, weight_initializer='glorot'),
                    nn.ELU(),
                    nn.Dropout(self.dropout_p),
                ]
            )
        self.agent_encoder = nn.Sequential(*enc_layers)

        # Shared output head for all robots.
        self.node_classifier_head = Linear(
            self.heads * self.hid_channels,
            self.out_channels,
            bias=True,
            weight_initializer='glorot',
        )

        # One shared attention vector per hop.
        self.atts = nn.ParameterList(
            [nn.Parameter(torch.empty(self.heads, 2 * self.hid_channels)) for _ in range(self.K)]
        )

    def reset_parameters(self):
        for lin in self.dense_lins: lin.reset_parameters()
        for layer in self.agent_encoder:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
        torch.nn.init.xavier_uniform_(self.node_classifier_head.weight)
        if self.node_classifier_head.bias is not None:
            torch.nn.init.zeros_(self.node_classifier_head.bias)
        for att in self.atts:
            nn.init.xavier_uniform_(att)

    def hid_feat_init(self, x):
        # x shape: (batch_size, num_agents, in_channels)
        x = self.dropout(x)
        x = self.agent_encoder(x)  # (batch_size, num_agents, hid_channels_)

        # Reshape to (batch_size, num_agents, heads, hid_channels)
        x = x.view(-1, self.num_nodes, self.heads, self.hid_channels)
        return x


    def aero_propagate(self, h, edge_index):
        # h shape: (batch_size, num_agents, heads, hid_channels)
        batch_size = h.size(0)
        self.k = 0
        
        # Create batch-aware edge index
        edge_index_batch = edge_index
        
        # Initial hop attention
        z = h
        z_scale = z * self.decay_weights[0]

        for k in range(self.K):
            self.k = k + 1
            
            # Flatten for batch processing
            h_flat = h.reshape(-1, self.heads, self.hid_channels)  # (batch_size * num_agents, heads, hid_channels)
            z_scale_flat = z_scale.reshape(-1, self.heads, self.hid_channels)
            
            # Prepare edge features
            row, col = edge_index_batch
            z_scale_i = z_scale_flat[row]
            z_scale_j = z_scale_flat[col]
            
            # Compute attention coefficients
            a_ij = self.edge_att_pred(z_scale_i, z_scale_j, edge_index_batch, hop_k=k)
            
            # Prepare messages
            x_j = h_flat[col]
            messages = a_ij.unsqueeze(-1) * x_j
            
            # Aggregate messages
            out = torch.zeros_like(h_flat)
            out = scatter_add(messages, row, dim=0, out=out)
            
            # Reshape back
            h = out.view(batch_size, self.num_nodes, self.heads, self.hid_channels)
            
            # Update z and z_scale
            z += h
            if (k + 1) < len(self.decay_weights):
                z_scale = z * self.decay_weights[k + 1]
            else:
                z_scale = z
        
        return z #.clip(-1e6, 1e6)

    def node_classifier(self, z):
        # z shape: (batch_size, num_agents, heads, hid_channels)
        batch_size, num_agents, _, _ = z.size()
        z = z.reshape(batch_size, num_agents, -1)  # flatten heads
        z = self.elu(z)
        if self.add_dropout:
            z = self.dropout(z)
        z = self.node_classifier_head(z)

        if torch.isnan(z).any():
            print("Warning: NaNs in node_classifier output")

        return z # .clip(-1e6, 1e6)

    def forward(self, x, edge_index):
        # x shape: (batch_size, num_agents, in_channels)
        # edge_index shape: (2, num_edges)    
        h0 = self.hid_feat_init(x)  # (batch_size, num_agents, heads, hid_channels)
        z_k_max = self.aero_propagate(h0, edge_index)  # (batch_size, num_agents, heads, hid_channels)
        z_star = self.node_classifier(z_k_max)  # (batch_size, num_agents, out_channels)
        
        return z_star # .clip(-1e6, 1e6)

    def edge_att_pred(self, z_scale_i, z_scale_j, edge_index_batch, hop_k):
        # z_scale_i, z_scale_j shape: (batch_size * num_edges, heads, hid_channels)
        # edge_index_batch shape: (2, batch_size * num_edges)

        # edge attention (alpha_check_ij)
        # a_ij = z_scale_i + z_scale_j
        a_ij = torch.cat((z_scale_i, z_scale_j), dim=-1)
        a_ij = self.elu(a_ij)

        att_vec = self.atts[hop_k].unsqueeze(0)

        a_ij = (att_vec * a_ij).sum(dim=-1)
        a_ij = self.softplus(a_ij).clamp(max=1e6) + 1e-6

        # symmetric normalization (alpha_ij)
        row, col = edge_index_batch[0], edge_index_batch[1]
        num_flat_nodes = int(edge_index_batch.max().item()) + 1
        deg = scatter_add(a_ij, col, dim=0, dim_size=num_flat_nodes)
        deg_inv_sqrt = deg.pow(-0.5)  
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle zero degrees
        a_ij = deg_inv_sqrt[row] * a_ij * deg_inv_sqrt[col]        

        if torch.isnan(a_ij).any():
            raise ValueError("NaNs detected in edge attention normalization while handling layer k+1 = {self.k}")

        return a_ij
