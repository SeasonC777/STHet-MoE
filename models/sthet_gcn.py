"""
STHetGCN: Spatio-Temporal Heterogeneity-Aware Graph Convolutional Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class STHetGCN(nn.Module):
    """
    Spatio-temporal heterogeneity-aware graph convolutional network
    Handles trend heterogeneity through multi-graph fusion
    """
    def __init__(self, in_dim, hidden_dim, out_dim, static_adj, mlp_hidden=8,
                 dropout=0.3, graph_mode='dynamic', use_graph_conv=True,
                 use_edge_gate=True):
        super(STHetGCN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        if graph_mode not in {'dynamic', 'static', 'fused'}:
            raise ValueError("graph_mode must be 'dynamic', 'static', or 'fused'")
        self.graph_mode = graph_mode
        self.use_graph_conv = use_graph_conv
        self.use_edge_gate = use_edge_gate

        # Static adjacency matrix (Gaussian kernel based)
        self.register_buffer('static_adj', static_adj)

        if use_graph_conv:
            # MLP for graph fusion
            self.graph_fusion = nn.Sequential(
                nn.Linear(2, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, 1)
            )

            # Edge-level gating (tanh mask)
            self.edge_gate = (
                nn.Parameter(torch.randn(static_adj.shape[0], static_adj.shape[0]))
                if use_edge_gate else None
            )

            # GCN layers
            self.gcn1 = GraphConvolution(in_dim, hidden_dim)
            self.gcn2 = GraphConvolution(hidden_dim, out_dim)
            self.skip_projection = None
        else:
            self.graph_fusion = None
            self.edge_gate = None
            self.gcn1 = None
            self.gcn2 = None
            self.skip_projection = nn.Linear(in_dim, out_dim)

        self.dropout = nn.Dropout(dropout)

    def compute_dynamic_adj(self, x):
        """
        Compute dynamic adjacency matrix based on time series similarity
        Args:
            x: (B, T, N, F) input features
        """
        B, T, N, num_features = x.shape

        # Reshape for embedding: (B, N, T*F)
        x_flat = x.permute(0, 2, 1, 3).reshape(B, N, -1)

        # L2 normalize
        x_norm = torch.nn.functional.normalize(x_flat, p=2, dim=2)  # (B, N, T*F)

        # Compute similarity: (B, N, N)
        similarity = torch.bmm(x_norm, x_norm.transpose(1, 2))

        # Remove diagonal and apply softmax
        mask = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)
        similarity = similarity.masked_fill(mask.bool(), -1e9)
        dynamic_adj = torch.nn.functional.softmax(similarity, dim=-1)  # (B, N, N)

        return dynamic_adj

    def fuse_graphs(self, static_adj, dynamic_adj):
        """
        Fuse static and dynamic adjacency matrices
        Args:
            static_adj: (N, N)
            dynamic_adj: (B, N, N)
        """
        B, N, _ = dynamic_adj.shape

        # Expand static adj for batch
        static_adj_exp = static_adj.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)

        # Concatenate dynamic and static graphs as A_d || A_s in the paper.
        adj_concat = torch.stack([dynamic_adj, static_adj_exp], dim=-1)  # (B, N, N, 2)

        # Apply MLP fusion
        fused_adj = self.graph_fusion(adj_concat).squeeze(-1)  # (B, N, N)

        if self.use_edge_gate:
            edge_weights = torch.tanh(self.edge_gate)  # (N, N) in range [-1, 1]
            fused_adj = fused_adj * edge_weights.unsqueeze(0)  # (B, N, N)

        return fused_adj

    def forward(self, x, adj_mask=None):
        """
        Args:
            x: (B, T, N, F) input features
            adj_mask: (N, N) optional mask for expert-specific adjacency
        Returns:
            out: (B, T, N, out_dim)
        """
        B, T, N, num_features = x.shape

        if not self.use_graph_conv:
            return self.skip_projection(x)

        if self.graph_mode == 'static':
            fused_adj = self.static_adj.unsqueeze(0).expand(B, -1, -1)
        else:
            # Compute dynamic adjacency
            dynamic_adj = self.compute_dynamic_adj(x)  # (B, N, N)

        if self.graph_mode == 'dynamic':
            fused_adj = dynamic_adj
        elif self.graph_mode == 'fused':
            # Fuse static and dynamic graphs
            fused_adj = self.fuse_graphs(self.static_adj, dynamic_adj)  # (B, N, N)

        # Apply expert-specific mask if provided
        if adj_mask is not None:
            fused_adj = fused_adj * adj_mask.unsqueeze(0)  # (B, N, N)

        # Normalize adjacency with self-loops
        fused_adj = self.normalize_adj(fused_adj)

        # Apply GCN to each time step
        out_list = []
        for t in range(T):
            x_t = x[:, t, :, :]  # (B, N, num_features)

            # First GCN layer
            h = self.gcn1(x_t, fused_adj)  # (B, N, hidden_dim)
            h = torch.nn.functional.gelu(h)
            h = self.dropout(h)

            # Second GCN layer
            h = self.gcn2(h, fused_adj)  # (B, N, out_dim)

            out_list.append(h)

        # Stack along time dimension
        out = torch.stack(out_list, dim=1)  # (B, T, N, out_dim)

        return out

    def normalize_adj(self, adj, eps=1e-6):
        """
        Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
        Args:
            adj: (B, N, N) or (N, N)
        """
        if adj.dim() == 2:
            # Add self-loops
            adj = adj + torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)

            # Use absolute signed degree so negative correlations are preserved.
            degree = torch.sum(torch.abs(adj), dim=1).clamp_min(eps)
            d_inv_sqrt = torch.pow(degree, -0.5).flatten()
            d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

            # Normalize
            adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

        else:  # (B, N, N)
            B, N, _ = adj.shape

            # Add self-loops
            eye = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand(B, -1, -1)
            adj = adj + eye

            # Use absolute signed degree so negative correlations are preserved.
            degree = torch.sum(torch.abs(adj), dim=2).clamp_min(eps)  # (B, N)
            d_inv_sqrt = torch.pow(degree, -0.5)

            # Normalize
            adj_normalized = d_inv_sqrt.unsqueeze(2) * adj * d_inv_sqrt.unsqueeze(1)

        return adj_normalized


class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        """
        Args:
            x: (B, N, F_in)
            adj: (B, N, N) or (N, N)
        Returns:
            out: (B, N, F_out)
        """
        # x: (B, N, F_in), weight: (F_in, F_out)
        support = torch.matmul(x, self.weight)  # (B, N, F_out)

        # adj: (B, N, N) or (N, N)
        if adj.dim() == 2:
            output = torch.matmul(adj, support)  # (N, N) x (B, N, F_out) - need to handle
            # Reshape for batch processing
            B = x.shape[0]
            adj_exp = adj.unsqueeze(0).expand(B, -1, -1)
            output = torch.bmm(adj_exp, support)  # (B, N, F_out)
        else:
            output = torch.bmm(adj, support)  # (B, N, F_out)

        return output + self.bias
