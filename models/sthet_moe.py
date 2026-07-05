"""
STHet-MoE: Spatio-Temporal Heterogeneity-Aware Mixture-of-Experts
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sthet_gcn import STHetGCN
from models.tcn import TemporalConvNet


class ExpertModel(nn.Module):
    """
    Single expert model: 4 layers of STHetGCN + TCN
    """
    def __init__(self, in_dim, hidden_dim, static_adj, num_layers=4,
                 kernel_size=(3, 5, 7), temporal_dilations=(1, 2, 3),
                 dropout=0.3, graph_mode='dynamic', use_graph_conv=True,
                 use_tcn=True, use_edge_gate=True):
        super(ExpertModel, self).__init__()
        self.num_layers = num_layers
        self.use_tcn = use_tcn

        # Build layers alternating STHetGCN and TCN
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = in_dim if i == 0 else hidden_dim

            # STHetGCN layer
            sthet_gcn = STHetGCN(layer_in, hidden_dim, hidden_dim, static_adj,
                                 dropout=dropout, graph_mode=graph_mode,
                                 use_graph_conv=use_graph_conv,
                                 use_edge_gate=use_edge_gate)

            # TCN layer
            tcn = None
            if use_tcn:
                tcn = TemporalConvNet(
                    hidden_dim,
                    [hidden_dim],
                    kernel_size=kernel_size,
                    dilations=temporal_dilations,
                    dropout=dropout
                )

            self.layers.append(nn.ModuleDict({
                'sthet_gcn': sthet_gcn,
                'tcn': tcn if tcn is not None else nn.Identity()
            }))

    def forward(self, x, adj_mask=None):
        """
        Args:
            x: (B, T, N, F)
            adj_mask: (N, N) adjacency mask for this expert
        Returns:
            out: (B, T, N, hidden_dim)
        """
        h = x
        for layer in self.layers:
            # Apply STHetGCN
            h = layer['sthet_gcn'](h, adj_mask)
            # Apply TCN
            if self.use_tcn:
                h = layer['tcn'](h)

        return h


class NMoE(nn.Module):
    """
    Node-wise Mixture-of-Experts
    """
    def __init__(self, num_experts, in_dim, hidden_dim, static_adj, num_layers=4,
                 kernel_size=(3, 5, 7), temporal_dilations=(1, 2, 3),
                 dropout=0.3, temperature=1.0, graph_mode='dynamic',
                 use_moe=True, use_graph_conv=True, use_tcn=True,
                 use_edge_gate=True, use_routing_mask=True):
        super(NMoE, self).__init__()
        if graph_mode not in {'dynamic', 'static', 'fused'}:
            raise ValueError("graph_mode must be 'dynamic', 'static', or 'fused'")
        self.use_moe = use_moe
        self.num_experts = num_experts if use_moe else 1
        self.num_nodes = static_adj.shape[0]
        self.temperature = temperature
        self.graph_mode = graph_mode
        self.use_routing_mask = use_routing_mask

        # Register static adjacency
        self.register_buffer('static_adj', static_adj)

        # Node cardinality representation learning
        self.node_repr_gcn = nn.Linear(1, hidden_dim)
        self.node_repr_weight = nn.Parameter(torch.randn(1, hidden_dim))

        # Expert centers for gating
        self.expert_centers = nn.Parameter(torch.randn(self.num_experts, hidden_dim))

        # Create experts
        self.experts = nn.ModuleList([
            ExpertModel(in_dim, hidden_dim, static_adj, num_layers, kernel_size,
                        temporal_dilations, dropout, graph_mode,
                        use_graph_conv, use_tcn, use_edge_gate)
            for _ in range(self.num_experts)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, in_dim)

        # Output gating weights
        self.output_gate_weight = nn.Parameter(torch.randn(self.num_experts, self.num_nodes))

    def compute_node_cardinality_repr(self, x):
        """
        Learn node cardinality representation
        Args:
            x: (B, T, N, F) input data
        Returns:
            node_repr: (N, hidden_dim) node representation
        """
        # Compute node-wise mean: (N,)
        node_mean = torch.mean(x, dim=(0, 1, 3))  # Average over batch, time, features -> (N,)
        node_mean = node_mean.unsqueeze(1)  # (N, 1)

        if self.graph_mode in {'static', 'fused'}:
            adj_norm = self.normalize_adj(self.static_adj)  # (N, N)
            node_repr = torch.matmul(adj_norm, node_mean)  # (N, N) x (N, 1) = (N, 1)
        else:
            node_repr = node_mean

        # Apply linear transformation + activation (蟽 in paper Eq. 6)
        node_repr = self.node_repr_gcn(node_repr)  # (N, 1) -> (N, hidden_dim)

        # Z-score normalization
        node_repr = (node_repr - node_repr.mean(0)) / (node_repr.std(0) + 1e-8)

        return node_repr

    def compute_gating_weights(self, node_repr):
        """
        Compute node-to-expert assignment probabilities
        Args:
            node_repr: (N, hidden_dim)
        Returns:
            gating_probs: (N, num_experts)
        """
        N = node_repr.shape[0]

        # Compute distances to expert centers: (N, num_experts)
        distances = torch.cdist(node_repr, self.expert_centers, p=2)  # (N, K)

        # Apply temperature-controlled softmax
        gating_probs = F.softmax(-distances / self.temperature, dim=1)  # (N, K)

        return gating_probs

    def compute_adjacency_masks(self, gating_probs):
        """
        Compute adjacency masks for each expert based on gating probabilities
        Args:
            gating_probs: (N, K) node-to-expert probabilities
        Returns:
            adj_masks: list of (N, N) masks for each expert
        """
        adj_masks = []
        for k in range(self.num_experts):
            # Get probabilities for expert k: (N,)
            prob_k = gating_probs[:, :k+1]  # (N, k+1)

            # Compute mask: outer product-like operation
            mask = torch.matmul(prob_k, prob_k.transpose(0, 1))  # (N, N)

            adj_masks.append(mask)

        return adj_masks

    def normalize_adj(self, adj):
        """Normalize adjacency matrix"""
        adj = adj + torch.eye(adj.shape[0], device=adj.device)
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    def forward(self, x):
        """
        Args:
            x: (B, T, N, F) input data
        Returns:
            out: (B, T', N, F) predictions
        """
        B, T, N, num_features = x.shape

        if not self.use_moe:
            out = self.experts[0](x, None)
            out = self.output_proj(out)
            return out

        # Compute node cardinality representation
        node_repr = self.compute_node_cardinality_repr(x)  # (N, hidden_dim)

        # Compute gating weights
        gating_probs = self.compute_gating_weights(node_repr)  # (N, K)

        # Compute adjacency masks for each expert
        adj_masks = self.compute_adjacency_masks(gating_probs) if self.use_routing_mask else None

        # Forward through all experts
        expert_outputs = []
        for k, expert in enumerate(self.experts):
            adj_mask = adj_masks[k] if adj_masks is not None else None
            expert_out = expert(x, adj_mask)  # (B, T, N, hidden_dim)
            expert_outputs.append(expert_out)

        # Stack expert outputs: (K, B, T, N, hidden_dim)
        expert_outputs = torch.stack(expert_outputs, dim=0)

        # Output gating
        output_weights = torch.sigmoid(self.output_gate_weight)  # (K, N)

        # Weighted aggregation: (B, T, N, hidden_dim)
        # output_weights: (K, N), expert_outputs: (K, B, T, N, hidden_dim)
        output_weights = output_weights.unsqueeze(1).unsqueeze(2).unsqueeze(4)  # (K, 1, 1, N, 1)
        expert_outputs_weighted = expert_outputs * output_weights  # (K, B, T, N, hidden_dim)

        out = torch.sum(expert_outputs_weighted, dim=0)  # (B, T, N, hidden_dim)

        # Project to output dimension
        out = self.output_proj(out)  # (B, T, N, num_features)

        return out


class STHetMoE(nn.Module):
    """
    Complete STHet-MoE model for spatio-temporal forecasting
    """
    def __init__(self, num_nodes, in_dim, hidden_dim, static_adj, num_experts=14,
                 num_layers=4, input_len=30, output_len=15,
                 kernel_size=(3, 5, 7), temporal_dilations=(1, 2, 3),
                 dropout=0.3, temperature=1.0, out_dim=1,
                 graph_mode='dynamic', use_moe=True, use_graph_conv=True,
                 use_tcn=True, use_edge_gate=True, use_routing_mask=True):
        super(STHetMoE, self).__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.input_len = input_len
        self.output_len = output_len

        # Input embedding
        self.input_embedding = nn.Linear(in_dim, hidden_dim)

        # NMoE module
        self.nmoe = NMoE(num_experts, hidden_dim, hidden_dim, static_adj,
                        num_layers, kernel_size, temporal_dilations,
                        dropout, temperature, graph_mode,
                        use_moe, use_graph_conv, use_tcn,
                        use_edge_gate, use_routing_mask)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_len * out_dim)
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, N, F) input sequence
        Returns:
            out: (B, T', N, F) predicted sequence
        """
        B, T, N, num_features = x.shape

        # Input embedding
        x_emb = self.input_embedding(x)  # (B, T, N, hidden_dim)

        # Apply NMoE
        h = self.nmoe(x_emb)  # (B, T, N, hidden_dim)

        # Aggregate temporal info: recent state + global context (cf. ST-MoE)
        h_recent = h[:, -1, :, :]      # (B, N, hidden_dim), last step
        h_context = h.mean(dim=1)      # (B, N, hidden_dim), mean over all T
        h_agg = h_recent + h_context   # (B, N, hidden_dim)

        # Project to output
        out = self.output_proj(h_agg)  # (B, N, output_len * out_dim)

        # Reshape to (B, output_len, N, out_dim)
        out = out.reshape(B, N, self.output_len, self.out_dim)
        out = out.permute(0, 2, 1, 3)  # (B, output_len, N, out_dim)

        return out
