"""
Configuration file for STHet-MoE model
"""
import argparse

def parse_int_tuple(value):
    values = tuple(int(part.strip()) for part in str(value).split(',') if part.strip())
    if not values:
        raise argparse.ArgumentTypeError('value must contain at least one integer')
    return values[0] if len(values) == 1 else values

def get_config():
    parser = argparse.ArgumentParser(description='STHet-MoE for Ocean Chlorophyll Prediction')

    # Data parameters
    parser.add_argument('--dataset', type=str, default='bohai',
                        choices=['bohai', 'nanhai'], help='Dataset name')
    parser.add_argument('--data_path', type=str, default='data/', help='Data path')
    parser.add_argument('--input_len', type=int, default=30, help='Input sequence length')
    parser.add_argument('--output_len', type=int, default=15, help='Output sequence length')
    parser.add_argument('--seasonal_features', type=str, default='dayofyear',
                        choices=['dayofyear', 'none'], help='Seasonal input features')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test data ratio')

    # Model parameters
    parser.add_argument('--num_experts', type=int, default=14, help='Number of experts in MoE')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--gcn_hidden', type=int, default=64, help='GCN hidden dimension')
    parser.add_argument('--mlp_hidden', type=int, default=8, help='MLP hidden dimension for graph fusion')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in each expert')
    parser.add_argument('--kernel_size', type=parse_int_tuple, default=(3, 5, 7),
                        help='Kernel size(s) for TCN, e.g. 3 or 3,5,7')
    parser.add_argument('--temporal_dilations', type=parse_int_tuple, default=(1, 2, 3),
                        help='Temporal dilation(s) for multi-scale TCN, e.g. 1 or 1,2,3')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for gating')
    parser.add_argument('--graph_mode', type=str, default='dynamic',
                        choices=['dynamic', 'static', 'fused'],
                        help='Graph mode: dynamic, static, or fused dynamic+static adjacency')
    parser.add_argument('--no_moe', action='store_true', help='Ablation: use a single expert without MoE gating')
    parser.add_argument('--no_graph_conv', action='store_true', help='Ablation: disable STHetGCN graph convolution')
    parser.add_argument('--no_tcn', action='store_true', help='Ablation: disable TCN layers')
    parser.add_argument('--no_edge_gate', action='store_true',
                        help='Ablation: remove edge-level gating from fused dynamic+static graph fusion')
    parser.add_argument('--no_routing_mask', action='store_true',
                        help='Ablation: remove routing-based expert graph masks')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--lambda_fair', type=float, default=0.5, help='Fairness loss weight')
    parser.add_argument('--early_stop_patience', type=int, default=50, help='Early stopping patience')

    # Gaussian kernel parameters for static adjacency
    parser.add_argument('--sigma', type=float, default=100.0, help='Gaussian kernel sigma')
    parser.add_argument('--epsilon', type=float, default=0.5, help='Adjacency threshold')
    parser.add_argument('--knn_k', type=int, default=10, help='Number of nearest neighbors for static adjacency')

    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--save_path', type=str, default='checkpoints/', help='Model save path')
    parser.add_argument('--log_path', type=str, default='logs/', help='Log save path')
    parser.add_argument('--exp_name', type=str, default='', help='Experiment name suffix for checkpoints and logs')

    args = parser.parse_args()
    return args
