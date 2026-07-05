"""
Training script for HiMoE model
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from datetime import datetime

from config import get_config
from utils.data_loader import load_data
from utils.metrics import fairness_loss, compute_all_metrics
from models.himoe import HiMoE


def output_stem(dataset, exp_name):
    return f'{dataset}_{exp_name}' if exp_name else dataset


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_static_adjacency(adj, sigma=10.0, epsilon=0.5, knn_k=None):
    """
    Create static adjacency matrix from a distance matrix.
    Args:
        adj: original adjacency matrix (distance-based)
        sigma: Gaussian kernel sigma
        epsilon: threshold for adjacency
        knn_k: if provided, use kNN instead of fixed distance threshold
    """
    num_nodes = adj.shape[0]
    static_adj = np.zeros_like(adj)

    for i in range(num_nodes):
        if knn_k is not None:
            distances = adj[i].copy()
            distances[i] = np.inf
            distances[distances <= 0] = np.inf
            neighbors = np.argsort(distances)[:knn_k]
            neighbors = neighbors[np.isfinite(distances[neighbors])]
            static_adj[i, neighbors] = np.exp(-(distances[neighbors] ** 2) / (sigma ** 2))
        else:
            for j in range(num_nodes):
                if i == j:
                    static_adj[i, j] = 1.0
                else:
                    dist = adj[i, j] if adj[i, j] > 0 else np.inf
                    if dist < np.inf and dist <= epsilon * sigma:
                        static_adj[i, j] = np.exp(-(dist ** 2) / (sigma ** 2))

    static_adj = np.maximum(static_adj, static_adj.T)
    np.fill_diagonal(static_adj, 1.0)

    return torch.FloatTensor(static_adj)


def train_epoch(model, train_loader, optimizer, device, data_mean, scaler, lambda_fair):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_mwmae = 0
    total_swmae = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (x, y) in enumerate(pbar):
        x = x.to(device)  # (B, T, N, F)
        y = y.to(device)  # (B, T', N, F)

        optimizer.zero_grad()

        # Forward
        pred = model(x)  # (B, T', N, F)
        if not torch.isfinite(pred).all():
            raise ValueError(f"Non-finite predictions at training batch {batch_idx}.")

        loss, mwmae, swmae = fairness_loss(pred, y, data_mean, lambda_fair, scaler=scaler)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at training batch {batch_idx}.")

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        total_mwmae += mwmae.item()
        total_swmae += swmae.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mwmae': f'{mwmae.item():.4f}',
            'swmae': f'{swmae.item():.4f}'
        })

    num_batches = len(train_loader)
    return total_loss / num_batches, total_mwmae / num_batches, total_swmae / num_batches


@torch.no_grad()
def evaluate(model, data_loader, device, data_mean, scaler):
    """Evaluate model"""
    model.eval()

    preds_all = []
    labels_all = []

    for batch_idx, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        if not torch.isfinite(pred).all():
            raise ValueError(f"Non-finite predictions at evaluation batch {batch_idx}.")

        preds_all.append(pred.cpu())
        labels_all.append(y.cpu())

    preds_all = torch.cat(preds_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)

    # Compute metrics
    metrics = compute_all_metrics(preds_all, labels_all, data_mean, scaler)

    return metrics, preds_all, labels_all


def compute_horizon_metrics(preds, labels, data_mean, scaler):
    """Compute metrics separately for each forecast step."""
    horizon_metrics = []
    for step in range(preds.shape[1]):
        metrics = compute_all_metrics(
            preds[:, step:step + 1],
            labels[:, step:step + 1],
            data_mean,
            scaler
        )
        horizon_metrics.append({
            'step': step + 1,
            **metrics
        })
    return horizon_metrics


def average_horizon_metrics(horizon_metrics):
    """Average per-step metrics across forecast horizons."""
    metric_names = ['MAE', 'RMSE', 'MAPE', 'MWMAE', 'SWMAE']
    return {
        metric_name: float(np.mean([row[metric_name] for row in horizon_metrics]))
        for metric_name in metric_names
    }


def print_test_metrics(test_metrics, horizon_metrics):
    """Print per-horizon and average test metrics."""
    avg_metrics = average_horizon_metrics(horizon_metrics)

    print("\nTest Results by Horizon:")
    print(f"{'Step':>6} {'MAE':>12} {'RMSE':>12} {'MAPE':>12} {'MWMAE':>12} {'SWMAE':>12}")
    for row in horizon_metrics:
        print(
            f"{row['step']:>6} "
            f"{row['MAE']:>12.4f} "
            f"{row['RMSE']:>12.4f} "
            f"{row['MAPE']:>12.4f} "
            f"{row['MWMAE']:>12.4f} "
            f"{row['SWMAE']:>12.4f}"
        )
    print(
        f"{'Avg':>6} "
        f"{avg_metrics['MAE']:>12.4f} "
        f"{avg_metrics['RMSE']:>12.4f} "
        f"{avg_metrics['MAPE']:>12.4f} "
        f"{avg_metrics['MWMAE']:>12.4f} "
        f"{avg_metrics['SWMAE']:>12.4f}"
    )
    print(
        f"{'Overall':>6} "
        f"{test_metrics['MAE']:>12.4f} "
        f"{test_metrics['RMSE']:>12.4f} "
        f"{test_metrics['MAPE']:>12.4f} "
        f"{test_metrics['MWMAE']:>12.4f} "
        f"{test_metrics['SWMAE']:>12.4f}"
    )


def main():
    # Get configuration
    args = get_config()

    # Set random seed
    set_seed(args.seed)

    # Create directories
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, adj, scaler, data_mean, num_nodes, num_features = load_data(args)

    # Create graph buffer. In dynamic mode the model ignores static edges and
    # only keeps an identity placeholder for node count / tensor shape.
    if args.graph_mode == 'dynamic':
        print("Graph mode: dynamic only (static adjacency disabled)")
        static_adj = torch.eye(num_nodes, dtype=torch.float32)
    elif args.graph_mode == 'static':
        print("Graph mode: static only")
        print("Creating static adjacency matrix...")
        static_adj = create_static_adjacency(adj, sigma=args.sigma, epsilon=args.epsilon, knn_k=args.knn_k)
        offdiag_edges = int((static_adj.numpy() > 0).sum() - static_adj.shape[0])
        print(f"Static adjacency: {offdiag_edges} off-diagonal edges, sigma={args.sigma}, knn_k={args.knn_k}")
    else:
        print("Graph mode: fused dynamic + static")
        print("Creating static adjacency matrix...")
        static_adj = create_static_adjacency(adj, sigma=args.sigma, epsilon=args.epsilon, knn_k=args.knn_k)
        offdiag_edges = int((static_adj.numpy() > 0).sum() - static_adj.shape[0])
        print(f"Static adjacency: {offdiag_edges} off-diagonal edges, sigma={args.sigma}, knn_k={args.knn_k}")
    static_adj = static_adj.to(device)

    # Move WMAE/STFairBench training statistics to the active device.
    print("Computing node statistics...")
    data_mean = {
        'node_mean': torch.as_tensor(data_mean['node_mean'], dtype=torch.float32, device=device),
        'global_mean': torch.as_tensor(data_mean['global_mean'], dtype=torch.float32, device=device)
    }

    # Create model
    print("Creating model...")
    print(f"Model channels: input_dim={num_features}, output_dim=1")
    print(f"Graph mode: {args.graph_mode}")
    print(
        f"Ablations: no_moe={args.no_moe}, no_graph_conv={args.no_graph_conv}, "
        f"no_tcn={args.no_tcn}, no_edge_gate={args.no_edge_gate}, "
        f"no_routing_mask={args.no_routing_mask}"
    )
    print(f"Temporal TCN: kernel_size={args.kernel_size}, dilations={args.temporal_dilations}")
    model = HiMoE(
        num_nodes=num_nodes,
        in_dim=num_features,
        hidden_dim=args.hidden_dim,
        static_adj=static_adj,
        num_experts=args.num_experts,
        num_layers=args.num_layers,
        input_len=args.input_len,
        output_len=args.output_len,
        kernel_size=args.kernel_size,
        temporal_dilations=args.temporal_dilations,
        dropout=args.dropout,
        temperature=args.temperature,
        out_dim=1,
        graph_mode=args.graph_mode,
        use_moe=not args.no_moe,
        use_graph_conv=not args.no_graph_conv,
        use_tcn=not args.no_tcn,
        use_edge_gate=not args.no_edge_gate,
        use_routing_mask=not args.no_routing_mask
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training loop
    print("Start training...")
    best_val_metric = float('inf')
    best_metric_name = 'MAE'
    patience_counter = 0
    train_history = []
    val_history = []
    last_lr = args.lr

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_mwmae, train_swmae = train_epoch(
            model, train_loader, optimizer, device, data_mean, scaler, args.lambda_fair
        )

        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, device, data_mean, scaler)
        if not all(np.isfinite(v) for v in val_metrics.values()):
            raise ValueError(f"Non-finite validation metrics at epoch {epoch + 1}: {val_metrics}")

        # Update scheduler and check for LR change
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics[best_metric_name])
        new_lr = optimizer.param_groups[0]['lr']

        if old_lr != new_lr:
            print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

        # Print metrics
        print(f"Train - Loss: {train_loss:.4f}, MWMAE: {train_mwmae:.4f}, SWMAE: {train_swmae:.4f}")
        print(f"Val - MAE: {val_metrics['MAE']:.4f}, RMSE: {val_metrics['RMSE']:.4f}, "
              f"MAPE: {val_metrics['MAPE']:.4f}, MWMAE: {val_metrics['MWMAE']:.4f}, "
              f"SWMAE: {val_metrics['SWMAE']:.4f}")

        # Save history
        train_history.append({
            'epoch': epoch + 1,
            'loss': train_loss,
            'MWMAE': train_mwmae,
            'SWMAE': train_swmae
        })
        val_history.append({
            'epoch': epoch + 1,
            **val_metrics
        })

        # Save best model based on the selected validation metric.
        if val_metrics[best_metric_name] < best_val_metric:
            best_val_metric = val_metrics[best_metric_name]
            patience_counter = 0

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'best_metric': best_metric_name,
                'best_metric_value': best_val_metric,
                'args': vars(args)
            }
            save_path = os.path.join(args.save_path, f'{output_stem(args.dataset, args.exp_name)}_best_model.pth')
            torch.save(checkpoint, save_path)
            print(f"Model saved to {save_path}")

        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model and test
    print("\nLoading best model for testing...")
    checkpoint = torch.load(
        os.path.join(args.save_path, f'{output_stem(args.dataset, args.exp_name)}_best_model.pth'),
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test
    print("Testing...")
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, device, data_mean, scaler)
    test_horizon_metrics = compute_horizon_metrics(test_preds, test_labels, data_mean, scaler)
    test_horizon_average = average_horizon_metrics(test_horizon_metrics)
    print_test_metrics(test_metrics, test_horizon_metrics)

    # Save results
    results = {
        'dataset': args.dataset,
        'test_metrics': test_metrics,
        'test_horizon_metrics': test_horizon_metrics,
        'test_horizon_average': test_horizon_average,
        'train_history': train_history,
        'val_history': val_history,
        'model_params': num_params,
        'args': vars(args),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    results_path = os.path.join(args.log_path, f'{output_stem(args.dataset, args.exp_name)}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_path}")

    # Save predictions
    np.savez(
        os.path.join(args.log_path, f'{output_stem(args.dataset, args.exp_name)}_predictions.npz'),
        predictions=test_preds.numpy(),
        labels=test_labels.numpy()
    )
    print(f"Predictions saved to {args.log_path}")


if __name__ == '__main__':
    main()
