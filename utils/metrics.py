"""
Utility functions for metrics and losses
"""
import torch
import numpy as np


def _check_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN/Inf.")


def _valid_label_mask(labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val

    mask = mask & torch.isfinite(labels)
    mask = mask.float()
    mask_mean = torch.mean(mask)
    if mask_mean == 0:
        raise ValueError("No valid labels for metric computation.")

    return mask / mask_mean


def masked_mae(preds, labels, null_val=np.nan):
    """Masked MAE loss"""
    _check_finite("preds", preds)
    mask = _valid_label_mask(labels, null_val)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    """Masked RMSE loss"""
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mse(preds, labels, null_val=np.nan):
    """Masked MSE loss"""
    _check_finite("preds", preds)
    mask = _valid_label_mask(labels, null_val)
    loss = (preds - labels) ** 2
    loss = loss * mask
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    """Masked MAPE loss"""
    _check_finite("preds", preds)
    mask = _valid_label_mask(labels, null_val)
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    return torch.mean(loss) * 100


def compute_wmae(preds, labels, data_mean, eps=1e-6):
    """
    Compute Weighted MAE (WMAE) as defined in the paper
    WMAE = mean(X) / mean(X_i) * (1/T') * sum|y_i - y_hat_i|

    Args:
        preds: (B, T, N, F) predictions
        labels: (B, T, N, F) ground truth
        data_mean: mean value of entire dataset for each node (N,) or scalar
    """
    _check_finite("preds", preds)
    _check_finite("labels", labels)

    # Compute node-wise MAE across batch, time, and feature dimensions: (N,)
    node_mae = torch.mean(torch.abs(preds - labels), dim=(0, 1, 3))

    # data_mean: dict with training global/node means, scalar, or (N,)
    if isinstance(data_mean, dict):
        node_mean = data_mean['node_mean']
        global_mean = data_mean.get('global_mean', None)

        if not isinstance(node_mean, torch.Tensor):
            node_mean = torch.tensor(node_mean, device=preds.device, dtype=preds.dtype)
        else:
            node_mean = node_mean.to(device=preds.device, dtype=preds.dtype)

        if global_mean is None:
            global_mean = node_mean.mean()
        elif not isinstance(global_mean, torch.Tensor):
            global_mean = torch.tensor(global_mean, device=preds.device, dtype=preds.dtype)
        else:
            global_mean = global_mean.to(device=preds.device, dtype=preds.dtype)
    elif isinstance(data_mean, (int, float)):
        data_mean = torch.tensor(data_mean, device=preds.device, dtype=preds.dtype)
        global_mean = data_mean
        node_mean = data_mean
    else:
        if not isinstance(data_mean, torch.Tensor):
            data_mean = torch.tensor(data_mean, device=preds.device, dtype=preds.dtype)
        else:
            # Ensure data_mean is on the same device
            data_mean = data_mean.to(device=preds.device, dtype=preds.dtype)
        global_mean = data_mean.mean()
        node_mean = data_mean  # (N,)

    # WMAE per node: (N,)
    node_mean = torch.clamp(node_mean, min=eps)
    global_mean = torch.clamp(global_mean, min=eps)
    wmae = (global_mean / node_mean) * node_mae

    return wmae


def compute_mwmae(wmae):
    """
    Mean of WMAE across all nodes
    Args:
        wmae: (B, N) or (N,)
    """
    return torch.mean(wmae)


def compute_swmae(wmae):
    """
    Standard deviation of WMAE across nodes
    Args:
        wmae: (B, N) or (N,)
    """
    return torch.std(wmae, unbiased=False)


def inverse_transform(data, scaler):
    """Convert normalized data back to original scale."""
    return data * scaler['std'] + scaler['mean']


def fairness_loss(preds, labels, data_mean, lambda_fair=0.5, scaler=None):
    """
    Fairness-aware loss: L = L_MWMAE + lambda * L_SWMAE

    Args:
        preds: (B, T, N, F) predictions
        labels: (B, T, N, F) ground truth
        data_mean: node-wise mean from training data
        lambda_fair: weight for fairness term
    """
    if scaler is not None:
        preds = inverse_transform(preds, scaler)
        labels = inverse_transform(labels, scaler)

    # Compute WMAE for each node
    wmae = compute_wmae(preds, labels, data_mean)  # (B, N)

    # Compute MWMAE (accuracy term)
    mwmae = compute_mwmae(wmae)

    # Compute SWMAE (fairness term)
    swmae = compute_swmae(wmae)

    # Total loss
    loss = mwmae + lambda_fair * swmae

    return loss, mwmae, swmae


def compute_all_metrics(preds, labels, data_mean, scaler):
    """
    Compute all evaluation metrics
    Returns metrics in original scale
    """
    # Inverse transform
    preds_original = inverse_transform(preds, scaler)
    labels_original = inverse_transform(labels, scaler)

    _check_finite("predictions", preds_original)
    _check_finite("labels", labels_original)

    mae = masked_mae(preds_original, labels_original).item()
    rmse = masked_rmse(preds_original, labels_original).item()
    mape = masked_mape(preds_original, labels_original).item()
    wmae = compute_wmae(preds_original, labels_original, data_mean)
    mwmae = compute_mwmae(wmae).item()
    swmae = compute_swmae(wmae).item()

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'MWMAE': mwmae,
        'SWMAE': swmae
    }
