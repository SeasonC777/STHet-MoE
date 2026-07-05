"""
Data loader for ocean chlorophyll prediction
"""
import calendar
import csv
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ChlorophyllDataset(Dataset):
    def __init__(self, input_data, target_data, input_len, output_len):
        """
        Args:
            input_data: (T, N, F_in) - normalized Chl-a plus optional time features
            target_data: (T, N, F_out) - normalized Chl-a target only
            input_len: input sequence length
            output_len: output sequence length
        """
        self.input_data = input_data
        self.target_data = target_data
        self.input_len = input_len
        self.output_len = output_len
        self.len = input_data.shape[0] - input_len - output_len + 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.input_data[idx:idx + self.input_len]  # (input_len, N, F_in)
        y = self.target_data[idx + self.input_len:idx + self.input_len + self.output_len]  # (output_len, N, F_out)
        return torch.FloatTensor(x), torch.FloatTensor(y)


def _csv_file_for_dataset(args, num_nodes):
    if args.dataset == 'bohai':
        filename = 'bohai_300.csv'
    else:
        filename = f'{args.dataset}_{num_nodes}.csv'
    return f'{args.data_path}/{args.dataset}/{filename}'


def _load_time_labels(csv_file):
    with open(csv_file, newline='', encoding='utf-8-sig') as handle:
        for row in csv.reader(handle):
            normalized = [cell.strip().lower() for cell in row[:3]]
            if normalized == ['date', 'lat', 'lon']:
                return [cell.strip() for cell in row[3:] if cell.strip()]

    raise ValueError(f"Could not find a 'date,lat,lon' header row in {csv_file}")


def _parse_datetime_label(label):
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%Y%m%d', '%m/%d/%Y', '%d/%m/%Y'):
        try:
            return datetime.strptime(label, fmt)
        except ValueError:
            continue
    return None


def build_dayofyear_features(time_labels, num_timesteps):
    """
    Build ST-MoE-style day-of-year sin/cos features for each time step.
    Numeric labels are treated as 1-based day indices and wrapped by 365.
    """
    if len(time_labels) != num_timesteps:
        raise ValueError(
            f"CSV time label count ({len(time_labels)}) does not match npz timesteps ({num_timesteps})."
        )

    numeric_labels = []
    all_numeric = True
    for label in time_labels:
        try:
            numeric_labels.append(float(label))
        except ValueError:
            all_numeric = False
            break

    if all_numeric:
        labels = np.asarray(numeric_labels, dtype=np.float32)
        day_of_year = np.mod(labels - 1.0, 365.0) + 1.0
        period = np.full_like(day_of_year, 365.0)
    else:
        parsed = [_parse_datetime_label(label) for label in time_labels]
        if all(item is not None for item in parsed):
            day_of_year = np.asarray([item.timetuple().tm_yday for item in parsed], dtype=np.float32)
            period = np.asarray([366.0 if calendar.isleap(item.year) else 365.0 for item in parsed], dtype=np.float32)
        else:
            step_index = np.arange(num_timesteps, dtype=np.float32)
            day_of_year = np.mod(step_index, 365.0) + 1.0
            period = np.full_like(day_of_year, 365.0)

    angle = 2.0 * np.pi * day_of_year / period
    return np.stack([np.sin(angle), np.cos(angle)], axis=-1).astype(np.float32)


def load_data(args):
    """
    Load chlorophyll data and adjacency matrix
    """
    # Load data
    data_file = f'{args.data_path}/{args.dataset}/{args.dataset}_300.npz' if args.dataset == 'bohai' \
                else f'{args.data_path}/{args.dataset}/{args.dataset}.npz'
    raw_data = np.load(data_file)['data']  # (T, N, 1)

    # Load adjacency matrix
    adj_file = f'{args.data_path}/{args.dataset}/adj.npy'
    adj = np.load(adj_file)  # (N, N)

    print(f"Data shape: {raw_data.shape}")
    print(f"Adjacency shape: {adj.shape}")

    num_samples, num_nodes, _ = raw_data.shape

    # Z-score normalization
    mean = raw_data.mean()
    std = raw_data.std()
    if std < 1e-6:
        std = 1.0
    target_data = (raw_data - mean) / std
    input_data = target_data

    seasonal_features = getattr(args, 'seasonal_features', 'dayofyear')
    if seasonal_features == 'dayofyear':
        csv_file = _csv_file_for_dataset(args, num_nodes)
        time_labels = _load_time_labels(csv_file)
        temporal_features = build_dayofyear_features(time_labels, num_samples)
        temporal_features = np.broadcast_to(
            temporal_features[:, None, :],
            (num_samples, num_nodes, temporal_features.shape[-1])
        )
        input_data = np.concatenate([target_data, temporal_features], axis=-1).astype(np.float32)
        print(f"Input features: Chl-a + day-of-year sin/cos ({input_data.shape[-1]} channels)")
    elif seasonal_features == 'none':
        print("Input features: Chl-a only")
    else:
        raise ValueError(f"Unknown seasonal_features '{seasonal_features}'. Use 'dayofyear' or 'none'.")

    # Split data: 6:2:2
    train_size = int(num_samples * args.train_ratio)
    val_size = int(num_samples * args.val_ratio)

    train_input = input_data[:train_size]
    val_input = input_data[train_size:train_size + val_size]
    test_input = input_data[train_size + val_size:]

    train_target = target_data[:train_size]
    val_target = target_data[train_size:train_size + val_size]
    test_target = target_data[train_size + val_size:]

    # Training statistics in original scale for WMAE/STFairBench.
    train_raw = raw_data[:train_size]
    node_mean = train_raw.mean(axis=(0, 2))  # (N,)
    global_mean = train_raw.mean()

    print(f"Train samples: {train_input.shape[0]}")
    print(f"Val samples: {val_input.shape[0]}")
    print(f"Test samples: {test_input.shape[0]}")

    # Create datasets
    train_dataset = ChlorophyllDataset(train_input, train_target, args.input_len, args.output_len)
    val_dataset = ChlorophyllDataset(val_input, val_target, args.input_len, args.output_len)
    test_dataset = ChlorophyllDataset(test_input, test_target, args.input_len, args.output_len)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Store normalization parameters
    scaler = {'mean': mean, 'std': std, 'target_dim': target_data.shape[-1]}
    data_mean = {'node_mean': node_mean, 'global_mean': global_mean}

    return train_loader, val_loader, test_loader, adj, scaler, data_mean, num_nodes, input_data.shape[-1]


def compute_node_distances(adj):
    """
    Compute pairwise distances from adjacency matrix
    Assumes adj encodes spatial relationships
    """
    # If adj is already a distance matrix, use it directly
    # Otherwise, compute from adjacency (simplified approach)
    num_nodes = adj.shape[0]

    # Create distance matrix from adjacency
    # For simplicity, we use: distance = 1 / (adj + eps) for connected nodes
    distances = np.zeros_like(adj)
    eps = 1e-5

    # If adjacency represents distances already
    if np.diag(adj).sum() == 0:  # no self-loops, likely distance matrix
        distances = adj.copy()
    else:
        # Convert adjacency to distance
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    distances[i, j] = 0
                elif adj[i, j] > 0:
                    distances[i, j] = 1.0 / (adj[i, j] + eps)
                else:
                    distances[i, j] = np.inf

    return distances
