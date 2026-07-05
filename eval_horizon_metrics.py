"""
Compute per-horizon metrics from saved HiMoE predictions.
"""
import argparse
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Compute per-step MAE/RMSE/MAPE for saved predictions.')
    parser.add_argument('--dataset', type=str, default='nanhai', choices=['bohai', 'nanhai'])
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--prediction_file', type=str, default=None)
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--output_json', type=str, default=None)
    return parser.parse_args()


def data_file_for_dataset(data_path, dataset):
    if dataset == 'bohai':
        return os.path.join(data_path, dataset, f'{dataset}_300.npz')
    return os.path.join(data_path, dataset, f'{dataset}.npz')


def output_stem(dataset, exp_name):
    return f'{dataset}_{exp_name}' if exp_name else dataset


def compute_step_metrics(preds, labels, mean, std, eps=1e-5):
    preds = preds * std + mean
    labels = labels * std + mean

    diff = preds - labels
    abs_diff = np.abs(diff)
    sq_diff = diff ** 2
    denom = np.maximum(np.abs(labels), eps)

    mae = abs_diff.mean(axis=(0, 2, 3))
    rmse = np.sqrt(sq_diff.mean(axis=(0, 2, 3)))
    mape = (abs_diff / denom).mean(axis=(0, 2, 3)) * 100.0

    overall = {
        'MAE': float(abs_diff.mean()),
        'RMSE': float(np.sqrt(sq_diff.mean())),
        'MAPE': float((abs_diff / denom).mean() * 100.0),
    }

    return mae, rmse, mape, overall


def main():
    args = parse_args()

    stem = output_stem(args.dataset, args.exp_name)
    prediction_file = args.prediction_file or os.path.join('logs', f'{stem}_predictions.npz')
    output_csv = args.output_csv or os.path.join('logs', f'{stem}_horizon_metrics.csv')
    output_json = args.output_json or os.path.join('logs', f'{stem}_horizon_metrics.json')

    archive = np.load(prediction_file)
    preds = archive['predictions']
    labels = archive['labels']

    raw_data = np.load(data_file_for_dataset(args.data_path, args.dataset))['data']
    mean = float(raw_data.mean())
    std = float(raw_data.std())
    if std < 1e-6:
        std = 1.0

    mae, rmse, mape, overall = compute_step_metrics(preds, labels, mean, std)

    rows = []
    for step, (step_mae, step_rmse, step_mape) in enumerate(zip(mae, rmse, mape), start=1):
        rows.append({
            'step': step,
            'MAE': float(step_mae),
            'RMSE': float(step_rmse),
            'MAPE': float(step_mape),
        })

    avg = {
        'step': 'avg',
        'MAE': float(mae.mean()),
        'RMSE': float(rmse.mean()),
        'MAPE': float(mape.mean()),
    }

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', encoding='utf-8') as handle:
        handle.write('step,MAE,RMSE,MAPE\n')
        for row in rows:
            handle.write(f"{row['step']},{row['MAE']:.10f},{row['RMSE']:.10f},{row['MAPE']:.10f}\n")
        handle.write(f"avg,{avg['MAE']:.10f},{avg['RMSE']:.10f},{avg['MAPE']:.10f}\n")
        handle.write(f"overall,{overall['MAE']:.10f},{overall['RMSE']:.10f},{overall['MAPE']:.10f}\n")

    payload = {
        'dataset': args.dataset,
        'prediction_file': prediction_file,
        'scale': 'original',
        'mean': mean,
        'std': std,
        'per_step': rows,
        'step_average': avg,
        'overall': overall,
    }
    with open(output_json, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=4)

    print('step,MAE,RMSE,MAPE')
    for row in rows:
        print(f"{row['step']},{row['MAE']:.6f},{row['RMSE']:.6f},{row['MAPE']:.6f}")
    print(f"avg,{avg['MAE']:.6f},{avg['RMSE']:.6f},{avg['MAPE']:.6f}")
    print(f"overall,{overall['MAE']:.6f},{overall['RMSE']:.6f},{overall['MAPE']:.6f}")
    print(f"\nSaved CSV to {output_csv}")
    print(f"Saved JSON to {output_json}")


if __name__ == '__main__':
    main()
