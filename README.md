# STHet-MoE

Spatio-Temporal Heterogeneity-Aware Mixture-of-Experts for Satellite-Derived Marine Chlorophyll-a Forecasting

Core implementation for the paper code of STHet-MoE, a heterogeneous mixture-of-experts model for fair spatial-temporal forecasting on ocean chlorophyll concentration data.

The repository keeps only the source code needed for model definition, training, evaluation, and ablation runs. Datasets, checkpoints, logs, prediction archives, and generated figures are intentionally excluded.

## Repository Structure

```text
STHet-MoE/
|-- config.py                  # Command-line configuration
|-- train.py                   # Main training and evaluation entry point
|-- eval_horizon_metrics.py    # Per-horizon metric computation from saved predictions
|-- test_model.py              # Lightweight model instantiation smoke test
|-- requirements.txt           # Python dependencies
|-- models/
|   |-- higcn.py               # Heterogeneity-informed graph convolution
|   |-- tcn.py                 # Multi-scale temporal convolution
|   `-- himoe.py               # HiMoE / node-wise MoE model
|-- utils/
|   |-- data_loader.py         # Dataset loading and temporal feature construction
|   `-- metrics.py             # Accuracy and fairness-aware metrics
`-- scripts/ablations/         # Shell scripts for ablation experiments
```

## Requirements

Python 3.8+ is recommended.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch build that matches your CUDA version before running the experiments.

## Data Preparation

Data files are not included in this repository. Put them under `data/` with the following layout:

```text
data/
|-- bohai/
|   |-- bohai_300.npz          # npz file with key "data", shape (T, N, 1)
|   |-- bohai_300.csv          # time labels used for day-of-year features
|   `-- adj.npy                # adjacency or distance matrix, shape (N, N)
`-- nanhai/
    |-- nanhai.npz             # npz file with key "data", shape (T, N, 1)
    |-- nanhai_265.csv         # time labels used for day-of-year features
    `-- adj.npy                # adjacency or distance matrix, shape (N, N)
```

The training split defaults to 60% train, 20% validation, and 20% test. By default, the loader appends day-of-year sine and cosine features to the chlorophyll input channel.

## Quick Check

Run a lightweight forward-pass test:

```bash
python test_model.py
```

## Training

Train on the Bohai dataset:

```bash
python train.py \
  --dataset bohai \
  --input_len 30 \
  --output_len 15 \
  --batch_size 64 \
  --epochs 500 \
  --lr 0.003 \
  --num_experts 14 \
  --hidden_dim 64 \
  --num_layers 4 \
  --lambda_fair 0.5
```

Train on the South China Sea dataset:

```bash
python train.py \
  --dataset nanhai \
  --input_len 30 \
  --output_len 15 \
  --batch_size 64 \
  --epochs 500 \
  --lr 0.003 \
  --num_experts 14 \
  --hidden_dim 64 \
  --num_layers 4 \
  --lambda_fair 0.5
```

Outputs are written to:

- `checkpoints/`: best model checkpoints
- `logs/`: result JSON files and prediction archives

Both directories are ignored by git.

## Ablation Experiments

Ablation scripts are provided in `scripts/ablations/`. Examples:

```bash
bash scripts/ablations/train_bohai_full.sh
bash scripts/ablations/train_nanhai_no_moe.sh
bash scripts/ablations/train_nanhai_no_graph_conv.sh
```

Supported ablation flags include:

- `--no_moe`
- `--no_graph_conv`
- `--no_tcn`
- `--no_edge_gate`
- `--no_routing_mask`

## Evaluation

The training script reports MAE, RMSE, MAPE, MWMAE, and SWMAE. It also saves per-horizon metrics after testing.

To recompute per-horizon metrics from saved predictions:

```bash
python eval_horizon_metrics.py --dataset bohai
python eval_horizon_metrics.py --dataset nanhai
```

## Citation

If you use this code, please cite the corresponding paper. The BibTeX entry can be added here after the paper metadata is finalized.

## License

No license file is included yet. Add a license before distributing the code for reuse outside the intended research release.
