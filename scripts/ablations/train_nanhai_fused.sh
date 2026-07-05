#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

python train.py \
    --dataset nanhai \
    --input_len 30 \
    --output_len 15 \
    --seasonal_features dayofyear \
    --batch_size 8 \
    --epochs 200 \
    --lr 0.0005 \
    --weight_decay 0.0001 \
    --num_experts 4 \
    --hidden_dim 32 \
    --num_layers 1 \
    --lambda_fair 0.5 \
    --graph_mode fused \
    --sigma 200 \
    --knn_k 6 \
    --kernel_size 3,5,7 \
    --temporal_dilations 1,2,3 \
    --dropout 0.2 \
    --temperature 1.0 \
    --early_stop_patience 25 \
    --seed 42 \
    --exp_name fused
