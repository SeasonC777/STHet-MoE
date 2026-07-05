#!/bin/bash
# Quick start script for training HiMoE on Bohai dataset

echo "========================================"
echo "HiMoE Training Script"
echo "========================================"
echo ""

echo "Training on Bohai Sea dataset..."
echo "Input: 30 days, Output: 15 days"
echo ""

python train.py \
    --dataset bohai \
    --input_len 30 \
    --output_len 15 \
    --batch_size 64 \
    --epochs 500 \
    --lr 0.003 \
    --weight_decay 0.0001 \
    --num_experts 14 \
    --hidden_dim 64 \
    --num_layers 4 \
    --lambda_fair 0.5 \
    --dropout 0.3 \
    --temperature 1.0 \
    --early_stop_patience 50 \
    --seed 42

echo ""
echo "========================================"
echo "Training completed!"
echo "Check results in logs/ directory"
echo "========================================"
