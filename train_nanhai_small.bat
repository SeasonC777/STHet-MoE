@echo off
REM Memory-optimized training script for Nanhai dataset

echo ========================================
echo HiMoE Training Script - Nanhai Dataset (Memory Optimized)
echo ========================================
echo.

echo Training on South China Sea dataset...
echo Input: 30 days, Output: 15 days
echo Memory optimized: smaller model size
echo.

python train.py ^
    --dataset nanhai ^
    --input_len 30 ^
    --output_len 15 ^
    --batch_size 8 ^
    --epochs 500 ^
    --lr 0.003 ^
    --weight_decay 0.0001 ^
    --num_experts 7 ^
    --hidden_dim 32 ^
    --num_layers 3 ^
    --lambda_fair 0.5 ^
    --dropout 0.3 ^
    --temperature 1.0 ^
    --early_stop_patience 50 ^
    --seed 42

echo.
echo ========================================
echo Training completed!
echo Check results in logs/ directory
echo ========================================
pause
