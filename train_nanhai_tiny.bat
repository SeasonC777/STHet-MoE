@echo off
REM Ultra memory-optimized training script for Nanhai dataset

echo ========================================
echo HiMoE Training Script - Nanhai Dataset (Ultra Memory Optimized)
echo ========================================
echo.

echo Training on South China Sea dataset...
echo Input: 30 days, Output: 15 days
echo Ultra memory optimized: minimal model size
echo.

python train.py ^
    --dataset nanhai ^
    --input_len 30 ^
    --output_len 15 ^
    --batch_size 4 ^
    --epochs 500 ^
    --lr 0.003 ^
    --weight_decay 0.0001 ^
    --num_experts 4 ^
    --hidden_dim 32 ^
    --num_layers 2 ^
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
