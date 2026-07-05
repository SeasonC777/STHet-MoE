@echo off
REM Quick start script for training HiMoE on Bohai dataset
REM Optimized for 8GB GPU (RTX 4060 Laptop)

cd /d "%~dp0"

echo ========================================
echo HiMoE Training Script - Bohai Dataset
echo ========================================
echo.

echo Training on Bohai Sea dataset...
echo Input: 30 days, Output: 15 days
echo Graph mode: dynamic only, no static adjacency
echo Seasonal input: day-of-year sin/cos
echo Temporal model: multi-scale TCN, kernels=3/5/7, dilations=1/2/3
echo GPU Memory Optimized for 8GB GPU
echo.

python train.py ^
    --dataset bohai ^
    --input_len 30 ^
    --output_len 15 ^
    --seasonal_features dayofyear ^
    --batch_size 16 ^
    --epochs 200 ^
    --lr 0.001 ^
    --weight_decay 0.0001 ^
    --num_experts 3 ^
    --hidden_dim 32 ^
    --num_layers 1 ^
    --lambda_fair 0.5 ^
    --graph_mode dynamic ^
    --kernel_size 3,5,7 ^
    --temporal_dilations 1,2,3 ^
    --dropout 0.3 ^
    --temperature 1.0 ^
    --early_stop_patience 20 ^
    --seed 42

echo.
echo ========================================
echo Training completed!
echo Check results in logs/ directory
echo ========================================
pause
