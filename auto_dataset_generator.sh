#!/bin/bash
# LineFuse 自动数据集生成脚本
# 使用tmux管理长时间任务

set -e

# 配置参数
STAGE1_SAMPLES=8000    # U-Net训练数据集
STAGE2_SAMPLES=25000   # 扩散模型数据集
BASE_OUTPUT_DIR="dataset/training_dataset"

echo "=== LineFuse 自动数据集生成 ==="
echo "Stage 1: ${STAGE1_SAMPLES} 样本 (U-Net baseline)"
echo "Stage 2: ${STAGE2_SAMPLES} 样本 (Diffusion model)"

# 创建tmux会话
tmux new-session -d -s linefuse_data

# Stage 1: U-Net数据集生成
tmux send-keys -t linefuse_data "echo '开始生成Stage 1数据集...'" C-m
tmux send-keys -t linefuse_data "python main.py generate --samples ${STAGE1_SAMPLES} --output ${BASE_OUTPUT_DIR}/stage1_unet --style-level 0.6 --image-size 1024" C-m
tmux send-keys -t linefuse_data "echo 'Stage 1完成! 开始生成Stage 2数据集...'" C-m

# Stage 2: 扩散模型数据集生成
tmux send-keys -t linefuse_data "python main.py generate --samples ${STAGE2_SAMPLES} --output ${BASE_OUTPUT_DIR}/stage2_diffusion --style-level 1.0 --image-size 1024" C-m
tmux send-keys -t linefuse_data "echo '数据集生成完成!'" C-m

# 数据集验证
tmux send-keys -t linefuse_data "python -c \"
import os
stage1_path = '${BASE_OUTPUT_DIR}/stage1_unet/final_dataset'
stage2_path = '${BASE_OUTPUT_DIR}/stage2_diffusion/final_dataset'

def count_images(path):
    count = 0
    for root, dirs, files in os.walk(path):
        count += len([f for f in files if f.endswith('.png')])
    return count

if os.path.exists(stage1_path):
    stage1_count = count_images(stage1_path)
    print(f'Stage 1 数据集: {stage1_count} 张图像')

if os.path.exists(stage2_path):
    stage2_count = count_images(stage2_path)
    print(f'Stage 2 数据集: {stage2_count} 张图像')

print('数据集生成验证完成!')
\"" C-m

echo "数据生成任务已在tmux会话 'linefuse_data' 中启动"
echo "监控进度: tmux attach -t linefuse_data"
echo "后台运行: tmux detach (Ctrl+B, D)"