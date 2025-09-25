#!/bin/bash
# LineFuse 两阶段自动训练脚本
# Stage 1: U-Net baseline → Stage 2: Diffusion model

set -e

# 配置参数
STAGE1_DATASET="training_datasets/stage1_unet/final_dataset"
STAGE2_DATASET="training_datasets/stage2_diffusion/final_dataset"
MODEL_OUTPUT_DIR="trained_models"
LOG_DIR="training_logs"

# 创建输出目录
mkdir -p ${MODEL_OUTPUT_DIR} ${LOG_DIR}

echo "=== LineFuse 两阶段自动训练 ==="

# 检查数据集
if [ ! -d "$STAGE1_DATASET" ]; then
    echo "错误: Stage 1 数据集不存在: $STAGE1_DATASET"
    echo "请先运行: ./auto_dataset_generator.sh"
    exit 1
fi

# 创建训练tmux会话
tmux new-session -d -s linefuse_train

# Stage 1: U-Net Baseline训练
tmux send-keys -t linefuse_train "echo '=== Stage 1: U-Net Baseline 训练开始 ==='" C-m
tmux send-keys -t linefuse_train "echo '数据集: $STAGE1_DATASET'" C-m
tmux send-keys -t linefuse_train "echo '预计训练时间: 1-2天'" C-m

# Easy难度训练
tmux send-keys -t linefuse_train "python main.py train \\
    --dataset $STAGE1_DATASET \\
    --model unet \\
    --difficulty easy \\
    --epochs 30 \\
    --batch-size 16 \\
    --lr 1e-4 \\
    2>&1 | tee ${LOG_DIR}/stage1_unet_easy.log" C-m

# Medium难度训练
tmux send-keys -t linefuse_train "python main.py train \\
    --dataset $STAGE1_DATASET \\
    --model unet \\
    --difficulty medium \\
    --epochs 30 \\
    --batch-size 16 \\
    --lr 1e-4 \\
    2>&1 | tee ${LOG_DIR}/stage1_unet_medium.log" C-m

# Hard难度训练
tmux send-keys -t linefuse_train "python main.py train \\
    --dataset $STAGE1_DATASET \\
    --model unet \\
    --difficulty hard \\
    --epochs 30 \\
    --batch-size 16 \\
    --lr 5e-5 \\
    2>&1 | tee ${LOG_DIR}/stage1_unet_hard.log" C-m

tmux send-keys -t linefuse_train "echo '=== Stage 1完成，开始Stage 2 ==='" C-m

# Stage 2: 扩散模型训练 (仅在Stage2数据集存在时)
tmux send-keys -t linefuse_train "
if [ -d '$STAGE2_DATASET' ]; then
    echo '=== Stage 2: 扩散模型训练开始 ==='
    echo '数据集: $STAGE2_DATASET'
    echo '预计训练时间: 5-7天'

    # Easy难度扩散模型
    python main.py train \\
        --dataset $STAGE2_DATASET \\
        --model diffusion \\
        --difficulty easy \\
        --epochs 100 \\
        --batch-size 8 \\
        --lr 1e-5 \\
        2>&1 | tee ${LOG_DIR}/stage2_diffusion_easy.log

    # Medium难度扩散模型
    python main.py train \\
        --dataset $STAGE2_DATASET \\
        --model diffusion \\
        --difficulty medium \\
        --epochs 100 \\
        --batch-size 8 \\
        --lr 1e-5 \\
        2>&1 | tee ${LOG_DIR}/stage2_diffusion_medium.log

    # Hard难度扩散模型
    python main.py train \\
        --dataset $STAGE2_DATASET \\
        --model diffusion \\
        --difficulty hard \\
        --epochs 100 \\
        --batch-size 6 \\
        --lr 5e-6 \\
        2>&1 | tee ${LOG_DIR}/stage2_diffusion_hard.log

    echo '=== 两阶段训练全部完成! ==='
else
    echo '跳过Stage 2: 扩散模型数据集不存在'
    echo '如需训练扩散模型，请先生成Stage 2数据集'
fi
" C-m

# 训练完成后的模型评估
tmux send-keys -t linefuse_train "echo '开始模型性能评估...'" C-m
tmux send-keys -t linefuse_train "python -c \"
import os
import glob

model_dir = '$MODEL_OUTPUT_DIR'
log_dir = '$LOG_DIR'

print('=== 训练结果汇总 ===')
print('模型文件:')
for model_file in glob.glob(os.path.join(model_dir, '*.pth')):
    size_mb = os.path.getsize(model_file) / (1024*1024)
    print(f'  {os.path.basename(model_file)}: {size_mb:.1f}MB')

print('\\n训练日志:')
for log_file in glob.glob(os.path.join(log_dir, '*.log')):
    size_kb = os.path.getsize(log_file) / 1024
    print(f'  {os.path.basename(log_file)}: {size_kb:.1f}KB')

print('\\n训练完成! 🎉')
print('下一步: 运行模型推理测试')
\"" C-m

echo "训练任务已在tmux会话 'linefuse_train' 中启动"
echo "监控进度: tmux attach -t linefuse_train"
echo "后台运行: tmux detach (Ctrl+B, D)"
echo "查看日志: tail -f ${LOG_DIR}/*.log"