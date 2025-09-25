#!/bin/bash
# LineFuse 训练进度监控脚本

echo "=== LineFuse 任务监控面板 ==="

# 检查tmux会话状态
check_session() {
    local session_name=$1
    if tmux has-session -t $session_name 2>/dev/null; then
        echo "✅ $session_name: 运行中"
        return 0
    else
        echo "❌ $session_name: 未运行"
        return 1
    fi
}

# 显示会话状态
echo "当前活动会话:"
check_session "linefuse_data" && echo "  - 数据生成任务运行中"
check_session "linefuse_train" && echo "  - 训练任务运行中"

echo -e "\n常用命令:"
echo "监控数据生成: tmux attach -t linefuse_data"
echo "监控训练进程: tmux attach -t linefuse_train"
echo "查看训练日志: tail -f training_logs/*.log"
echo "GPU使用情况: nvidia-smi"
echo "磁盘空间: df -h"

# 快速状态检查
echo -e "\n=== 快速状态检查 ==="

# 检查数据集大小
if [ -d "training_datasets" ]; then
    total_images=$(find training_datasets -name "*.png" 2>/dev/null | wc -l)
    dataset_size=$(du -sh training_datasets 2>/dev/null | cut -f1)
    echo "数据集: ${total_images} 张图像, 占用 ${dataset_size}"
fi

# 检查模型文件
if [ -d "trained_models" ]; then
    model_count=$(find trained_models -name "*.pth" 2>/dev/null | wc -l)
    model_size=$(du -sh trained_models 2>/dev/null | cut -f1)
    echo "模型文件: ${model_count} 个模型, 占用 ${model_size}"
fi

# 检查日志文件
if [ -d "training_logs" ]; then
    log_count=$(find training_logs -name "*.log" 2>/dev/null | wc -l)
    echo "日志文件: ${log_count} 个日志文件"

    # 显示最新的训练进度
    latest_log=$(find training_logs -name "*.log" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$latest_log" ]; then
        echo "最新训练进度 ($(basename $latest_log)):"
        tail -3 "$latest_log" 2>/dev/null | grep -E "(Epoch|Loss|PSNR)" | tail -1
    fi
fi

echo -e "\n退出任务: tmux kill-session -t [session_name]"