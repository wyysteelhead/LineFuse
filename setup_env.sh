#!/bin/bash

# LineFuse环境安装脚本
echo "=== LineFuse项目环境配置 ==="

# 检查conda是否已安装
if ! command -v conda &> /dev/null; then
    echo "错误: conda未找到，请先安装miniconda/anaconda"
    exit 1
fi

# 初始化conda
echo "初始化conda..."
conda init bash
source ~/.bashrc

# 删除已存在的环境（如果有）
echo "检查并删除已存在的linefuse环境..."
conda env remove -n linefuse -y 2>/dev/null || true

# 创建新环境
echo "创建linefuse conda环境..."
conda env create -f environment.yml

# 激活环境并验证安装
echo "激活环境并验证安装..."
eval "$(conda shell.bash hook)"
conda activate linefuse

# 验证PyTorch和CUDA
echo "验证PyTorch和CUDA安装..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
else:
    print('CUDA not available - 请检查CUDA驱动')
"

echo "=== 环境配置完成 ==="
echo "使用方法:"
echo "conda activate linefuse"
echo "cd /root/autodl-tmp/LineFuse"
echo "python test_enhanced_blur.py"