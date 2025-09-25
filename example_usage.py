#!/usr/bin/env python3
"""
LineFuse 使用示例
演示完整的数据生成->训练->推理流程
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"命令: {' '.join(cmd)}")
    print('='*60)

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ 命令执行失败: {result.returncode}")
        return False
    else:
        print(f"✅ 命令执行成功")
        return True

def main():
    print("""
    🌟 LineFuse 完整流程演示 🌟

    本脚本将演示：
    1. 生成小规模演示数据集
    2. 训练U-Net基线模型
    3. 训练扩散模型（可选）

    预计完成时间：10-30分钟（取决于硬件）
    """)

    # 检查环境
    print("🔍 检查Python环境...")
    try:
        import torch
        import cv2
        import matplotlib
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ OpenCV: {cv2.__version__}")
        print(f"✅ GPU可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU设备: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请先安装依赖: conda install pytorch torchvision opencv -c pytorch -c conda-forge")
        return False

    # 步骤1: 生成演示数据集
    if not run_command(['python', 'main.py', 'demo'],
                      "步骤1: 生成演示数据集 (10个样本)"):
        return False

    # 检查数据集是否生成成功
    dataset_path = Path('demo_dataset/final_dataset')
    if not dataset_path.exists():
        print("❌ 数据集生成失败")
        return False

    # 显示数据集统计
    difficulties = ['easy', 'medium', 'hard']
    for diff in difficulties:
        diff_path = dataset_path / diff
        if diff_path.exists():
            train_clean = len(list((diff_path / 'train' / 'clean').glob('*.png')))
            train_blur = len(list((diff_path / 'train' / 'blur').glob('*.png')))
            print(f"📊 {diff.upper()}: {train_clean} 清晰图 + {train_blur} 模糊图")

    print("\n🤔 是否继续训练模型？ (y/n): ", end='')
    if input().lower() != 'y':
        print("演示结束，数据集已生成在: demo_dataset/")
        return True

    # 步骤2: 训练U-Net基线模型
    if not run_command([
        'python', 'main.py', 'train',
        '--dataset', 'demo_dataset/final_dataset',
        '--model', 'unet',
        '--difficulty', 'easy',
        '--epochs', '10',
        '--batch-size', '4',
        '--lr', '1e-4'
    ], "步骤2: 训练U-Net基线模型 (easy难度, 10轮)"):
        print("⚠️  U-Net训练失败，但数据集已生成")

    # 检查是否训练扩散模型
    print("\n🤔 是否也训练扩散模型？ (实验性功能, y/n): ", end='')
    if input().lower() == 'y':
        if not run_command([
            'python', 'main.py', 'train',
            '--dataset', 'demo_dataset/final_dataset',
            '--model', 'diffusion',
            '--difficulty', 'easy',
            '--epochs', '5',
            '--batch-size', '2',
            '--lr', '5e-5'
        ], "步骤3: 训练扩散模型 (easy难度, 5轮)"):
            print("⚠️  扩散模型训练失败")

    print(f"""
    🎉 LineFuse 演示完成! 🎉

    📁 生成的文件:
    - 数据集: demo_dataset/final_dataset/
    - 模型: models/unet_easy/, models/diffusion_easy/

    🚀 下一步你可以:
    1. 生成更大的数据集: python main.py generate --samples 100
    2. 尝试更高难度: python main.py train --difficulty medium/hard
    3. 调整超参数: --epochs, --batch-size, --lr
    4. 查看模型性能: 检查 models/ 目录中的checkpoint

    📚 更多信息请查看: README.md, docs/
    """)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)