# LineFuse 快速上手指南

## 🎯 当前状态
**数据生成完成 ✅** → **准备GPU环境训练模型** 🚀

## ⚡ 立即可用功能

### 1. 一键测试全部功能
```bash
./run_tests.sh  # 运行完整测试套件
```

### 2. 单独功能测试
```bash
python tests/test_clean_chart.py      # 清晰图表生成
python tests/test_enhanced_blur.py    # 增强模糊效果(需GPU)
python tests/test_data_generation.py  # 完整数据流程
```

### 3. 大规模数据集
```bash
python generate_large_dataset.py  # 生成生产数据集
```

## 🔄 GPU环境切换后执行

### 1. 安装依赖
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python albumentations diffusers transformers
```

### 2. 测试增强功能
```bash
python test_enhanced_blur.py  # 测试12种模糊效果
```

### 3. 开始训练
```bash
# Baseline模型训练
python src/models/trainer.py --model unet --data dataset_generation/final_dataset

# 扩散模型训练
python src/models/trainer.py --model diffusion --data dataset_generation/final_dataset
```

## 📊 可用数据
- **测试数据**: `test_data/dataset/` (12张图像)
- **生产数据**: `dataset_generation/final_dataset/` (400张图像)

## 🎪 核心功能
- **12种模糊效果**: gaussian, motion, compression, scan, lowres, text, lines, morphology, localblur, threshold, saltpepper, composite
- **自动数据集划分**: train/val/test (8:1:1)
- **批量处理**: 支持大规模数据生成

## 📋 下次要做的事
1. 切换GPU环境 ✋
2. 安装完整依赖 📦
3. 测试增强模糊功能 🧪
4. 训练Baseline模型 🏃‍♂️
5. 实现扩散模型 🌟

**数据已就绪，开始深度学习训练！** 🎯