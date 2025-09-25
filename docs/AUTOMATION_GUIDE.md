# LineFuse 自动化脚本使用指南

## 📋 概览

LineFuse 提供了完整的自动化流程，从真实数据转换到模型训练，通过3个核心脚本实现：

1. **数据转换**: `augmented_results/convert_data.py`
2. **数据集生成**: `auto_dataset_generator.sh`
3. **模型训练**: `auto_train_pipeline.sh`
4. **进度监控**: `monitor_progress.sh`

---

## 🔄 完整工作流程

### 第一步: 数据转换
**目标**: 将真实光谱数据转换为LineFuse格式

```bash
# 1. 进入数据目录
cd augmented_results

# 2. 运行转换脚本
python convert_data.py
```

**预期输出**:
```
=== LineFuse 数据格式转换 (v2.0) ===
发现 583 个CSV文件
✅ 转换完成!
生成光谱: 29150 条
输出目录: /root/autodl-tmp/LineFuse/dataset/csv_data
```

**结果检查**:
```bash
# 返回项目根目录
cd ..

# 检查转换结果
ls -la dataset/csv_data/ | head -5
find dataset/csv_data -name "*.csv" | wc -l
```

### 第二步: 自动生成训练数据集
**目标**: 生成两阶段训练所需的数据集

```bash
# 直接在项目根目录运行
./auto_dataset_generator.sh
```

**脚本功能**:
- **Stage 1**: 生成8,000样本的U-Net训练数据集
- **Stage 2**: 生成25,000样本的扩散模型数据集
- **自动验证**: 检查数据集完整性

**预期输出结构**:
```
training_datasets/
├── stage1_unet/final_dataset/          # U-Net训练数据 (~24,000张图)
│   ├── easy/
│   ├── medium/
│   └── hard/
└── stage2_diffusion/final_dataset/     # 扩散模型数据 (~75,000张图)
    ├── easy/
    ├── medium/
    └── hard/
```

### 第三步: 自动化模型训练
**目标**: 两阶段自动训练，U-Net基线 → 扩散模型

```bash
# 确保数据集已生成完成
./auto_train_pipeline.sh
```

**训练流程**:
1. **Stage 1: U-Net Baseline**
   - Easy难度: 30 epochs
   - Medium难度: 30 epochs
   - Hard难度: 30 epochs
   - 预计时间: 1-2天

2. **Stage 2: 扩散模型**
   - Easy难度: 100 epochs
   - Medium难度: 100 epochs
   - Hard难度: 100 epochs
   - 预计时间: 5-7天

### 第四步: 实时监控
**目标**: 监控任务进度和系统状态

```bash
# 随时检查任务状态
./monitor_progress.sh
```

---

## 🎮 详细使用说明

### 1. 数据转换脚本 (`convert_data.py`)

#### **位置**: `augmented_results/convert_data.py`
#### **功能**:
- 读取583个多光谱CSV文件
- 每个CSV包含50条光谱线
- 分离为29,150个单光谱CSV文件

#### **使用方法**:
```bash
cd augmented_results
python convert_data.py
```

#### **参数说明**:
- **输入**: `*_augmented_spectra.csv` 文件
- **输出**: `dataset/csv_data/spectrum_XXXXX.csv`
- **格式**: `wavelength,intensity` 两列

#### **交互选项**:
```
是否清理现有数据? (y/N): y  # 清理旧的测试数据
```

#### **故障排除**:
```bash
# 检查输入文件
ls -la augmented_results/*_spectra.csv | wc -l

# 检查输出结果
find dataset/csv_data -name "spectrum_*.csv" | wc -l

# 查看转换示例
head dataset/csv_data/spectrum_00000.csv
```

---

### 2. 自动数据集生成 (`auto_dataset_generator.sh`)

#### **位置**: 项目根目录
#### **功能**: 生成两阶段训练数据集

#### **配置参数**:
```bash
STAGE1_SAMPLES=8000     # U-Net数据集大小
STAGE2_SAMPLES=25000    # 扩散模型数据集大小
BASE_OUTPUT_DIR="training_datasets"
```

#### **使用方法**:
```bash
# 启动数据生成（后台运行）
./auto_dataset_generator.sh

# 监控生成进度
tmux attach -t linefuse_data

# 后台运行: 按 Ctrl+B, 然后按 D
```

#### **生成策略**:
- **Stage 1**: 中等样式多样性 (`--style-level 0.6`)
- **Stage 2**: 高样式多样性 (`--style-level 1.0`)
- **自动采样**: 从29,150条光谱中智能选择

#### **输出结果**:
```
Stage 1 数据集: 24,000+ 张图像 (8,000清晰 + 16,000+模糊)
Stage 2 数据集: 75,000+ 张图像 (25,000清晰 + 50,000+模糊)
```

---

### 3. 自动训练管道 (`auto_train_pipeline.sh`)

#### **位置**: 项目根目录
#### **功能**: 两阶段自动训练

#### **使用方法**:
```bash
# 启动训练（长时间运行）
./auto_train_pipeline.sh

# 监控训练进度
tmux attach -t linefuse_train

# 查看训练日志
tail -f training_logs/*.log
```

#### **训练配置**:

**Stage 1 (U-Net)**:
```bash
# Easy难度
--epochs 30 --batch-size 16 --lr 1e-4

# Medium难度
--epochs 30 --batch-size 16 --lr 1e-4

# Hard难度
--epochs 30 --batch-size 16 --lr 5e-5
```

**Stage 2 (扩散模型)**:
```bash
# Easy难度
--epochs 100 --batch-size 8 --lr 1e-5

# Medium难度
--epochs 100 --batch-size 8 --lr 1e-5

# Hard难度
--epochs 100 --batch-size 6 --lr 5e-6
```

#### **输出文件**:
```
trained_models/
├── unet_easy_best.pth      # U-Net模型
├── unet_medium_best.pth
├── unet_hard_best.pth
├── diffusion_easy_best.pth # 扩散模型
├── diffusion_medium_best.pth
└── diffusion_hard_best.pth

training_logs/
├── stage1_unet_easy.log    # 训练日志
├── stage1_unet_medium.log
├── stage1_unet_hard.log
├── stage2_diffusion_easy.log
├── stage2_diffusion_medium.log
└── stage2_diffusion_hard.log
```

---

### 4. 进度监控脚本 (`monitor_progress.sh`)

#### **位置**: 项目根目录
#### **功能**: 实时状态监控

#### **使用方法**:
```bash
# 随时检查状态
./monitor_progress.sh
```

#### **监控信息**:
```
=== LineFuse 任务监控面板 ===
当前活动会话:
✅ linefuse_data: 运行中
  - 数据生成任务运行中
✅ linefuse_train: 运行中
  - 训练任务运行中

常用命令:
监控数据生成: tmux attach -t linefuse_data
监控训练进程: tmux attach -t linefuse_train
查看训练日志: tail -f training_logs/*.log
GPU使用情况: nvidia-smi
磁盘空间: df -h

=== 快速状态检查 ===
数据集: 99,150 张图像, 占用 15.2G
模型文件: 6 个模型, 占用 2.1G
日志文件: 6 个日志文件
最新训练进度 (stage2_diffusion_hard.log):
Epoch 045/100 - Loss: 0.0234, PSNR: 31.45dB, SSIM: 0.912
```

---

## 🔧 高级使用技巧

### 并行监控
```bash
# 开启多个终端同时监控
# 终端1: 监控数据生成
tmux attach -t linefuse_data

# 终端2: 监控训练
tmux attach -t linefuse_train

# 终端3: 实时日志
tail -f training_logs/stage1_unet_easy.log

# 终端4: 系统监控
watch nvidia-smi
```

### 自定义配置
```bash
# 修改数据集大小
vim auto_dataset_generator.sh
# 更改: STAGE1_SAMPLES=5000  # 更小的测试集

# 修改训练参数
vim auto_train_pipeline.sh
# 更改: --epochs 20  # 更快的测试训练
```

### 故障恢复
```bash
# 如果数据生成中断
tmux kill-session -t linefuse_data
./auto_dataset_generator.sh

# 如果训练中断
tmux kill-session -t linefuse_train
./auto_train_pipeline.sh  # 会从检查点自动恢复

# 清理所有任务
tmux kill-server  # 杀死所有tmux会话
```

---

## ⏱️ 时间规划

### 完整流程时间表:
```
数据转换:       30分钟
数据集生成:     2-4小时 (取决于硬件)
Stage 1训练:    1-2天 (U-Net)
Stage 2训练:    5-7天 (扩散模型)
总时间:        ~7-10天
```

### 阶段性验证:
```
Day 1:     数据转换 + 数据集生成完成
Day 2-3:   Stage 1 U-Net训练完成，验证基线性能
Day 4-10:  Stage 2扩散模型训练，追求SOTA性能
```

---

## 🚀 快速开始命令

### 一键启动完整流程:
```bash
# 1. 数据转换
cd augmented_results && python convert_data.py && cd ..

# 2. 启动数据生成（后台）
./auto_dataset_generator.sh

# 3. 等数据生成完成后，启动训练（后台）
# (可以通过 ./monitor_progress.sh 检查数据生成完成)
./auto_train_pipeline.sh

# 4. 监控进度
./monitor_progress.sh
```

### 测试流程 (小规模):
```bash
# 修改为小规模测试
sed -i 's/STAGE1_SAMPLES=8000/STAGE1_SAMPLES=100/' auto_dataset_generator.sh
sed -i 's/STAGE2_SAMPLES=25000/STAGE2_SAMPLES=500/' auto_dataset_generator.sh

# 运行测试
./auto_dataset_generator.sh
```

---

## 📊 预期结果

### 数据集规模:
- **原始光谱**: 29,150 条真实光谱数据
- **Stage 1数据集**: ~24,000 张训练图像
- **Stage 2数据集**: ~75,000 张训练图像
- **总计**: ~99,150 张高质量训练图像

### 模型性能目标:
- **U-Net Baseline**: PSNR > 25dB, SSIM > 0.85
- **扩散模型**: PSNR > 30dB, SSIM > 0.90
- **光谱特征保持**: 峰值位置误差 < 2nm

---

## ⚠️ 注意事项

### 硬件要求:
- **内存**: 32GB+ 推荐 (大数据集处理)
- **GPU**: RTX 4090级别推荐 (扩散模型训练)
- **存储**: 100GB+ 可用空间

### 常见问题:
1. **内存不足**: 减少batch_size或samples数量
2. **磁盘空间**: 定期清理中间文件
3. **训练中断**: tmux会话会保持，可以重新attach
4. **性能监控**: 使用nvidia-smi监控GPU利用率

### 最佳实践:
- 在stable的环境下运行长时间任务
- 定期备份trained_models目录
- 监控磁盘空间避免中断
- 使用tmux保持会话持久性

---

*本指南涵盖了LineFuse自动化训练的完整流程，按步骤执行即可实现从真实数据到生产模型的端到端自动化。*