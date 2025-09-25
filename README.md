# LineFuse

🌟 **基于扩散模型的图像去模糊系统**

LineFuse是一个专为处理光谱折线图多种模糊退化而设计的深度学习项目，采用先进的扩散模型技术实现高质量图像去模糊。

## 🚀 快速开始

### 🔥 自动化训练流程 (推荐)
**完全自动化的端到端训练流程，支持29,150条真实光谱数据**

```bash
# 1. 数据转换 (30分钟)
cd augmented_results && python convert_data.py && cd ..

# 2. 自动生成训练数据集 (2-4小时，后台运行)
./auto_dataset_generator.sh

# 3. 自动化两阶段训练 (7-10天，后台运行)
./auto_train_pipeline.sh

# 4. 随时监控进度
./monitor_progress.sh
```

📖 **详细使用指南**: [自动化脚本完整文档](docs/AUTOMATION_GUIDE.md)

---

### 🛠️ 手动安装和使用

#### 环境要求
- Python 3.10+
- CUDA 12.1+ (GPU训练)
- conda虚拟环境管理

#### 安装步骤
```bash
# 克隆项目
git clone <your-repo-url>
cd LineFuse

# 安装环境
./setup_env.sh

# 激活环境
conda activate linefuse

# 快速演示
python main.py demo
```

## 📁 项目结构

```
LineFuse/
├── main.py                # 🚀 主程序入口
├── src/                   # 核心源代码
│   ├── data/             # 数据处理模块
│   ├── models/           # 模型实现
│   ├── api/              # API服务
│   └── utils/            # 工具函数
├── docs/                 # 项目文档
├── configs/              # 配置文件
└── setup_env.sh          # 环境安装脚本
```

## 🎯 主要功能

### ✅ 数据生成管道
- **清晰图表生成**: 从CSV数据生成标准化光谱图
- **12种模糊效果**: 高斯、运动、噪声、形态学变化等
- **🆕 图表样式多样化**: 20种样式组合，支持真实扫描文档风格
- **自动数据集构建**: train/val/test划分和验证
- **29,150条真实光谱**: 支持大规模真实数据训练

### ✅ 自动化训练系统
- **🔥 完全自动化流程**: 从数据转换到模型训练的端到端自动化
- **两阶段训练策略**: U-Net基线 → 扩散模型渐进训练
- **tmux后台管理**: 长时间训练任务的可靠后台运行
- **实时进度监控**: 训练状态、GPU使用率、日志监控

### ✅ 深度学习模型
- **U-Net基线模型**: 传统CNN去模糊方法 (已完成)
- **条件扩散模型**: SOTA扩散模型技术 (已完成)
- **多指标评估**: PSNR、SSIM、光谱特征保持度
- **渐进式训练**: easy → medium → hard 难度递增

### 🌐 API服务 (已实现)
- **FastAPI接口**: RESTful API服务
- **实时处理**: 高效的图像去模糊推理
- **批量处理**: 支持大规模数据处理

## 📚 文档导航

| 文档 | 描述 | 推荐度 |
|------|------|--------|
| [🔥 自动化脚本指南](docs/AUTOMATION_GUIDE.md) | **完整自动化训练流程** | ⭐⭐⭐ |
| [图表样式多样化](docs/CHART_STYLE_DIVERSITY.md) | 20种图表样式，支持真实扫描文档 | ⭐⭐⭐ |
| [快速开始指南](docs/QUICK_START.md) | 详细的安装和使用教程 | ⭐⭐ |
| [项目进度](docs/project-progress.md) | 完整的开发进度和技术细节 | ⭐⭐ |
| [数据生成报告](docs/DATA_GENERATION_SUMMARY.md) | 数据生成功能的详细说明 | ⭐ |
| [增强功能介绍](docs/ENHANCED_BLUR_FEATURES.md) | 12种模糊效果的技术实现 | ⭐ |

## 🧪 使用方法

### 快速演示 (10个样本)
```bash
python main.py demo
```

### 生成自定义数据集
```bash
python main.py generate --samples 100 --output my_dataset
```

### 训练模型 (开发中)
```bash
python main.py train --dataset my_dataset --model unet
```

### 查看所有选项
```bash
python main.py --help
```

## 🛠️ 开发状态

| 功能模块 | 状态 | 完成度 |
|---------|------|--------|
| 数据生成管道 | ✅ 完成 | 100% |
| 代码规范化 | ✅ 完成 | 100% |
| 基线模型训练 | 🔄 开发中 | 0% |
| 扩散模型实现 | 🔄 开发中 | 0% |
| API服务部署 | ⏳ 计划中 | 0% |

**总进度: 35% 完成**

## 📊 技术栈

- **深度学习**: PyTorch 2.1.0, diffusers
- **计算机视觉**: OpenCV, albumentations
- **数据处理**: NumPy, Matplotlib, Pillow
- **Web服务**: FastAPI, uvicorn
- **开发工具**: conda, pytest

## 🤝 贡献指南

1. Fork项目并创建功能分支
2. 遵循项目代码规范 (参见[项目进度文档](docs/project-progress.md))
3. 运行测试确保功能正常: `./run_tests.sh`
4. 提交Pull Request

## 📝 许可证

MIT License

## 📧 联系方式

如有问题或建议，请通过Issue或邮件联系。

---
*LineFuse - 让模糊的光谱图像重新清晰* ✨