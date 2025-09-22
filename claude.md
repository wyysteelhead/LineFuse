# LineFuse - 扩散模型图像去模糊系统

LineFuse是一个基于扩散模型的图像去模糊系统，专门用于光谱折线图的清晰化处理。

## 项目架构

```
LineFuse/
├── src/
│   ├── data/              # 数据处理模块
│   ├── models/            # 模型定义
│   ├── api/               # API服务
│   └── utils/             # 工具函数
├── configs/               # 配置文件
└── requirements.md        # 项目需求文档
```

## 核心功能

### 1. 数据准备模块 (`src/data/`)

#### 清晰图像生成
```python
from src.data.clean_chart_generator import CleanChartGenerator

generator = CleanChartGenerator(
    figure_size=(512, 512),
    dpi=150,
    background_color='white',
    line_color='black'
)

# 从CSV生成清晰折线图
generator.process_csv_to_chart('data.csv', 'clean_chart.png')

# 批量处理
generator.batch_process('csv_dir/', 'output_dir/')
```

#### 模糊图像生成
```python
from src.data.blur_generator import BlurGenerator

blur_gen = BlurGenerator()

# 应用随机模糊
result = blur_gen.apply_random_blur(image)
blurred_image = result['image']
blur_type = result['blur_type']

# 批量生成模糊变体
blur_gen.batch_generate_blur('clean_dir/', 'blur_dir/', num_variants_per_image=5)
```

支持的模糊类型：
- `gaussian`: 高斯模糊
- `motion`: 运动模糊  
- `compression`: 压缩伪影
- `scan`: 打印扫描模拟
- `lowres`: 低分辨率放大
- `text`: 文字干扰
- `lines`: 线条干扰

#### 数据集构建
```python
from src.data.dataset_builder import DatasetBuilder

builder = DatasetBuilder()

# 创建数据集结构并分割数据
builder.split_data(
    clean_dir='clean_images/',
    blur_dir='blur_images/', 
    output_dir='dataset/',
    split_ratios=(0.8, 0.1, 0.1)  # train/val/test
)

# 验证数据集
stats = builder.validate_dataset('dataset/')
```

### 2. 模型开发模块 (`src/models/`)

#### U-Net基线模型
```python
from src.models.unet_baseline import UNetBaseline

model = UNetBaseline(n_channels=3, n_classes=3)
output = model(blurred_input)
```

#### 条件扩散模型
```python
from src.models.diffusion_model import ConditionalDiffusionModel, DiffusionLoss

# 模型初始化
model = ConditionalDiffusionModel(
    in_channels=3,
    out_channels=3,
    sample_size=512
)

# 训练时的前向传播
noise_pred, noise_target = model(clean_images, blur_images)

# 推理
deblurred = model.inference(blur_images, num_inference_steps=50)

# 损失函数
loss_fn = DiffusionLoss(l1_weight=1.0, l2_weight=0.5, perceptual_weight=0.1)
```

#### 模型训练
```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device='cuda',
    mixed_precision=True
)

# 训练
results = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    save_dir='./models/',
    save_every=10
)
```

### 3. API服务 (`src/api/`)

#### 启动服务器
```bash
cd src/api/
python server.py
```

#### API端点

**健康检查**
```
GET /health
```

**模型管理**
```
POST /models/load?model_type=diffusion&model_path=path/to/model.pth
POST /models/unload?model_type=diffusion
```

**图像去模糊**
```python
import requests
import base64

# 单张图像去模糊
with open('blurred_image.png', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

response = requests.post('http://localhost:8000/deblur', json={
    'image_data': image_data,
    'model_type': 'diffusion',
    'num_inference_steps': 50,
    'guidance_scale': 1.0
})

# 批量处理
response = requests.post('http://localhost:8000/deblur/batch', json={
    'image_paths': ['img1.png', 'img2.png'],
    'output_dir': 'results/',
    'model_type': 'diffusion'
})
```

**数据生成**
```python
# 生成训练数据
response = requests.post('http://localhost:8000/data/generate', json={
    'csv_dir': 'csv_files/',
    'output_dir': 'generated_data/',
    'num_blur_variants': 5
})

# 构建数据集
response = requests.post('http://localhost:8000/data/build-dataset', json={
    'clean_dir': 'clean/',
    'blur_dir': 'blur/',
    'output_dir': 'dataset/',
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1
})
```

### 4. 评估与可视化 (`src/utils/`)

#### 指标计算
```python
from src.utils.metrics import MetricsCalculator

calculator = MetricsCalculator()

# 计算所有指标
metrics = calculator.calculate_all_metrics(
    gt_image=ground_truth,
    pred_image=deblurred,
    blur_image=blurred_input
)

# 批量评估
batch_metrics = calculator.evaluate_batch(
    gt_images=gt_list,
    pred_images=pred_list,
    blur_images=blur_list
)
```

支持的指标：
- 图像质量：PSNR, SSIM, MSE, MAE
- 光谱特征：峰值位置误差, 峰面积误差

#### 可视化
```python
from src.utils.visualization import VisualizationUtils

# 图像对比
VisualizationUtils.compare_images(
    images=[blurred, deblurred, ground_truth],
    titles=['Blurred', 'Deblurred', 'Ground Truth'],
    save_path='comparison.png'
)

# 线条轮廓分析
VisualizationUtils.plot_line_profiles(
    gt_profile=gt_profile,
    pred_profile=pred_profile,
    blur_profile=blur_profile
)

# 训练曲线
VisualizationUtils.plot_training_curves(
    train_losses=train_losses,
    val_losses=val_losses
)
```

## 配置管理

配置文件位于 `configs/config.yaml`，包含：
- 数据处理参数
- 模型架构设置
- 训练超参数
- 推理配置
- 路径设置

## 使用流程

### 1. 数据准备
```python
# 生成清晰图像
python -c "
from src.data.clean_chart_generator import CleanChartGenerator
gen = CleanChartGenerator()
gen.batch_process('csv_data/', 'clean_images/')
"

# 生成模糊数据
python -c "
from src.data.blur_generator import BlurGenerator
blur_gen = BlurGenerator()
blur_gen.batch_generate_blur('clean_images/', 'blur_images/', 5)
"

# 构建数据集
python -c "
from src.data.dataset_builder import DatasetBuilder
builder = DatasetBuilder()
builder.split_data('clean_images/', 'blur_images/', 'dataset/')
"
```

### 2. 模型训练
```python
# 初始化模型和训练器
from src.models.diffusion_model import ConditionalDiffusionModel, DiffusionLoss
from src.models.trainer import ModelTrainer
import torch.optim as optim

model = ConditionalDiffusionModel()
loss_fn = DiffusionLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

trainer = ModelTrainer(model, loss_fn, optimizer)
trainer.train(train_loader, val_loader, num_epochs=100, save_dir='./models/')
```

### 3. 模型推理
```python
# 启动API服务
python src/api/server.py

# 或直接使用模型
model = ConditionalDiffusionModel()
model.load_state_dict(torch.load('model.pth'))
deblurred = model.inference(blur_images)
```

## 技术栈

- **深度学习**: PyTorch, Hugging Face Diffusers
- **图像处理**: OpenCV, PIL, albumentations
- **数据科学**: NumPy, pandas, scikit-image
- **可视化**: Matplotlib, seaborn
- **API服务**: FastAPI, uvicorn
- **配置管理**: YAML

## 依赖安装

```bash
pip install torch torchvision
pip install diffusers transformers
pip install opencv-python albumentations
pip install matplotlib seaborn
pip install fastapi uvicorn
pip install pandas numpy scikit-image
pip install pydantic pyyaml
```

## 开发注意事项

1. **GPU内存**: 扩散模型需要大量显存（≥24GB推荐）
2. **混合精度**: 启用FP16可减少内存使用
3. **批处理**: 根据显存调整batch_size
4. **数据质量**: 确保清晰-模糊图像对的准确匹配
5. **模型保存**: 定期保存checkpoint以防训练中断

## 故障排除

- **CUDA内存不足**: 减少batch_size或启用gradient_accumulation
- **模型加载失败**: 检查模型文件路径和设备匹配
- **API连接超时**: 增加推理的超时时间设置
- **数据格式错误**: 确保图像为RGB格式，像素值0-255