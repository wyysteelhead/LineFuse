# Enhanced Blur Generator Features

## 🎯 新增功能概览

基于你的要求，我已经为 `BlurGenerator` 类添加了5个新的高级模糊效果，使生成的数据更加真实和多样化。

## 🏗️ 实现的功能

### 1. 线条粗细变化 (`random_morphology`)
```python
def random_morphology(self, image, operation_range=(1, 3)) -> np.ndarray
```
- **技术**: `cv2.erode` / `cv2.dilate`
- **效果**: 模拟打印/扫描导致的笔画粗细差异
- **参数**: 可调节形态学操作的核大小
- **应用**: 线条变细（腐蚀）或变粗（膨胀）

### 2. 局部退化 (`local_blur`)
```python
def local_blur(self, image, num_patches_range=(3, 8), patch_size_range=(20, 60)) -> np.ndarray
```
- **技术**: 随机选取小patch，应用强高斯模糊后贴回原图
- **效果**: 部分区域模糊，像扫描件里"局部糊掉"的效果
- **参数**: 可控制patch数量和大小范围
- **真实性**: 模拟扫描仪局部焦点问题

### 3. 阈值化伪影 (`threshold_artifacts`)
```python
def threshold_artifacts(self, image, threshold_range=(100, 180)) -> np.ndarray
```
- **技术**:
  1. `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)` - 转灰度
  2. `cv2.threshold` + 随机阈值 - 二值化
  3. 转回BGR格式
- **效果**: 黑白化→灰度还原，产生边缘锯齿/断裂
- **应用**: 模拟低质量扫描的量化效应

### 4. 椒盐噪声 (`add_salt_pepper`)
```python
def add_salt_pepper(self, image, noise_ratio_range=(0.001, 0.005)) -> np.ndarray
```
- **技术**: 随机坐标设置为0（胡椒）或255（盐）
- **效果**: 打印件灰尘点、小黑点/白点
- **参数**: 可控制噪声密度
- **真实性**: 模拟物理打印扫描的颗粒噪声

### 5. 多重组合退化 (`apply_composite_blur`)
```python
def apply_composite_blur(self, image, num_ops=2) -> Dict[str, Any]
```
- **技术**: 随机选择2-3个现有方法顺序执行
- **效果**: 更真实的复合退化（如：JPEG压缩 + 局部模糊 + 线条粗细变化）
- **智能性**: 自动组合不同效果，生成复杂退化模式

## 📦 更新的类接口

```python
class BlurGenerator:
    # 原有方法
    def gaussian_blur(self, image): ...
    def motion_blur(self, image): ...
    def compression_artifacts(self, image): ...
    def print_scan_simulation(self, image): ...
    def low_resolution_upscale(self, image): ...
    def add_text_interference(self, image): ...
    def add_line_interference(self, image): ...

    # 新增方法 ✨
    def random_morphology(self, image): ...
    def local_blur(self, image): ...
    def threshold_artifacts(self, image): ...
    def add_salt_pepper(self, image): ...
    def apply_composite_blur(self, image, num_ops=2): ...
```

## 🎲 扩展的模糊类型列表

```python
blur_types = [
    # 基础效果
    'gaussian', 'motion', 'compression', 'scan', 'lowres', 'text', 'lines',
    # 新增高级效果
    'morphology', 'localblur', 'threshold', 'saltpepper', 'composite'
]
```

## 🔧 使用示例

### 单一效果测试
```python
from src.data.blur_generator import BlurGenerator

blur_gen = BlurGenerator()

# 测试线条粗细变化
result = blur_gen.apply_random_blur(image, blur_types=['morphology'])

# 测试局部模糊
result = blur_gen.apply_random_blur(image, blur_types=['localblur'])

# 测试组合效果
result = blur_gen.apply_random_blur(image, blur_types=['composite'])
```

### 批量生成增强数据
```python
# 生成包含所有新效果的数据集
blur_gen.batch_generate_blur(
    input_dir='clean_charts/',
    output_dir='enhanced_blur_charts/',
    num_variants_per_image=5  # 每张图生成5个模糊变体
)
```

## 🧪 测试脚本

运行 `test_enhanced_blur.py` 可以测试所有新功能：

```bash
# 在GPU环境下测试（需要cv2和albumentations）
python test_enhanced_blur.py
```

## 🚀 优势

1. **更真实的退化模拟**: 新增效果更贴近真实扫描/打印件
2. **线条质量变化**: morphology解决了"均匀细线"问题
3. **局部复杂度**: local_blur增加空间变化
4. **边缘真实性**: threshold_artifacts产生真实锯齿效果
5. **多层次噪声**: 高斯噪声+椒盐噪声组合
6. **智能组合**: composite自动生成复杂退化模式

## 📋 后续计划

数据生成功能现已完全增强，包含12种不同的模糊效果。现在可以：

1. **切换到GPU环境** 🔄
2. **安装完整依赖** (PyTorch, OpenCV, albumentations)
3. **开始Baseline训练** 🏃‍♂️
4. **实施扩散模型** 🌟

**增强的模糊生成器已就绪！** 🎉