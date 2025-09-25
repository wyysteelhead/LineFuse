# LineFuse 图表样式多样化系统

## 🎯 概览

LineFuse的图表样式多样化系统为训练数据生成提供了丰富的视觉变化，模拟真实世界中的各种文档样式。特别针对扫描文档风格进行了优化，以匹配目标应用场景。

## 🎨 支持的样式类型

### 1. 背景样式 (Background Styles)

#### `clean_white` - 清洁白色背景
- **用途**: 现代学术论文、数字化图表
- **特征**: 纯白背景，无任何纹理或干扰
- **出现频率**: 25%

#### `graph_paper` - 方格纸背景 ⭐
- **用途**: 手工绘制图表、实验室记录
- **特征**: 规则方格纸图案，模拟实际方格纸
- **关键价值**: 匹配image.png的方格纸特征
- **出现频率**: 25%

#### `aged_paper` - 泛黄纸张
- **用途**: 历史文档、老旧实验记录
- **特征**: 微黄背景色，带有轻微斑点和变色
- **出现频率**: 20%

#### `scan_document` - 扫描文档 ⭐
- **用途**: 扫描的纸质文档
- **特征**: 扫描伪影、轻微色彩偏移、纸张纹理
- **关键价值**: 直接匹配image.png的扫描效果
- **出现频率**: 20%

#### `lab_notebook` - 实验室笔记本
- **用途**: 实验室记录本、工程笔记
- **特征**: 装订线、打孔、横线纸效果
- **出现频率**: 10%

### 2. 网格样式 (Grid Styles)

#### `major_minor` - 主次网格
- **特征**: 主要网格线（粗）+ 次要网格线（细）
- **用途**: 精确的科学测量图表
- **出现频率**: 25%

#### `square_grid` - 方格网格 ⭐
- **特征**: 均匀方格，与graph_paper背景完美配合
- **关键价值**: image.png风格的核心组成部分
- **出现频率**: 30%

#### `no_grid` - 无网格
- **特征**: 完全无网格线，简洁风格
- **用途**: 现代简约设计
- **出现频率**: 15%

#### `custom_grid` - 自定义网格
- **特征**: 轻微不规则的网格，更自然
- **用途**: 手工绘制效果
- **出现频率**: 20%

#### `minimal_grid` - 最简网格
- **特征**: 仅几条关键指导线
- **用途**: 焦点突出的展示
- **出现频率**: 10%

### 3. 坐标轴样式 (Axis Styles)

#### `full_axis` - 完整坐标轴
- **特征**: 完整的轴标签、刻度、单位
- **用途**: 学术论文、正式报告
- **出现频率**: 30%

#### `ticks_only` - 仅刻度
- **特征**: 保留刻度数字，移除轴标签
- **用途**: 简化版科学图表
- **出现频率**: 25%

#### `handwritten` - 手写风格 ⭐
- **特征**: 手写体数字标记，不规则位置
- **关键价值**: 直接匹配image.png的手写数字(40,80,100,120)
- **用途**: 实验室手记、现场记录
- **出现频率**: 20%

#### `minimal` - 最简坐标
- **特征**: 极少的刻度标记
- **用途**: 概念展示、趋势显示
- **出现频率**: 15%

#### `no_axis` - 无坐标轴
- **特征**: 完全隐藏坐标轴信息
- **用途**: 纯数据可视化
- **出现频率**: 10%

### 4. 标注样式 (Annotation Styles)

#### `typed_labels` - 打字标签
- **特征**: 标准打字机字体标签
- **用途**: 正式文档、出版物
- **出现频率**: 40%

#### `handwritten_notes` - 手写笔记 ⭐
- **特征**: 手写体注释、峰值标记、边注
- **关键价值**: 匹配image.png的手写注释风格
- **用途**: 实验记录、分析笔记
- **出现频率**: 25%

#### `measurement_marks` - 测量标记
- **特征**: 技术测量线、数值标注
- **用途**: 工程图纸、精密测量
- **出现频率**: 20%

#### `minimal_text` - 最简文字
- **特征**: 极少的文字信息
- **用途**: 现代简约设计
- **出现频率**: 15%

## 🔧 使用方法

### 1. 基本用法

```python
from src.data.clean_chart_generator import CleanChartGenerator

# 启用样式多样化
generator = CleanChartGenerator(
    enable_style_diversity=True,
    style_diversity_level=1.0  # 完全随机
)

# 生成图表 - 将随机选择样式
generator.process_csv_to_chart("data.csv", "output.png")
```

### 2. 指定特定样式模板

```python
# 专门生成scan_document风格 (匹配image.png)
generator = CleanChartGenerator(
    enable_style_diversity=True,
    target_style="scan_document"
)

# 其他可用模板
templates = [
    "scan_document",    # 扫描文档风格 ⭐
    "academic_paper",   # 学术论文风格
    "lab_notebook",     # 实验室笔记风格
    "field_notes"       # 野外记录风格
]
```

### 3. 控制样式多样性程度

```python
# 低多样性 - 偏向常见样式
generator = CleanChartGenerator(
    enable_style_diversity=True,
    style_diversity_level=0.3
)

# 高多样性 - 完全随机
generator = CleanChartGenerator(
    enable_style_diversity=True,
    style_diversity_level=1.0
)
```

### 4. 在训练流程中使用

```python
# 在main.py中集成
difficulty_generator = CleanChartGenerator(
    line_width=config['line_width'],
    enable_style_diversity=True,
    style_diversity_level=0.8  # 高多样性
)
```

## 📊 样式组合概率

系统使用智能概率分布确保样式多样性：

### 重点样式 (匹配image.png)
- `scan_document` + `square_grid` + `handwritten`: **高概率组合**
- `graph_paper` + `square_grid` + `handwritten_notes`: **高概率组合**

### 其他常用组合
- `clean_white` + `major_minor` + `full_axis`: 学术风格
- `aged_paper` + `minimal_grid` + `handwritten`: 历史文档风格
- `lab_notebook` + `square_grid` + `measurement_marks`: 工程风格

## 🎯 image.png 风格专项支持

特别针对您提供的image.png样式进行了专项优化：

### 核心特征匹配
1. **方格纸背景**: `graph_paper` 背景样式
2. **方格网格**: `square_grid` 网格类型
3. **手写数字**: `handwritten` 坐标轴样式
4. **扫描效果**: `scan_document` 综合效果
5. **泛黄色调**: 轻微暖色背景

### 自动启用方法
```python
# 方法1: 使用scan_document模板
generator = CleanChartGenerator(target_style="scan_document")

# 方法2: 高多样性自动包含 (20-25%概率)
generator = CleanChartGenerator(style_diversity_level=1.0)
```

## 📈 训练数据影响

启用样式多样化将显著提升模型性能：

### 预期改进
- **鲁棒性**: 适应各种真实文档样式
- **泛化能力**: 处理未见过的文档格式
- **实用性**: 直接支持image.png类型的目标应用

### 数据集构成 (建议)
- `scan_document`风格: 20-25%
- 其他真实风格: 60-65%
- 传统清洁风格: 15-20%

## 🚀 快速开始

立即体验样式多样化：

```bash
# 生成带样式多样化的数据集
python main.py generate --samples 50 --output styled_dataset

# 训练时指定高样式多样性
python main.py train --dataset styled_dataset/final_dataset --difficulty medium
```

---

*样式多样化系统专为您的目标应用场景优化，确保生成的训练数据能够处理image.png等真实扫描文档。*