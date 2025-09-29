# Windows数据集生成脚本使用指南

LineFuse项目提供了两个Windows版本的自动数据集生成脚本，替代Linux版本的tmux依赖。

## 📁 脚本文件

### 1. PowerShell版本 (推荐)
**文件**: `auto_dataset_generator.ps1`
**适用于**: Windows 10/11，支持PowerShell 5.0+

### 2. 批处理版本 (兼容性最佳)
**文件**: `auto_dataset_generator.bat`
**适用于**: 所有Windows版本，包括较旧的系统

## 🚀 使用方法

### PowerShell版本使用

```powershell
# 1. 打开PowerShell (以管理员身份推荐)
# 2. 导航到LineFuse目录
cd C:\path\to\LineFuse

# 3. 执行脚本 (使用默认参数)
.\auto_dataset_generator.ps1

# 4. 自定义参数执行
.\auto_dataset_generator.ps1 -Stage1Samples 5000 -Stage2Samples 15000 -BaseOutputDir "my_dataset"
```

### 批处理版本使用

```cmd
REM 1. 打开命令提示符
REM 2. 导航到LineFuse目录
cd C:\path\to\LineFuse

REM 3. 执行脚本
auto_dataset_generator.bat
```

## ⚙️ 配置参数

### PowerShell版本参数
- `Stage1Samples`: Stage 1样本数量 (默认: 8000)
- `Stage2Samples`: Stage 2样本数量 (默认: 25000)
- `BaseOutputDir`: 输出目录 (默认: "dataset\training_dataset")
- `PythonPath`: Python解释器路径 (默认: "/root/miniconda3/envs/linefuse/bin/python")

### 批处理版本配置
编辑`.bat`文件顶部的配置变量：
```batch
set STAGE1_SAMPLES=8000
set STAGE2_SAMPLES=25000
set BASE_OUTPUT_DIR=dataset\training_dataset
set PYTHON_PATH=/root/miniconda3/envs/linefuse/bin/python
```

## 🔄 工作流程

### Stage 1: U-Net基线数据集
- **样本数**: 8000个光谱图
- **生成图像**: 16000张 (8000 clean + 8000 blur)
- **样式多样性**: 0.6 (适中)
- **用途**: U-Net基线模型训练

### Stage 2: 扩散模型数据集
- **样本数**: 25000个光谱图
- **生成图像**: 50000张 (25000 clean + 25000 blur)
- **样式多样性**: 1.0 (完全)
- **用途**: 扩散模型训练

## 📊 执行特性

### PowerShell版本优势
- ✅ **后台任务管理**: 使用PowerShell Jobs，支持进度监控
- ✅ **实时日志**: 自动生成带时间戳的日志文件
- ✅ **状态监控**: 每30秒显示任务状态和日志行数
- ✅ **错误处理**: 完善的错误捕获和退出码检查
- ✅ **数据验证**: 自动验证生成的数据集完整性
- ✅ **磁盘监控**: 显示数据集大小和磁盘使用情况

### 批处理版本特性
- ✅ **兼容性**: 支持所有Windows版本
- ✅ **简单易用**: 双击运行，无需额外依赖
- ✅ **日志记录**: 生成详细的执行日志
- ✅ **数据验证**: Python脚本验证数据集完整性
- ✅ **错误处理**: 基础错误检查和提示

## 📝 日志文件

两个版本都会自动生成日志文件：
```
logs/
├── stage1_generation_20241229_143052.log
├── stage1_generation_20241229_143052.log.error  (仅PowerShell版本)
├── stage2_generation_20241229_145123.log
└── stage2_generation_20241229_145123.log.error  (仅PowerShell版本)
```

## 🎯 数据集输出结构

```
dataset/training_dataset/
├── stage1_unet/
│   └── final_dataset/
│       ├── easy/
│       ├── medium/
│       └── hard/
└── stage2_diffusion/
    └── final_dataset/
        ├── easy/
        ├── medium/
        └── hard/
```

## ⚠️ 注意事项

### 系统要求
- **磁盘空间**: 至少20GB可用空间
- **内存**: 建议8GB以上RAM
- **Python环境**: 确保LineFuse conda环境已激活

### PowerShell执行策略
如果PowerShell脚本无法执行，需要设置执行策略：
```powershell
# 查看当前策略
Get-ExecutionPolicy

# 临时允许脚本执行
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### 路径设置
确保Python路径正确设置：
- Linux/WSL环境: `/root/miniconda3/envs/linefuse/bin/python`
- Windows环境: `C:\Users\YourName\miniconda3\envs\linefuse\python.exe`

## 🐛 故障排除

### 常见问题

1. **Python路径错误**
   - 检查并更新脚本中的`PYTHON_PATH`变量
   - 确保LineFuse环境已正确安装

2. **权限问题**
   - 以管理员身份运行PowerShell/命令提示符
   - 确保输出目录有写入权限

3. **内存不足**
   - 减少样本数量参数
   - 关闭其他占用内存的程序

4. **磁盘空间不足**
   - 检查可用磁盘空间
   - 选择不同的输出目录

### 错误日志检查
- PowerShell版本: 检查`.log.error`文件
- 批处理版本: 检查主日志文件中的错误信息

## 🔄 与Linux版本对比

| 特性 | Linux (tmux) | Windows (PowerShell) | Windows (批处理) |
|------|-------------|---------------------|------------------|
| 后台运行 | ✅ | ✅ | ❌ (阻塞运行) |
| 进度监控 | ✅ | ✅ | ❌ |
| 会话恢复 | ✅ | ❌ | ❌ |
| 日志记录 | ✅ | ✅ | ✅ |
| 错误处理 | ✅ | ✅ | ✅ |
| 兼容性 | Linux Only | Windows 10+ | All Windows |

## ✅ 验证成功标志

脚本执行成功后会显示：
```
✅ 数据集验证通过!
=== 数据集生成完成! ===
Stage 1 数据集: 16000 张图像
Stage 2 数据集: 50000 张图像
总计: 66000 张图像
```

现在你可以在Windows环境中无缝使用LineFuse的自动数据集生成功能！