项目目标
- 构建一个 基于扩散模型（Diffusion）的图像去模糊系统。
- 输入：经过多种模糊退化处理的光谱折线图。
- 输出：清晰、锐利、保持物理特征的折线图。
- 应用场景：科研数据预处理、光谱分析自动化、OCR/特征提取前的增强。
工程内容

---
1. 数据准备（1人）
整体人力/时间：
- 1 人，2 周（调参 + 批量生成）。
1.1 清晰图像生成
- 输入：CSV 光谱数据。
- 输出：统一风格的干净折线图 (PNG)。
技术细节：
- 使用 matplotlib 绘制折线图。
- 统一参数：
  - 分辨率：512x512
  - 背景：白色/灰色单色
  - 线条颜色：黑色
  - 坐标轴字体：Arial，字号固定
- 保存：plt.savefig("xxx.png", dpi=150, bbox_inches="tight")

---
1.2 模糊图像生成
- 输入：清晰图像。
- 输出：多样化模糊图。
模糊类型与实现：
1. 高斯模糊 / 运动模糊
  - cv2.GaussianBlur(img, (k, k), sigma)
  - cv2.filter2D(img, -1, motion_kernel)
2. 压缩伪影
  - cv2.imwrite("xxx.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
  - q 在 10–50 随机。
3. 打印扫描模拟
  - 几何变换：albumentations.ShiftScaleRotate
  - 噪声：albumentations.GaussNoise
  - 亮度对比：albumentations.RandomBrightnessContrast
4. 低分辨率 → 放大
  - cv2.resize(img, (64,64)) → cv2.resize(..., (512,512))
5. 数字/线条干扰
  - 用 cv2.putText() 在折线上随机叠字母/数字。
  - cv2.line() 添加细线条。

---
1.3 数据集组织
- 结构：
dataset/
  train/
    clean/xxx.png
    blur/xxx.png
  val/
    clean/yyy.png
    blur/yyy.png
  test/
    clean/zzz.png
    blur/zzz.png
- 数量：建议 ≥10,000 对清晰-模糊图像。一周内先跑通数据制作流程，两周完成数据集
- 划分：80% 训练 / 10% 验证 / 10% 测试。

---
2. 模型开发
2.1 模型选择
- Baseline：U-Net 去模糊网络。
- 正式方案：条件扩散模型（Conditional Diffusion）。
技术栈：
- 深度学习框架：PyTorch
- Diffusion 库：huggingface diffusers
- 训练框架：PyTorch Lightning 或 DeepSpeed
- GPU：显存 ≥24GB
关键问题：
- 模糊图作为条件输入 → 与噪声图拼接进入 U-Net。
- 控制去模糊过程的噪声步数，避免过度锐化。
人力/时间：
- 1 人，1 周（baseline + diffusion 原型）。和数据准备并行

---
2.2 模型训练
训练配置：
- Batch size: 16–32（视显存）
- 学习率：1e-4，CosineAnnealingLR 调度
- Iteration: 100k–500k
- 数据增强：旋转 ±5°、缩放 ±10%、随机裁剪
损失函数：
- L1/L2 Loss（像素差异）
- Perceptual Loss (VGG feature loss)
- Contrastive Loss（增强清晰-模糊差异）
优化技巧：
- Mixed precision (FP16)
- Gradient accumulation
- EMA (Exponential Moving Average) 模型参数更新
人力/时间：
- 1 人，3–5 周（含调参）。

---
2.3 模型验证
指标：
- PSNR (skimage.metrics.peak_signal_noise_ratio)
- SSIM (skimage.metrics.structural_similarity)
- 光谱特征指标：
  - 峰值位置误差 (peak shift)
  - 峰面积误差 (integral error)
对照：
- 模糊输入 vs 输出 vs 清晰 GT
- Baseline 模型 vs Diffusion 模型
人力/时间：
- 1 人，1 周。

---
3. 系统集成
3.1 api搭建
封装模型，使得后续工作可以直接调用模型
- 输入：模糊光谱图 (PNG)
- 输出：清晰光谱图 (PNG)
实现细节：
- 导出模型：
  - torch.jit.trace → TorchScript
  - 或 torch.onnx.export → ONNX
- 部署：
  - Web API: FastAPI
  - Batch 处理：CLI 脚本
人力/时间：
- 1 人，2 周。

---
3.2 可视化界面（暂时不考虑）
- 需求：科研人员能直观比较模糊 vs 输出 vs GT。
- 技术栈：
  - 快速界面：streamlit / Gradio
  - 绘图：matplotlib、plotly
- 功能：上传模糊图 → 展示恢复结果 → 可下载。
人力/时间：
- 1 人，1 周。

---
6. 项目时间线（3–4 人团队）
暂时无法在飞书文档外展示此内容

---
7. 关键风险与对策
暂时无法在飞书文档外展示此内容

---
总结
- 项目核心：多样化模糊数据生成 + 条件扩散模型训练。
- 技术栈：PyTorch + diffusers + opencv + albumentations + FastAPI/Gradio。
- 时间：3–4 个月，3–4 人团队。
- 最大挑战：模糊多样性 与 科学指标对齐。
