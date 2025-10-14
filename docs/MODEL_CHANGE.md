# 🧩 LineFuse 训练性能与资源利用优化建议报告

## 🎯 一、总体目标

针对当前 **双卡 A6000 (2×49GB)** 训练速度偏慢的情况，本报告提出从 **架构、训练、显存管理与调度** 四个层面系统优化策略，目标是：

| 指标       | 当前状态      | 优化后预期           |
| -------- | --------- | --------------- |
| 单epoch时长 | 100%      | ↓ 50–60%        |
| 显存占用     | 约45–48 GB | ↓ 20–30 GB      |
| 收敛速度     | baseline  | ↑ 1.5–2×        |
| 稳定性      | 稳定        | 保持稳定            |
| 可扩展性     | 中等        | 支持多模糊LoRA模块并行训练 |

---

## ⚙️ 二、模型级优化

### 1️⃣ 引入 **Hybrid LoRA Fine-tuning**（强烈推荐）

#### ✅ 方案概述

* 冻结大部分 UNet 主干，仅在关键层（`mid_block`、`up_blocks[-2:]`）注入 LoRA 模块。
* LoRA rank 设为 8–16，可灵活控制容量与显存占用。
* 保留部分解冻策略（Hybrid），在后 10–20 epochs 联合微调中间层。

#### 🚀 预期收益

| 项目       | 当前       | LoRA优化后    |
| -------- | -------- | ---------- |
| 可训练参数量   | 254M     | 18M        |
| 显存使用     | 48 GB    | 22–25 GB   |
| 单epoch时间 | baseline | 提升 2–3×    |
| 收敛速度     | 标准       | 加快约 1.7×   |
| 性能稳定性    | 稳定       | 稳定（rank≥8） |

#### 💡 衍生优势

* 支持**多模糊域 LoRA 权重独立保存与组合**（例如：Gaussian、Scan、Compression 各自 LoRA）。
* 快速增量训练：1–2小时即可适配新模糊风格。

---

### 2️⃣ 使用 **Parameter-Efficient Fine-tuning (PEFT)** 框架统一管理

建议使用 HuggingFace `peft` 库来封装 LoRA，方便参数冻结与加载管理。

* 支持混合 LoRA + Adapter 模式。
* 自动记录 LoRA 参数差异，便于版本追踪与部署。

---

### 3️⃣ 优化模型调度器与采样步数

* 将 `DDPMScheduler` 替换为 `DDIMScheduler`（去噪收敛更快）。
* 在推理与验证阶段使用 `num_inference_steps=25` 而非 50，性能下降 <2%，速度提升约 2×。

---

## 🧠 三、训练流程与优化策略

### 1️⃣ 渐进式训练阶段（Progressive Curriculum Fine-tuning）

* 采用 **Easy → Medium → Hard** 难度分阶段微调。
* 在每个阶段缩减训练步数（例如 15+20+15 epochs）。
* 每阶段可使用不同 LoRA 子模块或权重初始化，增强适应性。

---

### 2️⃣ 梯度与精度优化

* 启用 `torch.cuda.amp.autocast(dtype=torch.bfloat16)`（bf16 精度）以进一步减少显存压力。
* 结合 `GradScaler()` 动态缩放梯度防止溢出。
* 使用 **梯度累积（gradient accumulation）** 扩大有效 batch size，控制在 GPU 可承受范围内。

---

### 3️⃣ 优化器与调度器调整

| 模块   | 原设置               | 建议调整                          |
| ---- | ----------------- | ----------------------------- |
| 优化器  | AdamW(lr=1e-4)    | AdamW(lr=5e-4)，冻结非LoRA层       |
| 权重衰减 | 0.01              | 0.02（增强正则）                    |
| 调度器  | CosineAnnealingLR | Cosine + Warmup 500 steps     |
| 损失组合 | MSE + L1 + VGG    | 可试加 Charbonnier Loss（更适合模糊边缘） |

---

## 🧮 四、资源调度与并行策略

### 1️⃣ 双卡数据并行 → 混合并行

* 使用 `torch.nn.parallel.DistributedDataParallel (DDP)` 替换 `DataParallel`。

  * 同步 BatchNorm
  * 更高通信效率
* 或在 LoRA-only 模式下启用 **模型并行 + 数据并行**（Pipeline方式）。

### 2️⃣ 激活检查点 (Activation Checkpointing)

* 在 `diffusers` UNet 中插入 checkpoint 分块以节省中间激活缓存。
* 对于 512×512 输入，约可节省 30–40% 显存。

---

### 3️⃣ 数据加载性能优化

* 采用 **WebDataset** 或 **LMDB 数据格式** 加速读取。
* 设置 `prefetch_factor=4`, `num_workers=8`。
* 缓存数据索引（`persistent_workers=True`）减少初始化延迟。

---

## 💾 五、存储与日志优化

* 启用 `DeepSpeed Zero2` 优化策略以实现优化器状态分片。
* 每 N epoch 保存一次权重快照，仅保存 LoRA 差分权重（几 MB 级别）。
* 启用 `tensorboardX` 或 `wandb` 日志追踪主要指标：

  * 训练/验证 MSE、PSNR、SSIM
  * GPU 利用率与内存曲线
  * 不同模糊类型下的恢复性能对比

---

## 🔬 六、整体收益预估

| 优化方向   | 技术手段                        | 收益类型   | 预期改善      |
| ------ | --------------------------- | ------ | --------- |
| 模型参数精简 | LoRA Hybrid Fine-tune       | 显存、速度  | 50–70% 节省 |
| 精度优化   | bf16 + GradScaler           | 稳定性    | 防溢出、收敛更快  |
| 并行机制   | DDP + Checkpoint            | 显存效率   | 再降 30–40% |
| 调度优化   | Warmup Cosine + LoRA rank调节 | 收敛速度   | ↑ 1.5–2×  |
| 数据加载   | LMDB/WebDataset             | I/O效率  | 提升 20–30% |
| 模糊分域训练 | 多LoRA独立模块                   | 模型可扩展性 | 极大提升      |
| 采样策略   | DDIM+25steps                | 推理效率   | 2×加速      |

> **综合预计：**
>
> * 每epoch时长可下降至原来的约 35–45%。
> * 单GPU显存降至 22–26GB（可适配更大batch）。
> * 全训练过程整体加速 2.5–3.5×，推理速度提升约 2×。

---

## 🧭 七、推荐实施顺序

| 优先级    | 优化方向                   | 建议操作      |
| ------ | ---------------------- | --------- |
| 🥇 高   | LoRA Hybrid Fine-tune  | 立即实施      |
| 🥈 高   | bf16 + GradScaler      | 与LoRA并行启用 |
| 🥉 中   | DDP替换DataParallel      | 稳定后迁移     |
| ⚙️ 中   | Checkpointing + Warmup | 训练中期引入    |
| 🔁 可选  | WebDataset / LMDB 数据结构 | 后期批量部署时优化 |
| 💡 研究性 | 多LoRA分域适配              | 拓展性实验阶段实施 |

---

## 🧠 八、结论摘要

* LoRA 是 **当前阶段最具性价比的训练加速路径**，适合你的条件扩散架构。
* 结合混合精度与梯度累积，可在不牺牲性能的情况下显著降低计算成本。
* DDP + Checkpointing 可在后期进一步释放显存，支撑更大 batch 与分布式实验。
* 多 LoRA 适配策略为未来多模糊场景恢复提供结构化扩展基础。

---

## 🛠️ 九、落地实施蓝图

### 1️⃣ 代码分层改造（Week 0-1）
- **LoRA核心模块**：确认 `src/models/lora_adapter.py` 与 `src/models/lora_diffusion.py` 的类接口保持稳定，补充必要的单元验证（参数统计、前向输出尺寸）。
- **Trainer对接**：在 `src/models/trainer.py` 中引入 LoRA 版扩散模型封装，并通过 `use_lora`、`scheduler_type` 等配置开关实现与旧版训练逻辑的并存。
- **配置联动**：更新 `configs/` 下的训练配置文件，新增 `lora_rank`、`lora_alpha`、`module_filters` 等字段，确保命令行与YAML双入口可控。

### 2️⃣ 训练策略渐进（Week 1-2）
- **混合精度落地**：默认启用 `torch.cuda.amp.autocast(dtype=torch.bfloat16)`，并在 Trainer 中封装 GradScaler 逻辑，允许通过配置禁用。
- **阶段式数据切换**：利用现有难度分层目录（`easy/medium/hard`）配置三段 curriculum，使用 `train_schedule` 列表驱动自动切换。
- **LoRA冻结策略**：实现 Hybrid 策略入口，前期仅训练 LoRA，后期通过 `HybridLoRAStrategy` 解冻部分 backbone 层。

### 3️⃣ 资源调度与部署（Week 2-3）
- **DDP/DeepSpeed**：在现有启动脚本中加入 `torchrun`/`deepspeed` 入口，配置 ZeRO-2 时只保存 LoRA 权重差分。
- **监控联动**：整合 GPU 监控方法（`monitor_gpu_usage`）与 `tensorboardX/wandb` 钩子，统一在 `logs/` 目录输出 JSON/CSV 指标。
- **推理加速**：在推理服务（FastAPI）层提供 `num_inference_steps` 动态调节支持，默认 25 步，保留 50 步兜底选项。

---

## 📡 十、验证与监控指标

| 指标类型 | 关键指标 | 说明 |
| --- | --- | --- |
| 训练效率 | `epoch_time`, `samples_per_second` | 记录在 `trainer.py` 的阶段日志中，对比 LoRA 前后加速比 |
| 显存监控 | `memory_allocated_gb`, `memory_usage_percent` | 调用 `monitor_gpu_usage()`，异常高于 85% 触发告警 |
| 质量指标 | `PSNR`, `SSIM`, `CharbonnierLoss` | 每阶段结束在验证集统计，确保收敛不倒退 |
| 模糊域对比 | 各模糊类型 `PSNR` | 结合多 LoRA 权重，验证 Gaussian/Scan/Compression 三域均衡性 |
| 推理性能 | `latency_ms@25steps`, `latency_ms@50steps` | 推理服务压测结果，保证部署 SLA |

> 建议通过 `tensorboardX` + 自定义 JSON Logger 双轨记录，便于自动化回归比对。

---

## ⚠️ 十一、风险与兜底策略

| 风险点 | 触发条件 | 兜底方案 |
| --- | --- | --- |
| LoRA 收敛不稳定 | rank < 8 或 dropout 过大 | 提升 rank 至 12-16，减小 dropout，或暂时回退至全量 fine-tune |
| 显存仍超限 | DeepSpeed/Checkpoint 未生效 | 降低 batch_size 并增加梯度累积；临时禁用解冻层 |
| 质量指标下降 | curriculum 切换阶段震荡 | 延长阶段过渡 epoch，或对 Hard 集只训练 LoRA 不解冻主干 |
| 推理速度未提升 | DDIM 步数仍设置 50 | 下调默认步数为 25，并缓存高频模糊域的 LoRA 合成权重 |
| 代码集成回退 | 旧训练脚本未适配 | 使用 feature flag（如 `TRAINER_MODE=lora_hybrid`）逐步灰度发布 |

---

## ✅ 下一步行动清单

1. 合并 LoRA 模块代码并完成最小可行训练回归（Stage 1 数据集）。
2. 校准混合精度 + 梯度累积配置，记录显存与速度基准。
3. 配置 DDP/DeepSpeed 启动脚本与日志体系，确保监控全链路闭环。
4. 设计多模糊域验证集，完成三域指标对比并输出评估报告。

---

## 🔧 十二、最新代码落地速记

- `main.py:train_model` 增加 `--use-lora`、`--lora-rank`、`--precision`、`--grad-accum`、`--hybrid-unfreeze` 等参数，默认调度器切换为 `DDIM`，并支持 JSONL 训练日志输出。
- `src/models/trainer.py` 引入梯度累积、可选 `HybridLoRAStrategy`、BF16/FP16 精度控制与 per-epoch `samples_per_second` 指标收集。
- `src/models/diffusion_model.py` / `configs/config.yaml` 同步支持 `scheduler: ddim` 与 LoRA rank/模块过滤器配置，便于与文档蓝图对齐。
