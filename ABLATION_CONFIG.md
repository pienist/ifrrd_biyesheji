# 消融实验配置方案

## 实验配置总览

### 关键设计原则

```
┌─────────────────────────────────────────────────────────┐
│                    配置设计原则                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. 公平性: 所有实验使用相同的训练策略，只改变模型/损失   │
│                                                          │
│  2. 必要性: 红外小目标数据少(4k)，必须两阶段训练        │
│                                                          │
│  3. 效率性: 消融实验需要多次运行，配置要合理            │
│                                                          │
│  4. 可比性: 冻结/解冻配置保持一致，只改变模块           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 训练策略配置

### 方案选择：两阶段训练（推荐）

| 方案 | 冻结 | 解冻 | 推荐度 | 说明 |
|------|------|------|--------|------|
| **两阶段** | 30-50 epochs | 100-120 epochs | ⭐⭐⭐⭐⭐ | 适合红外小目标，防止过拟合 |
| 单阶段 | 无 | 150 epochs | ⭐⭐⭐ | 快但不稳定 |
| 长冻结 | 80 epochs | 80 epochs | ⭐⭐ | 冻结太长可能欠适配 |

### 推荐的两阶段配置

```
阶段1 (冻结): 30 epochs
├── 冻结: backbone (前5层)
├── 学习率: 1e-3 → 1e-4 (冻结) | 5e-4 → 5e-5 (解冻)
├── 训练部分: Neck + Head + 新增模块
└── 目的: 让新增模块快速收敛

阶段2 (解冻): 90 epochs
├── 解冻: 全部
├── 学习率: 5e-4 → 5e-5 (cosine)
├── 训练部分: 全部参数
└── 目的: 整体微调到红外任务
```

---

## 各实验具体配置

### 实验A: Baseline（基准）

```yaml
# 实验A: Baseline
experiment_name: "A_baseline"
description: "ConvNeXt-Base基准，无任何改进"

# 模型
backbone: "convnext_base_in22k_ft_in1k.pth"
freeze_layers: 5

# 训练策略（两阶段）
stage1:
  epochs: 30
  lr0: 0.001
  lrf: 0.1
  
stage2:
  epochs: 90
  lr0: 0.0005
  lrf: 0.1

# 数据增强（原始配置）
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0

# 损失函数
loss: "BCE Only"
```

### 实验B: 通道适配器 (Channel Adapter)

```yaml
# 实验B: 通道适配器
experiment_name: "B_channel_adapter"
description: "1→3通道卷积适配器"

# 模型
backbone: "convnext_base_in22k_ft_in1k.pth"
freeze_layers: 5
new_module:
  - "channel_adapter"  # 新增：1x1卷积 1→3通道

# 训练策略（与Baseline一致）
stage1:
  epochs: 30
  lr0: 0.001
  lrf: 0.1
  # 特别注意：通道适配器需要更多学习率来适应
  # 可以给新增模块稍高的学习率
  
stage2:
  epochs: 90
  lr0: 0.0005
  lrf: 0.1

# 数据增强（原始配置）
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0

# 损失函数
loss: "BCE Only"
```

### 实验C: CLAHE红外增强

```yaml
# 实验C: CLAHE增强
experiment_name: "C_clahe"
description: "CLAHE对比度增强"

# 模型
backbone: "convnext_base_in22k_ft_in1k.pth"
freeze_layers: 5

# 训练策略
stage1:
  epochs: 30
  lr0: 0.001
  lrf: 0.1
  
stage2:
  epochs: 90
  lr0: 0.0005
  lrf: 0.1

# 数据增强（关键改进）
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0
  # 新增CLAHE增强
  clahe:
    enabled: true
    clip_limit: 2.0
    tile_grid_size: [8, 8]

# 损失函数
loss: "BCE Only"
```

### 实验D: 边缘感知损失

```yaml
# 实验D: 边缘感知损失
experiment_name: "D_edge_loss"
description: "Sobel边缘损失"

# 模型
backbone: "convnext_base_in22k_ft_in1k.pth"
freeze_layers: 5

# 训练策略
stage1:
  epochs: 30
  lr0: 0.001
  lrf: 0.1
  
stage2:
  epochs: 90
  lr0: 0.0005
  lrf: 0.1

# 数据增强（原始配置）
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0

# 损失函数（关键改进）
loss:
  type: "BCE + Edge"
  bce_weight: 1.0
  edge_weight: 0.1
  edge_type: "sobel"  # sobel / canny / laplacian
```

### 实验E: 纹理感知损失

```yaml
# 实验E: 纹理感知损失
experiment_name: "E_texture_loss"
description: "分块方差纹理损失"

# 模型
backbone: "convnext_base_in22k_ft_in1k.pth"
freeze_layers: 5

# 训练策略
stage1:
  epochs: 30
  lr0: 0.001
  lrf: 0.1
  
stage2:
  epochs: 90
  lr0: 0.0005
  lrf: 0.1

# 数据增强
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0

# 损失函数
loss:
  type: "BCE + Texture"
  bce_weight: 1.0
  texture_weight: 0.05
  patch_size: 16
```

### 实验F: 通道适配器 + CLAHE (B+C)

```yaml
# 实验F: 通道适配器 + CLAHE
experiment_name: "F_channel_clahe"
description: "通道适配 + CLAHE增强"

# 模型
backbone: "convnext_base_in22k_ft_in1k.pth"
freeze_layers: 5
new_module:
  - "channel_adapter"

# 训练策略
stage1:
  epochs: 30
  lr0: 0.001
  lrf: 0.1
  
stage2:
  epochs: 90
  lr0: 0.0005
  lrf: 0.1

# 数据增强
augmentation:
  clahe:
    enabled: true
    clip_limit: 2.0
    tile_grid_size: [8, 8]
  # 其他保持原始

# 损失函数
loss: "BCE Only"
```

### 实验G: 通道适配器 + CLAHE + 边缘损失 (B+C+D)

```yaml
# 实验G: 通道适配 + CLAHE + 边缘损失
experiment_name: "G_full_minus_texture"
description: "去除纹理损失的次优组合"

# 模型
backbone: "convnext_base_in22k_ft_in1k.pth"
freeze_layers: 5
new_module:
  - "channel_adapter"

# 训练策略
stage1:
  epochs: 30
  lr0: 0.001
  lrf: 0.1
  
stage2:
  epochs: 90
  lr0: 0.0005
  lrf: 0.1

# 数据增强
augmentation:
  clahe:
    enabled: true
    clip_limit: 2.0
    tile_grid_size: [8, 8]

# 损失函数
loss:
  type: "BCE + Edge"
  bce_weight: 1.0
  edge_weight: 0.1
```

### 实验H: 全量组合 (B+C+D+E)

```yaml
# 实验H: 全量组合
experiment_name: "H_full"
description: "通道适配 + CLAHE + 边缘 + 纹理"

# 模型
backbone: "convnext_base_in22k_ft_in1k.pth"
freeze_layers: 5
new_module:
  - "channel_adapter"

# 训练策略
stage1:
  epochs: 30
  lr0: 0.001
  lrf: 0.1
  
stage2:
  epochs: 90
  lr0: 0.0005
  lrf: 0.1

# 数据增强
augmentation:
  clahe:
    enabled: true
    clip_limit: 2.0
    tile_grid_size: [8, 8]

# 损失函数
loss:
  type: "BCE + Edge + Texture"
  bce_weight: 1.0
  edge_weight: 0.1
  texture_weight: 0.05
  patch_size: 16
```

---

## 学习率配置详解

### 为什么这样设置学习率？

```
┌─────────────────────────────────────────────────────────┐
│                    学习率策略                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  阶段1 (冻结): lr0 = 1e-3                               │
│  ├── 理由: Neck和Head是随机初始化，需要大学习率         │
│  ├── 目的: 快速收敛到合理范围                           │
│  └── cos_lr衰减: 1e-3 → 1e-4                           │
│                                                          │
│  阶段2 (解冻): lr0 = 5e-4                               │
│  ├── 理由: backbone是预训练权重，需要小学习率微调        │
│  ├── 目的: 保护预训练知识，避免破坏                     │
│  └── cos_lr衰减: 5e-4 → 5e-5                           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 学习率调整建议

| 情况 | 调整建议 |
|------|----------|
| **损失不收敛** | 提高lr0 1.5-2倍 |
| **损失震荡** | 降低lr0 0.5倍 |
| **过拟合** | 降低lr0，增加weight_decay |
| **欠拟合** | 提高lr0，增加warmup |

### 各实验学习率一致性

```
重要: 所有8个实验使用相同的学习率配置！

原因:
1. 公平对比 - 只有被测试的模块不同
2. 避免学习率作为变量影响结果
3. 便于定位哪个模块真正有效

例外情况:
- 如果实验B/C/D/E中某个效果很差，
  可以尝试给新增模块单独调大学习率
```

---

## Epoch配置详解

### 为什么是 30 + 90 = 120 epochs？

```
┌─────────────────────────────────────────────────────────┐
│                    Epoch配置                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  阶段1 (30 epochs):                                      │
│  ├── 目的: 训练新增模块（Neck, Head, Channel Adapter）  │
│  ├── 时间: 约30%总训练时间                               │
│  └── 判断标准: val loss稳定下降                         │
│                                                          │
│  阶段2 (90 epochs):                                     │
│  ├── 目的: 整体微调                                      │
│  ├── 时间: 约75%总训练时间                               │
│  └── 判断标准: val mAP达到最优                          │
│                                                          │
│  总计: 120 epochs                                        │
│  ├── 单GPU训练时间: 约15-20小时                          │
│  └── 8个实验总时间: 约5-7天                              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Epoch调整建议

| 数据集大小 | 阶段1 | 阶段2 | 总计 |
|-----------|-------|-------|------|
| < 2k | 50 | 150 | 200 |
| 2k - 5k | **30** | **90** | **120** |
| 5k - 10k | 20 | 80 | 100 |
| > 10k | 10 | 50 | 60 |

---

## 训练命令模板

### 基础命令

```bash
# 实验A: Baseline
python train_convnext_seg.py \
    --epochs 120 \
    --batch 24 \
    --device 2 \
    --freeze 5 \
    --freeze_epochs 30 \
    --name "A_baseline"

# 实验B: 通道适配器
python train_convnext_seg.py \
    --epochs 120 \
    --batch 8 \
    --device 0 \
    --freeze 5 \
    --freeze_epochs 30 \
    --name "B_channel_adapter" \
    --channel_adapter

# 实验C: CLAHE
python train_convnext_seg.py \
    --epochs 120 \
    --batch 8 \
    --device 0 \
    --freeze 5 \
    --freeze_epochs 30 \
    --name "C_clahe" \
    --clahe
```

### 完整命令示例

```bash
#!/bin/bash
# 消融实验脚本

# 实验A: Baseline
python train_convnext_seg.py \
    --epochs 120 \
    --batch 8 \
    --device 0 \
    --freeze 5 \
    --freeze_epochs 30 \
    --name "A_baseline"

# 实验B: 通道适配器
python train_convnext_seg.py \
    --epochs 120 \
    --batch 8 \
    --device 0 \
    --freeze 5 \
    --freeze_epochs 30 \
    --name "B_channel_adapter" \
    --channel_adapter

# 实验C: CLAHE
python train_convnext_seg.py \
    --epochs 120 \
    --batch 8 \
    --device 0 \
    --freeze 5 \
    --freeze_epochs 30 \
    --name "C_clahe" \
    --clahe

# 实验D: 边缘损失
python train_convnext_seg.py \
    --epochs 120 \
    --batch 8 \
    --device 0 \
    --freeze 5 \
    --freeze_epochs 30 \
    --name "D_edge_loss" \
    --edge_loss

# 实验E: 纹理损失
python train_convnext_seg.py \
    --epochs 120 \
    --batch 8 \
    --device 0 \
    --freeze 5 \
    --freeze_epochs 30 \
    --name "E_texture_loss" \
    --texture_loss

# 实验F: B+C
python train_convnext_seg.py \
    --epochs 120 \
    --batch 8 \
    --device 0 \
    --freeze 5 \
    --freeze_epochs 30 \
    --name "F_channel_clahe" \
    --channel_adapter --clahe

# 实验G: B+C+D
python train_convnext_seg.py \
    --epochs 120 \
    --batch 8 \
    --device 0 \
    --freeze 5 \
    --freeze_epochs 30 \
    --name "G_full_minus_texture" \
    --channel_adapter --clahe --edge_loss

# 实验H: 全量
python train_convnext_seg.py \
    --epochs 120 \
    --batch 8 \
    --device 0 \
    --freeze 5 \
    --freeze_epochs 30 \
    --name "H_full" \
    --channel_adapter --clahe --edge_loss --texture_loss
```

---

## 实验执行顺序建议

```
┌─────────────────────────────────────────────────────────┐
│                   执行顺序建议                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  第1批 (快速验证):                                       │
│  1. 实验A (Baseline) - 建立基准                          │
│  2. 实验B (通道适配) - 单模块验证                        │
│  3. 实验C (CLAHE) - 单模块验证                          │
│                                                          │
│  第2批 (组合验证):                                       │
│  4. 实验F (B+C) - 验证协同效果                          │
│                                                          │
│  第3批 (损失函数验证):                                   │
│  5. 实验D (边缘损失)                                     │
│  6. 实验E (纹理损失)                                     │
│                                                          │
│  第4批 (最终验证):                                       │
│  7. 实验G (B+C+D)                                       │
│  8. 实验H (全量)                                         │
│                                                          │
│  总时间: 约5-7天 (单GPU)                                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 配置检查清单

### 开始实验前检查

```
□ 预训练权重文件存在
  - convnext_base_in22k_ft_in1k.pth

□ 数据集配置正确
  - dataset.yaml
  - 训练集路径
  - 验证集路径

□ 硬件资源
  - GPU显存 >= 16GB (batch=8)
  - 或 batch=4 (10GB+)

□ 输出目录
  - runs/segment/A_baseline/
  - runs/segment/B_channel_adapter/
  - ...

□ 日志保存
  - results.csv
  - args.yaml
  - 权重文件
```

---

## 总结

### 最终配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **冻结Epochs** | 30 | 训练Neck+Head+新增模块 |
| **解冻Epochs** | 90 | 整体微调 |
| **总Epochs** | 120 | 30 + 90 |
| **冻结学习率** | 1e-3 → 1e-4 | cosine衰减 |
| **解冻学习率** | 5e-4 → 5e-5 | cosine衰减 |
| **Batch Size** | 8 | 16GB显存 |
| **冻结层数** | 5 | 前5层 |

### 关键原则

```
1. 所有实验使用相同训练配置
2. 只改变被测试的模块
3. 保证公平对比
4. 记录每个实验的详细配置
```

---

*配置生成时间: 2026-04-19*
