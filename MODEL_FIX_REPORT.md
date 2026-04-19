# ConvNeXt-Seg 模型修复报告

## 🔍 发现的问题

### 核心问题：YAML配置与预训练权重不匹配

| Stage | 预训练权重 (ConvNeXt-Base) | YAML配置 (修复前) | 状态 |
|-------|--------------------------|-------------------|------|
| Stage 0 | 3 blocks, 128ch | 3 blocks, 128ch | ✅ 匹配 |
| Stage 1 | 3 blocks, 256ch | 3 blocks, 256ch | ✅ 匹配 |
| Stage 2 | **27 blocks**, 512ch | **3 blocks**, 512ch | ❌ **不匹配** |
| Stage 3 | 3 blocks, 1024ch | 3 blocks, 1024ch | ✅ 匹配 |

### 影响

- **修复前**: Stage 2 的24个block权重无法加载 (~200MB预训练权重丢失)
- **模型相当于随机初始化**,这就是为什么GPU内存占用小、训练效果差

## ✅ 修复内容

### `ultralytics/cfg/models/convnext/convnext-seg.yaml`

```yaml
# 修复前 (第18行):
- [-1, 1, ConvNeXtStage, [512, 3, 0.0]]  # 3 stride 32

# 修复后:
- [-1, 1, ConvNeXtStage, [512, 27, 0.0]]  # 3 stride 32 (27 blocks 匹配预训练权重!)
```

### 参数变化

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 总参数 | 38.8M | **89.9M** |
| 可加载权重比例 | ~0% | **94.2%** |

## 📊 红外小目标迁移学习方案

### 1. 数据特点分析

| 特点 | 影响 | 迁移学习策略 |
|------|------|-------------|
| **非RGB单通道** | 预训练RGB特征迁移受限 | 考虑灰度预训练权重或设计通道适配层 |
| **目标小** | 浅层特征更重要 | 浅层backbone充分微调,保留小目标检测能力 |
| **背景噪声大** | 需要强特征提取能力 | 使用强backbone + 适当的数据增强 |
| **单一类别** | 分类难度低,分割难度高 | 关注分割精度而非分类头 |

### 2. 推荐迁移学习策略

#### 策略A: 两阶段渐进式微调 (推荐)

```
阶段1: 冻结backbone，训练Neck+Head (50 epochs)
  - 学习率: 1e-3 (快速收敛)
  - 冻结: model.0 - model.4 (backbone全部冻结)

阶段2: 解冻全部，轻量微调 (100-150 epochs)
  - 学习率: 1e-4 (backbone用1e-5)
  - 解冻: 全部解冻
  - 可选: 只解冻浅层 (model.0 - model.2)
```

#### 策略B: 分层解冻

```
Epoch 1-50:   冻结全部 (backbone + neck)
Epoch 51-100: 解冻backbone深层 (model.3, model.4)
Epoch 101-200: 解冻backbone浅层 (model.0 - model.2) + neck
```

#### 策略C: 小目标增强训练

```yaml
# 数据增强配置
mosaic: 1.0        # 保持，增强小目标
mixup: 0.0         # 关闭，避免模糊小目标
copy_paste: 0.1    # 开启，增强小目标样本

# 多尺度训练
multi_scale: 0.5    # 允许图像尺寸变化 ±50%

# 损失函数权重
box: 7.5           # 保持
seg: 1.0           # 可适当提高
```

### 3. ConvNeXt-Base 特有的迁移建议

| 方面 | 建议 |
|------|------|
| **学习率** | backbone用1e-5 ~ 5e-5, head用1e-4 |
| **权重衰减** | 0.05 (ConvNeXt常用值) |
| **Layer Scale** | 保持预训练的1e-6 |
| **DropPath** | 0.0 ~ 0.1 (避免过拟合) |
| **Epochs** | 冻结50 + 解冻150 = 总共200 epochs |

### 4. 关键代码修改 (train_convnext_seg.py)

```python
# 推荐配置
config = {
    # 学习率策略
    "lr0": 0.0001,        # 整体学习率
    "lrf": 0.01,          # 最终学习率比例
    
    # 冻结策略 (两阶段)
    # 阶段1: freeze=5 (冻结backbone)
    # 阶段2: freeze=0 (全部解冻) + lr0=1e-4
    
    # 数据增强
    "mosaic": 1.0,
    "copy_paste": 0.1,
    "hsv_h": 0.015,
    "hsv_s": 0.5,         # 降低饱和度增强 (红外图像)
    "hsv_v": 0.3,         # 降低亮度增强 (红外图像)
}
```

### 5. 红外图像特定优化

```python
# 预处理建议
def preprocess_ir_image(img):
    # 红外图像通常是单通道，需要复制到3通道
    if img.shape[0] == 1:  # 灰度图
        img = img.repeat(3, 1, 1)
    
    # 或者保持灰度但修改backbone输入通道
    # 将ConvNeXt的stem从3通道改为1通道
    
    return img
```

## 🚀 下一步操作

1. **重新训练** (修复后):
   ```bash
   python train_convnext_seg.py --epochs 200 --batch 8 --freeze 5 --freeze_epochs 50
   ```

2. **预期改进**:
   - GPU显存占用增加 (~2-3倍)
   - 训练时间增加
   - 性能预期大幅提升

3. **验证**:
   - 观察日志中的 `model: 89.9M` 参数
   - 确认预训练权重加载比例 > 90%
