import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Tuple
import numpy as np


class LayerNorm(nn.Module):
    """
    LayerNorm 模块 - 用于 ConvNeXt
    
    对最后一个维度（通道维度）进行归一化
    输入格式: (B, H, W, C)
    """
    
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.weight = None
            self.bias = None
    
    def forward(self, x):
        """
        前向传播
        输入: (B, H, W, C) 格式
        """
        return F.layer_norm(x, self.normalized_shape, 
                          self.weight, self.bias, self.eps)

class StochasticDepth(nn.Module):
    """
    随机深度模块，在训练时随机跳过某些层
    
    参数：
        drop_prob: 丢弃概率（0-1 之间）
        scale_by_keep: 是否按保留比例缩放
    """
    def __init__(self, drop_prob: float = 0.1, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """
        前向传播
        
        输入：
            x: 任意形状的张量
            
        工作原理：
            训练时：按概率丢弃，其余缩放
            推理时：直接返回
        """
        if not self.training or self.drop_prob == 0.0:
            return x
        
        # 生成随机掩码，形状为 (batch_size, 1, 1, 1)
        # 这样可以在空间维度上保持一致的丢弃
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.empty(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.bernoulli_(keep_prob)
        
        if self.scale_by_keep:
            random_tensor = random_tensor / keep_prob
        
        return x * random_tensor


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt 基本模块，实现现代化卷积块设计
    
    参数：
        dim: 输入通道数
        layer_scale: 层缩放因子初始值
        stochastic_depth_prob: 随机深度概率
    
    内部结构：
        输入 → DW Conv 7×7 → LayerNorm → 
        1×1 Conv (上采样) → GELU → 
        1×1 Conv (下采样) → 残差 + 随机深度
    """
    def __init__(
        self, 
        dim: int,
        layer_scale: float = 1e-6,
        stochastic_depth_prob: float = 0.0
    ):
        super().__init__()
        
        # 深度卷积：7×7 的深度卷积，保持空间维度
        self.dwconv = nn.Conv2d(
            dim, dim, 
            kernel_size=7, 
            padding=3, 
            groups=dim,
            bias=True
        )
        
        # LayerNorm：基于通道维度的归一化
        self.norm = LayerNorm(dim, eps=1e-6)
        
        # 点卷积上采样：将通道数扩展 4 倍
        self.pwconv1 = nn.Linear(dim, 4 * dim, bias=True)
        
        # GELU 激活函数
        self.act = nn.GELU()
        
        # 点卷积下采样：将通道数恢复
        self.pwconv2 = nn.Linear(4 * dim, dim, bias=True)
        
        # 层缩放参数：可学习的缩放因子
        self.gamma = nn.Parameter(
            layer_scale * torch.ones(dim),
            requires_grad=True
        ) if layer_scale > 0 else None
        
        # 随机深度
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob)

    def forward(self, x):
        """
        前向传播
        
        输入：
            x: 形状为 (B, C, H, W) 的张量
            
        输出：
            y: 形状为 (B, C, H, W) 的张量
        """
        # 保存残差连接
        residual = x
        
        # 深度卷积
        x = self.dwconv(x)
        
        # 转换为 (B, H, W, C) 以便应用 LayerNorm
        x = x.permute(0, 2, 3, 1)
        
        # LayerNorm 和点卷积操作
        x = self.norm(x)
        x = self.pwconv1(x)  # 上采样：C → 4C
        x = self.act(x)       # GELU 激活
        x = self.pwconv2(x)   # 下采样：4C → C
        
        # 应用层缩放参数（如果有的话）
        if self.gamma is not None:
            x = self.gamma * x
        
        # 转换回 (B, C, H, W) 格式
        x = x.permute(0, 3, 1, 2)
        
        # 随机深度和残差连接
        x = residual + self.stochastic_depth(x)
        
        return x


class ConvNeXtStem(nn.Module):
    """
    ConvNeXt Stem 模块，用于图像的初始处理
    
    参数：
        in_channels: 输入通道数（RGB 图像为 3）
        out_channels: 输出通道数
        
    设计特点：
        使用 4×4 卷积替代传统的 7×7 + MaxPool
        一步到位完成 4 倍下采样
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 96):
        super().__init__()
        
        # 4×4 卷积，步长为 4，完成 4 倍下采样
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size=4,
            stride=4,
            bias=True
        )
        
        # LayerNorm 用于特征归一化
        self.norm = LayerNorm(out_channels, eps=1e-6)

    def forward(self, x):
        """
        前向传播
        
        输入：
            x: 形状为 (B, 3, H, W) 的图像张量
            
        输出：
            y: 形状为 (B, C, H//4, W//4) 的特征张量
        """
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNeXtStage(nn.Module):
    """
    ConvNeXt 阶段模块，由多个 ConvNeXt Block 组成
    
    参数：
        dim: 通道数
        depth: 这个阶段的 Block 数量
        stochastic_depth_probs: 每个 Block 的随机深度概率列表
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        stochastic_depth_probs: List[float]
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(
                dim=dim,
                stochastic_depth_prob=stochastic_depth_probs[i]
            )
            for i in range(depth)
        ])

    def forward(self, x):
        """
        前向传播，依次通过所有 Block
        """
        for block in self.blocks:
            x = block(x)
        return x


class ConvNeXtDownsample(nn.Module):
    """
    下采样模块，用于阶段之间的过渡
    
    参数：
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # LayerNorm 用于特征归一化
        self.norm = LayerNorm(in_channels, eps=1e-6)
        
        # 2×2 卷积完成下采样和通道调整
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            bias=True
        )

    def forward(self, x):
        """
        前向传播
        
        输入：
            x: 形状为 (B, C_in, H, W)
            
        输出：
            y: 形状为 (B, C_out, H//2, W//2)
        """
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        return x


class ConvNeXt(nn.Module):
    """
    完整的 ConvNeXt 网络架构
    
    参数：
        in_channels: 输入通道数（默认 3 为 RGB 图像）
        num_classes: 分类类别数（不用于检测任务）
        depths: 每个阶段的 Block 数量列表 [3, 3, 27, 3]
        dims: 每个阶段的通道数列表 [96, 192, 384, 768]
        drop_path_rate: 随机深度的最大概率
        layer_scale_init_value: 层缩放的初始值
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        depths: List[int] = None,
        dims: List[int] = None,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        
        if depths is None:
            depths = [3, 3, 27, 3]  # ConvNeXt-Small 配置
        if dims is None:
            dims = [96, 192, 384, 768]

        # Stem 模块
        self.stem = ConvNeXtStem(in_channels, dims[0])
        
        # 计算每个 Block 的随机深度概率
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # 构建 4 个阶段
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        cur = 0
        for i in range(4):
            stage = ConvNeXtStage(
                dim=dims[i],
                depth=depths[i],
                stochastic_depth_probs=dp_rates[cur:cur + depths[i]]
            )
            self.stages.append(stage)
            cur += depths[i]
            
            # 添加下采样层（除了最后一个阶段后不添加）
            if i < 3:
                downsample = ConvNeXtDownsample(dims[i], dims[i + 1])
                self.downsamples.append(downsample)
        
        # 分类头（用于分类任务，检测任务中不使用）
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(dims[-1], num_classes)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """
        前向传播，返回多尺度特征（用于检测任务）
        
        返回：
            features: 字典，包含不同尺度的特征
            {
                'stage1': (B, C1, H/4, W/4),
                'stage2': (B, C2, H/8, W/8),
                'stage3': (B, C3, H/16, W/16),
                'stage4': (B, C4, H/32, W/32)
            }
        """
        features = {}
        
        # Stem 模块
        x = self.stem(x)
        
        # 通过 4 个阶段，收集中间特征
        for i in range(4):
            x = self.stages[i](x)
            features[f'stage{i+1}'] = x
            
            # 下采样到下一个阶段
            if i < 3:
                x = self.downsamples[i](x)
        
        return features

    def forward(self, x):
        """
        前向传播（分类模式）
        
        输入：
            x: 形状为 (B, 3, 224, 224) 的图像张量
            
        输出：
            logits: 形状为 (B, num_classes) 的分类 logits
        """
        features = self.forward_features(x)
        
        # 使用最后一个阶段的特征进行分类
        x = features['stage4']
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


def convnext_tiny(**kwargs):
    """ConvNeXt-Tiny 模型"""
    model = ConvNeXt(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        **kwargs
    )
    return model


def convnext_small(**kwargs):
    """ConvNeXt-Small 模型（推荐用于 YOLOv11）"""
    model = ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        **kwargs
    )
    return model


def convnext_base(**kwargs):
    """ConvNeXt-Base 模型"""
    model = ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        **kwargs
    )
    return model


# ============================================================================
# 代码解析
# ============================================================================
"""
ConvNeXt 基础模块的核心要点：

1. StochasticDepth 模块：
   - 随机丢弃整个残差连接
   - 保持比例缩放，防止数值变化
   - 形状保留技巧：(B, 1, 1, 1) 与 (B, C, H, W) 相乘会自动广播
   
2. ConvNeXtBlock：
   - 使用深度卷积进行空间处理（7×7）
   - 使用点卷积进行通道处理（反向瓶颈）
   - LayerNorm 替代 BatchNorm
   - 层缩放因子提升学习稳定性
   
3. 特征维度变化：
   (B, C, H, W)
      ↓ dwconv 7×7 (groups=C)
   (B, C, H, W) - 深度卷积，通道不变
      ↓ norm + permute
   (B, H, W, C) - 准备应用线性层
      ↓ pwconv1 (C → 4C)
   (B, H, W, 4C) - 上采样
      ↓ act + pwconv2 (4C → C)
   (B, H, W, C) - 下采样
      ↓ permute back
   (B, C, H, W) - 恢复原始格式

4. Stem 模块优化：
   原始 ResNet: 7×7 Conv(stride=2) + BN + ReLU + 3×3 MaxPool(stride=2)
              = 两次下采样，参数众多
   
   ConvNeXt: 4×4 Conv(stride=4) + LayerNorm
           = 一次下采样，参数更少
           
   参数对比：
   ResNet Stem: 7*7*3*64 + 64 (BN) + 0 (MaxPool) = 9408 参数
   ConvNeXt Stem: 4*4*3*96 = 4608 参数
   节省比例：51%

5. 多尺度特征提取：
   forward_features() 方法返回字典，包含 4 个不同分辨率的特征：
   - stage1: 最高分辨率，适合检测小物体
   - stage2: 中分辨率
   - stage3: 低分辨率
   - stage4: 最低分辨率，适合检测大物体
"""


# ============================================================================
# ConvNeXt Backbone for YOLO
# ============================================================================

class ConvNeXtBackboneForYOLO(nn.Module):
    """
    为 YOLOv11 定制的 ConvNeXt 骨干网络
    
    设计目标：
    1. 输出三个多尺度特征：P3 (1/8), P4 (1/16), P5 (1/32)
    2. 与 YOLOv11 检测头兼容
    3. 支持特征通道自适应
    
    参数：
        model_type: 'tiny', 'small', 'base' 等
        pretrained: 是否加载预训练权重
        output_channels: 输出通道数 [p3_c, p4_c, p5_c]
    """
    def __init__(
        self,
        model_type: str = 'small',
        pretrained: bool = False,
        output_channels: List[int] = None,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        
        # 默认输出通道
        if output_channels is None:
            output_channels = [128, 256, 512]
        
        # 创建 ConvNeXt 主体
        if model_type == 'tiny':
            self.backbone = convnext_tiny(
                drop_path_rate=drop_path_rate,
                num_classes=1000
            )
            backbone_channels = [96, 192, 384, 768]
        elif model_type == 'small':
            self.backbone = convnext_small(
                drop_path_rate=drop_path_rate,
                num_classes=1000
            )
            backbone_channels = [96, 192, 384, 768]
        elif model_type == 'base':
            self.backbone = convnext_base(
                drop_path_rate=drop_path_rate,
                num_classes=1000
            )
            backbone_channels = [128, 256, 512, 1024]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 特征通道适配器：将 ConvNeXt 输出映射到 YOLOv11 期望的通道数
        # stage2 -> P3 (80×80)
        self.adapter_p3 = nn.Sequential(
            nn.Conv2d(backbone_channels[1], output_channels[0], 1),
            nn.BatchNorm2d(output_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # stage3 -> P4 (40×40)
        self.adapter_p4 = nn.Sequential(
            nn.Conv2d(backbone_channels[2], output_channels[1], 1),
            nn.BatchNorm2d(output_channels[1]),
            nn.ReLU(inplace=True)
        )
        
        # stage4 -> P5 (20×20)
        self.adapter_p5 = nn.Sequential(
            nn.Conv2d(backbone_channels[3], output_channels[2], 1),
            nn.BatchNorm2d(output_channels[2]),
            nn.ReLU(inplace=True)
        )
        
        self.output_channels = output_channels

    def forward(self, x):
        """
        前向传播，返回三个多尺度特征
        
        输入：
            x: 形状为 (B, 3, 640, 640) 的图像张量
            
        输出：
            features: 包含 P3, P4, P5 的列表
            [
                P3 (B, C, 80, 80),
                P4 (B, C, 40, 40),
                P5 (B, C, 20, 20)
            ]
        """
        # 获取 ConvNeXt 的多尺度特征
        features = self.backbone.forward_features(x)
        
        # 特征尺度对应关系：
        # 输入 640×640
        # stage1 输出：640/4 = 160×160
        # stage2 输出：640/8 = 80×80 (P3)
        # stage3 输出：640/16 = 40×40 (P4)
        # stage4 输出：640/32 = 20×20 (P5)
        
        p3 = self.adapter_p3(features['stage2'])  # 80×80
        p4 = self.adapter_p4(features['stage3'])  # 40×40
        p5 = self.adapter_p5(features['stage4'])  # 20×20
        
        return [p3, p4, p5]
