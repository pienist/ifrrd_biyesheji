# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""ConvNeXt 模块 - 用于将 ImageNet-22k 预训练的 ConvNeXt 作为 YOLO 的 backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

__all__ = ("ConvNeXt", "ConvNeXtStage", "ConvNeXtBackbone")


class ConvNeXtBlock(nn.Module):
    """ConvNeXt 基本块。

    包含深度可分离卷积 + GELU 激活 + MLP 结构，模仿 Transformer 的设计。
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init=1e-6):
        """初始化 ConvNeXt Block。

        Args:
            dim: 输入通道数
            drop_path: 随机深度率
            layer_scale_init: 层缩放初始值
        """
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True) if layer_scale_init > 0 else None
        self.drop_path = nn.Identity()

    def forward(self, x):
        """前向传播。"""
        input_ = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input_ + self.drop_path(x)
        return x


class ConvNeXtStage(nn.Module):
    """ConvNeXt 的一个 Stage，包含多个 ConvNeXtBlock。

    参数格式: [in_channels, out_channels, depth, drop_path_rate]
    由 parse_model 自动从 YAML 配置解析后传递。
    """

    def __init__(self, in_channels, out_channels, depth, drop_path_rate=0.0, layer_scale_init=1e-6):
        """初始化 Stage。

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            depth: Block 数量（必须是整数）
            drop_path_rate: 随机深度率
            layer_scale_init: 层缩放初始值
        """
        super().__init__()

        # 确保 depth 是整数（parse_model 可能传递浮点数）
        depth = int(depth) if isinstance(depth, float) else depth

        # 下采样：只做空间下采样，不做通道转换
        # 注意：通道转换由 parse_model 处理（c1 -> c2）
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            )
        else:
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.stride = 2 if in_channels != out_channels else 1

        # 创建多个 ConvNeXt Block
        blocks = []
        for i in range(depth):
            block_drop_path = drop_path_rate * i / depth if drop_path_rate > 0 else 0
            blocks.append(
                ConvNeXtBlock(
                    out_channels,
                    drop_path=block_drop_path,
                    layer_scale_init=layer_scale_init,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        """前向传播。"""
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class ConvNeXt(nn.Module):
    """ConvNeXt 主干网络。

    用于提取多尺度特征，支持加载 timm 预训练权重。
    """

    def __init__(
        self,
        in_chans: int = 3,
        depths: tuple = (3, 3, 9, 3),
        dims: tuple = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        layer_scale_init: float = 1e-6,
        pretrained: str = None,
    ):
        """初始化 ConvNeXt。

        Args:
            in_chans: 输入通道数
            depths: 每个 Stage 的 Block 数量
            dims: 每个 Stage 的通道数
            drop_path_rate: 随机深度率
            layer_scale_init: 层缩放初始值
            pretrained: 预训练权重路径
        """
        super().__init__()
        self.pretrained = pretrained

        # Stem: 7x7 卷积 + LayerNorm
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6, elementwise_affine=False),
        )

        # 4 个 Stage
        stages = []
        for i in range(4):
            stages.append(
                ConvNeXtStage(
                    in_channels=dims[i - 1] if i > 0 else dims[0],
                    out_channels=dims[i],
                    depth=depths[i],
                    drop_path_rate=drop_path_rate,
                    layer_scale_init=layer_scale_init,
                )
            )
        self.stages = nn.ModuleList(stages)

        # 输出通道数列表
        self.out_channels = dims

        # 加载预训练权重
        if pretrained:
            self.load_pretrained(pretrained)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """初始化权重。"""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_pretrained(self, pretrained_path):
        """加载 timm 预训练权重。"""
        try:
            import timm
            import os

            # 检查是否有 HF_HUB_OFFLINE
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)

            # 尝试加载
            state_dict = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]

            # 转换权重格式
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("stem.1"):  # LayerNorm
                    new_k = k.replace("stem.1", "stem.norm")
                    new_state_dict[new_k] = v
                elif "pwconv1" in k:
                    new_k = k.replace("pwconv1", "mlp.fc1")
                    new_state_dict[new_k] = v
                elif "pwconv2" in k:
                    new_k = k.replace("pwconv2", "mlp.fc2")
                    new_state_dict[new_k] = v
                elif "dwconv" in k:
                    new_k = k.replace("dwconv", "conv_dw")
                    new_state_dict[new_k] = v
                elif k.startswith("stages."):
                    # 处理 stages
                    new_k = k.replace("stages.", "stages.")
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v

            self.load_state_dict(new_state_dict, strict=False)
            print(f"成功加载预训练权重: {pretrained_path}")
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
            print("继续使用随机初始化...")

    def forward(self, x):
        """前向传播，返回多尺度特征。

        Args:
            x: 输入张量 [B, 3, H, W]

        Returns:
            features: [stage2, stage3, stage4] 的特征列表
        """
        x = self.stem(x)  # [B, 96, H/4, W/4]

        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i >= 1:  # 只保留 stage1, stage2, stage3 (索引 1, 2, 3)
                features.append(x)

        return features  # [P3, P4, P5]


class ConvNeXtAdapter(nn.Module):
    """ConvNeXt 到 YOLO 的适配器。

    将 ConvNeXt 的输出特征转换为 YOLO Neck 所需的格式。
    """

    def __init__(self, dims: tuple = (192, 384, 768), out_channels: tuple = (256, 512, 1024)):
        """初始化适配器。

        Args:
            dims: ConvNeXt 各阶段的通道数
            out_channels: YOLO 需要的输出通道数
        """
        super().__init__()

        # 定义适配层，将 ConvNeXt 特征映射到 YOLO 兼容通道
        self.adapters = nn.ModuleList()
        for i, (dim, out_ch) in enumerate(zip(dims, out_channels)):
            self.adapters.append(
                nn.Sequential(
                    nn.Conv2d(dim, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.SiLU(inplace=True),
                )
            )

        self.out_channels = out_channels

    def forward(self, features):
        """前向传播。

        Args:
            features: ConvNeXt 输出的特征列表

        Returns:
            adapted_features: 适配后的特征列表
        """
        return [adapter(feat) for adapter, feat in zip(self.adapters, features)]


def convnext_base(pretrained: str = None, **kwargs):
    """创建 ConvNeXt-Base 模型。

    Args:
        pretrained: 预训练权重路径
        **kwargs: 其他参数

    Returns:
        ConvNeXt 模型
    """
    model = ConvNeXt(
        depths=(3, 3, 27, 3),  # ConvNeXt-Base 的深度配置
        dims=(128, 256, 512, 1024),  # ConvNeXt-Base 的通道配置
        pretrained=pretrained,
        **kwargs,
    )
    return model


def convnext_small(pretrained: str = None, **kwargs):
    """创建 ConvNeXt-Small 模型。"""
    model = ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(96, 192, 384, 768),
        pretrained=pretrained,
        **kwargs,
    )
    return model
