"""YOLOv11-ConvNeXt 完整模型定义

官方源码适配来源:
- YOLOv11 Head/Neck: https://github.com/ultralytics/ultralytics
- ConvNeXt Backbone: 自定义实现
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from .convnext_backbone import ConvNeXtBackboneForYOLO
from .yolov11_head import YOLOv11Neck, v11Detect, v11Segment


class YOLOv11WithConvNeXt(nn.Module):
    """
    集成 ConvNeXt 骨干网络的 YOLOv11 完整模型
    
    架构：
        ConvNeXt Backbone → YOLOv11 Neck → YOLOv11 Detect Head
    
    官方源码适配:
    - Backbone: ConvNeXt (models/convnext_backbone.py)
    - Neck: YOLOv11 PAFPN (models/yolov11_head.py)
    - Head: v11Detect (models/yolov11_head.py)
    """
    
    def __init__(
        self,
        convnext_model_type: str = 'small',
        num_classes: int = 80,
        drop_path_rate: float = 0.1,
        in_channels: List[int] = None
    ):
        super().__init__()
        
        if in_channels is None:
            in_channels = [128, 256, 512]
        
        self.num_classes = num_classes
        
        # ConvNeXt 骨干网络
        self.backbone = ConvNeXtBackboneForYOLO(
            model_type=convnext_model_type,
            output_channels=in_channels,
            drop_path_rate=drop_path_rate
        )
        
        # YOLOv11 Neck (PAFPN)
        self.neck = YOLOv11Neck(in_channels=in_channels)
        
        # YOLOv11 Detection Head
        self.head = v11Detect(nc=num_classes, ch=in_channels)
        
        # 初始化头部偏置
        self.head.stride = torch.tensor([8, 16, 32])
        self.head.bias_init()
    
    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)
        features = self.neck(features)
        predictions = self.head(features)
        return predictions


class YOLOv11SegWithConvNeXt(nn.Module):
    """
    集成 ConvNeXt 骨干网络的 YOLOv11-Seg 分割模型
    
    架构：
        ConvNeXt Backbone → YOLOv11 Neck → YOLOv11 Segment Head
    
    支持:
    - 目标检测
    - 实例分割
    
    官方源码适配:
    - Backbone: ConvNeXt (models/convnext_backbone.py)
    - Neck: YOLOv11 PAFPN (models/yolov11_head.py)
    - Head: v11Segment (models/yolov11_head.py)
    """
    
    def __init__(
        self,
        convnext_model_type: str = 'small',
        num_classes: int = 80,
        num_masks: int = 32,
        drop_path_rate: float = 0.1,
        in_channels: List[int] = None,
        proto_channels: int = 256
    ):
        super().__init__()
        
        if in_channels is None:
            in_channels = [128, 256, 512]
        
        self.num_classes = num_classes
        self.num_masks = num_masks
        
        # ConvNeXt 骨干网络
        self.backbone = ConvNeXtBackboneForYOLO(
            model_type=convnext_model_type,
            output_channels=in_channels,
            drop_path_rate=drop_path_rate
        )
        
        # YOLOv11 Neck (PAFPN)
        self.neck = YOLOv11Neck(in_channels=in_channels)
        
        # YOLOv11 Segmentation Head
        self.head = v11Segment(
            nc=num_classes, 
            nm=num_masks, 
            npr=proto_channels,
            ch=in_channels
        )
        
        # 初始化头部偏置
        self.head.stride = torch.tensor([8, 16, 32])
        self.head.bias_init()
    
    def forward(self, x) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, 3, H, W)
        
        Returns:
            outputs: 检测和分割输出列表 [(B, nc+nm, H, W), ...]
            proto: 原型掩码 (B, nm, H*2, W*2)
        """
        features = self.backbone(x)
        features = self.neck(features)
        outputs, proto = self.head(features)
        return outputs, proto
    
    def decode(self, outputs, proto, mask_threshold: float = 0.5):
        """
        解码输出，生成最终的分割掩码
        
        Args:
            outputs: 模型输出
            proto: 原型掩码
            mask_threshold: 掩码二值化阈值
        
        Returns:
            分割掩码列表
        """
        masks = self.head.decode_outputs(outputs, proto)
        return [torch.where(m > mask_threshold, 1.0, 0.0) for m in masks]


def build_yolov11_convnext(model_type='small', num_classes=80, drop_path_rate=0.1):
    """构建 YOLOv11-ConvNeXt 模型的工厂函数"""
    return YOLOv11WithConvNeXt(
        convnext_model_type=model_type,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate
    )


def build_yolov11_seg_convnext(model_type='small', num_classes=80, num_masks=32, drop_path_rate=0.1):
    """构建 YOLOv11-Seg-ConvNeXt 分割模型的工厂函数"""
    return YOLOv11SegWithConvNeXt(
        convnext_model_type=model_type,
        num_classes=num_classes,
        num_masks=num_masks,
        drop_path_rate=drop_path_rate
    )


# ============================================================================
# 代码解析
# ============================================================================
"""
YOLOv11 集成适配的关键设计：

1. ConvNeXtBackboneForYOLO 类职责：
   - 承载完整的 ConvNeXt 网络
   - 提取多尺度特征（stage2, stage3, stage4）
   - 通过 adapter 层调整通道维度
   - 输出标准的三层特征金字塔 [P3, P4, P5]

2. 特征尺度映射（640×640 输入）：
   ConvNeXt Stem: 640 ÷ 4 = 160
   Stage1: 160 (无下采样)
   Downsample1: 160 ÷ 2 = 80 (对应 P3)
   Stage2: 80
   Downsample2: 80 ÷ 2 = 40 (对应 P4)
   Stage3: 40
   Downsample3: 40 ÷ 2 = 20 (对应 P5)
   Stage4: 20

3. 通道适配器设计：
   ConvNeXt-Small 输出：[96, 192, 384, 768]
   YOLOv11 期望：[128, 256, 512]
   
   需要的映射：
   adapter_p3: 192 → 128
   adapter_p4: 384 → 256
   adapter_p5: 768 → 512

4. 适配器结构的选择：
   Conv1×1 + BatchNorm + ReLU
   
   为什么这样设计？
   - Conv1×1: 纯粹的通道线性变换，不改变空间维度
   - BatchNorm: 归一化特征分布，稳定训练
   - ReLU: 引入非线性，增强特征表达能力
   
   可选方案：
   ① Conv1×1 + LayerNorm（与 ConvNeXt 一致）
   ② Conv1×1 + InstanceNorm（对 batch size 不敏感）
   ③ Conv1×1 仅此（最轻量，适合边缘设备）

5. 为什么不用 Stem 的 stage1？
   - stage1 (160×160) 分辨率太高
   - YOLOv11 通常处理 80、40、20 三个尺度
   - 160×160 会增加计算量但改善有限
   - stage2 (80×80) 才是检测有用的最高分辨率
"""
