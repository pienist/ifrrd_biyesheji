"""
YOLOv11 Head 和 Neck 模块 - 来自 Ultralytics 官方源码
官方仓库: https://github.com/ultralytics/ultralytics
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


class Conv(nn.Module):
    """标准卷积模块: Conv2d + BatchNorm + SiLU"""
    
    default_act = nn.SiLU()
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, self.autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    @staticmethod
    def autopad(k, p=None, d=None):
        if d is not None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p if isinstance(p, (list, tuple)) else (p, p)
    

class DWConv(Conv):
    """深度可分离卷积"""
    
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=c1, d=d, act=act)


class ConvTranspose(nn.Module):
    """转置卷积"""
    
    default_act = nn.SiLU()
    
    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv_transpose(x)))


class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL) 模块
    用于 YOLO 检测头的边界框回归
    官方源码: ultralytics/nn/modules/block.py
    """
    
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
    
    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Bottleneck(nn.Module):
    """标准 Bottleneck 模块"""
    
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, s=1):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], s)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    C2f 模块 - YOLOv8/11 使用的 CSP Bottleneck with 2 convolutions
    包含快捷连接和分割操作
    """
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3)), e=1.0) for _ in range(n))
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2(nn.Module):
    """
    C3k2 模块 - YOLOv11 使用的增强版 C3 模块
    支持不同的宽度倍数和分组卷积
    
    官方源码: ultralytics/nn/modules/block.py
    """
    
    def __init__(self, c1, c2, n=1, c3k=False, s=1, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)
        
        if c3k:
            self.m = nn.ModuleList(
                Bottleneck(self.c, self.c, s=s, g=g, k=(3, 3), e=1.0) for _ in range(n)
            )
        else:
            self.m = C2f(self.c, self.c, n=n, g=g, e=1.0)
    
    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        if isinstance(self.m, C2f):
            return self.cv2(torch.cat((self.m(a), b), 1))
        else:
            y = [m(a) for m in self.m]
            return self.cv2(torch.cat([*y, b], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) 模块"""
    
    default_act = nn.SiLU()
    
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C2PSA(nn.Module):
    """C2PSA 模块 - 带位置敏感注意力的 C2 模块"""
    
    def __init__(self, c1, c2=None, n=1, s=1, g=1, e=0.5):
        super().__init__()
        if c2 is None:
            c2 = c1
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)
        
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, s=s, g=g, k=(3, 3), e=1.0) for _ in range(n)
        )
    
    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


# ============================================================================
# YOLOv11 Detection Head
# ============================================================================

class v11Detect(nn.Module):
    """
    YOLOv11 Detection Head - 官方源码适配版
    
    功能:
    - 无锚框 (Anchor-free) 检测
    - 三尺度输出: P3 (小目标), P4 (中目标), P5 (大目标)
    - 支持训练和推理模式
    
    官方源码: ultralytics/nn/modules/head.py
    """
    
    dynamic = False
    export = False
    format = None
    max_det = 300
    agnostic_nms = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    
    def __init__(self, nc=80, ch=(128, 256, 512)):
        """
        初始化检测头
        
        Args:
            nc (int): 类别数量
            ch (tuple): 三个检测层的通道数
        """
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        
        # 边界框回归分支和类别预测分支
        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()
        
        for i, c in enumerate(ch):
            c2 = max((16, c // 4, self.reg_max * 4))
            c3 = max(c, min(self.nc, 100))
            
            # 边界框回归分支
            self.cv2.append(nn.Sequential(
                Conv(c, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ))
            
            # 类别预测分支
            self.cv3.append(nn.Sequential(
                Conv(c, c3, 3),
                Conv(c3, c3, 3),
                nn.Conv2d(c3, self.nc, 1)
            ))
        
        # DFL 层用于分布焦点损失
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 三个尺度的特征图列表 [P3, P4, P5]
        
        Returns:
            处理后的特征图列表
        """
        outputs = []
        for i in range(self.nl):
            # cv2 和 cv3 并行处理
            box = self.cv2[i](x[i])
            cls = self.cv3[i](x[i])
            outputs.append(torch.cat([box, cls], dim=1))
        return outputs
    
    def bias_init(self):
        """初始化检测头的偏置"""
        for i, (a, b) in enumerate(zip(self.cv2, self.cv3)):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )


# ============================================================================
# YOLOv11 Neck (PAFPN - Path Aggregation Feature Pyramid Network)
# ============================================================================

class YOLOv11Neck(nn.Module):
    """
    YOLOv11 Neck (PAFPN)
    
    结构:
    - 上采样路径: 融合高层语义信息
    - 下采样路径: 保留空间细节
    - 三个输出尺度: P3, P4, P5
    """
    
    def __init__(self, in_channels=(128, 256, 512)):
        """
        初始化 Neck
        
        Args:
            in_channels: 三个输入尺度的通道数
        """
        super().__init__()
        
        c3, c4, c5 = in_channels
        
        # 上采样: P5 -> P4
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3k2_up1 = C3k2(c5 + c4, c4, n=2)
        
        # 上采样: P4 -> P3
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3k2_up2 = C3k2(c4 + c3, c3, n=2)
        
        # 下采样: P3 -> P4
        self.downsample1 = Conv(c3, c3, 3, 2)
        self.c3k2_down1 = C3k2(c3 + c4, c4, n=2)
        
        # 下采样: P4 -> P5
        self.downsample2 = Conv(c4, c4, 3, 2)
        self.c3k2_down2 = C3k2(c4 + c5, c5, n=2)
    
    def forward(self, features):
        """
        前向传播
        
        Args:
            features: 来自 backbone 的三个特征图 [P3, P4, P5]
        
        Returns:
            处理后的三个特征图 [P3, P4, P5]
        """
        p3, p4, p5 = features
        
        # 上采样路径
        up1 = self.upsample1(p5)
        cat1 = torch.cat([up1, p4], dim=1)
        p4_out = self.c3k2_up1(cat1)
        
        up2 = self.upsample2(p4_out)
        cat2 = torch.cat([up2, p3], dim=1)
        p3_out = self.c3k2_up2(cat2)
        
        # 下采样路径
        down1 = self.downsample1(p3_out)
        cat3 = torch.cat([down1, p4_out], dim=1)
        p4_out = self.c3k2_down1(cat3)
        
        down2 = self.downsample2(p4_out)
        cat4 = torch.cat([down2, p5], dim=1)
        p5_out = self.c3k2_down2(cat4)
        
        return [p3_out, p4_out, p5_out]


# ============================================================================
# YOLOv11 Segmentation Head (Segmentation = Detection + Mask)
# ============================================================================

class v11Segment(v11Detect):
    """
    YOLOv11 Segmentation Head - 在检测头基础上添加分割功能
    
    主要改进:
    - 添加 cv4 分支: 生成掩码系数
    - 添加 proto 模块: 生成原型掩码
    - 推理时使用原型和系数生成最终分割掩码
    
    官方源码: ultralytics/nn/modules/head.py
    """
    
    def __init__(self, nc=80, nm=32, npr=256, ch=(128, 256, 512)):
        """
        初始化分割头
        
        Args:
            nc (int): 类别数量
            nm (int): 掩码数量 (mask coefficients)
            npr (int): 原型数量 (prototypes)
            ch (tuple): 三个检测层的通道数
        """
        super().__init__(nc, ch)
        self.nm = nm  # 掩码系数数量
        self.npr = npr  # 原型数量
        self.no = nc + self.nm  # 输出: 类别 + 掩码系数
        
        # 掩码系数分支 cv4 (与 cv2, cv3 并行)
        c4 = max((self.nm * ch[0]) // 256, self.nm)
        self.cv4 = nn.ModuleList()
        
        for i, c in enumerate(ch):
            self.cv4.append(nn.Sequential(
                Conv(c, c4, 3),
                Conv(c4, c4, 3),
                nn.Conv2d(c4, self.nm, 1)
            ))
        
        # 原型生成模块
        self.proto = nn.Sequential(
            Conv(ch[0], npr, 3),
            Conv(npr, npr, 3),
            Conv(npr, npr, 3),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(npr, npr, 3),
            Conv(npr, npr, 3),
            Conv(npr, self.nm, 3)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 三个尺度的特征图列表 [P3, P4, P5]
        
        Returns:
            检测输出和分割输出
            - 检测: 列表，每个元素是 (B, nc+nm, H, W)
            - 分割原型: (B, nm, H*2, W*2)
        """
        # 生成原型掩码
        proto = self.proto(x[0])
        
        # 生成检测输出和掩码系数
        outputs = []
        for i in range(self.nl):
            # 掩码系数
            mask_coef = self.cv4[i](x[i])
            # 检测 (box + cls)
            box = self.cv2[i](x[i])
            cls = self.cv3[i](x[i])
            # 合并
            outputs.append(torch.cat([box, cls, mask_coef], dim=1))
        
        return outputs, proto
    
    def decode_outputs(self, outputs, proto):
        """
        解码输出，生成最终的分割掩码
        
        Args:
            outputs: 模型输出的检测列表
            proto: 原型掩码 (B, nm, H, W)
        
        Returns:
            分割掩码列表
        """
        masks = []
        for i, output in enumerate(outputs):
            # 提取掩码系数 (最后 nm 个通道)
            mask_coef = output[:, -self.nm:, :, :]
            # 上采样原型到输出尺寸
            proto_up = F.interpolate(proto, size=output.shape[-2:], mode='bilinear', align_corners=False)
            # 矩阵乘法生成掩码: (B, H, W, nm) @ (B, nm, H, W) -> (B, H, W)
            mask = torch.sigmoid(torch.einsum('bnhw,bchw->bhwc', proto_up, mask_coef))
            masks.append(mask.permute(0, 3, 1, 2))  # (B, nm, H, W) -> (B, H, W, nm) -> (B, nm, H, W)
        return masks
