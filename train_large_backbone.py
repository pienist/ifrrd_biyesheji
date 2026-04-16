#!/usr/bin/env python3
"""
YOLO11-seg + 多主干 红外小目标分割训练脚本
=============================================
支持主干: ConvNeXt-S/Base/Large, EfficientNet-B3/B4, Swin-Tiny/Small

使用方法:
    conda activate yolov11_seg
    
    # ConvNeXt-Base (默认)
    python train_large_backbone.py --backbone convnext_base --pretrained convnext_base_in22k_ft_in1k.pth
    
    # ConvNeXt-Small (更小更快)
    python train_large_backbone.py --backbone convnext_small --pretrained convnext_small_in22k_ft_in1k.pth
    
    # ConvNeXt-Large (更大更强, 建议冻结更长)
    python train_large_backbone.py --backbone convnext_large --pretrained convnext_large_in22k_ft_in1k.pth --freeze_epochs 65
    
    # EfficientNet-B3
    python train_large_backbone.py --backbone efficientnet_b3 --pretrained efficientnet_b3.pth --freeze_epochs 35
    
    # Swin-Tiny
    python train_large_backbone.py --backbone swin_tiny --pretrained swin_tiny.pth --freeze_epochs 40
"""

import argparse
import warnings
from datetime import datetime
import os
import torch
import torch.nn as nn
from ultralytics import YOLO

# ============== 主干配置 ==============
BACKBONE_CONFIG = {
    "convnext_small": {
        "name": "ConvNeXt-Small",
        "timm_name": "convnext_small.fb_in22k_ft_in1k",
        "params": "~50M",
        "default_freeze_epochs": 50,
        "weight_url": "https://hf-mirror.com/timm/convnext_small.fb_in22k_ft_in1k/resolve/main/pytorch_model.bin",
    },
    "convnext_base": {
        "name": "ConvNeXt-Base",
        "timm_name": "convnext_base.fb_in22k_ft_in1k",
        "params": "87.6M",
        "default_freeze_epochs": 50,
        "weight_url": "https://hf-mirror.com/timm/convnext_base.fb_in22k_ft_in1k/resolve/main/pytorch_model.bin",
    },
    "convnext_large": {
        "name": "ConvNeXt-Large",
        "timm_name": "convnext_large.fb_in22k_ft_in1k",
        "params": "~198M",
        "default_freeze_epochs": 65,
        "weight_url": "https://hf-mirror.com/timm/convnext_large.fb_in22k_ft_in1k/resolve/main/pytorch_model.bin",
    },
    "efficientnet_b3": {
        "name": "EfficientNet-B3",
        "timm_name": "efficientnet_b3",
        "params": "~12M",
        "default_freeze_epochs": 35,
        "weight_url": "https://hf-mirror.com/timm/efficientnet_b3/resolve/main/pytorch_model.bin",
    },
    "efficientnet_b4": {
        "name": "EfficientNet-B4",
        "timm_name": "efficientnet_b4",
        "params": "~19M",
        "default_freeze_epochs": 40,
        "weight_url": "https://hf-mirror.com/timm/efficientnet_b4/resolve/main/pytorch_model.bin",
    },
    "swin_tiny": {
        "name": "Swin-Tiny",
        "timm_name": "swin_tiny_patch4_window7_224",
        "params": "~28M",
        "default_freeze_epochs": 40,
        "weight_url": "https://hf-mirror.com/timm/swin_tiny_patch4_window7_224/resolve/main/pytorch_model.bin",
    },
    "swin_small": {
        "name": "Swin-Small",
        "timm_name": "swin_small_patch4_window7_224",
        "params": "~50M",
        "default_freeze_epochs": 45,
        "weight_url": "https://hf-mirror.com/timm/swin_small_patch4_window7_224/resolve/main/pytorch_model.bin",
    },
}


def print_backbone_info(backbone: str):
    """打印主干信息"""
    config = BACKBONE_CONFIG[backbone]
    print("=" * 60)
    print(f"主干: {config['name']}")
    print(f"参数量: {config['params']}")
    print(f"timm 模型名: {config['timm_name']}")
    print("=" * 60)


def download_backbone_weights(backbone: str, output_dir: str = "."):
    """下载主干预训练权重"""
    import urllib.request
    
    config = BACKBONE_CONFIG[backbone]
    filename = f"{backbone}_in22k_ft_in1k.pth"
    output_path = os.path.join(output_dir, filename)
    
    if os.path.exists(output_path):
        print(f"权重已存在: {output_path}")
        return output_path
    
    print(f"\n正在下载 {config['name']} 权重...")
    print(f"URL: {config['weight_url']}")
    
    try:
        urllib.request.urlretrieve(config['weight_url'], output_path)
        print(f"下载完成: {output_path}")
        return output_path
    except Exception as e:
        print(f"下载失败: {e}")
        print(f"请手动下载并放置到: {output_path}")
        return None


class MultiBackboneYOLO:
    """多主干 YOLO11-seg 模型"""
    
    def __init__(self, backbone: str, pretrained_path: str = None, device: int = 0):
        self.backbone = backbone
        self.pretrained_path = pretrained_path
        self.device = device
        
        # 加载基础 YOLO11-seg 模型用于获取 head 结构
        print("\n加载 YOLO11-seg 基础模型...")
        base_model = YOLO("yolo11s-seg.pt")
        
        # 保存 base_model 供后续使用
        self._base_model = base_model
        
        # 构建 ConvNeXt + YOLO Hybrid 模型
        self._build_hybrid_model()
    
    def _build_model_with_custom_backbone(self, base_model):
        """构建 ConvNeXt backbone + YOLO head 的 Hybrid 模型"""
        from ultralytics.nn.modules import C3k2, C2PSA, Segment
        from ultralytics.nn.tasks import SegmentationModel
        import timm
        
        config = BACKBONE_CONFIG[self.backbone]
        
        print(f"\n创建 {config['name']} 主干...")
        
        # 创建 timm 主干
        self.encoder = timm.create_model(
            config['timm_name'],
            pretrained=False,
            features_only=True,
            out_indices=[1, 2, 3],  # stride 8/16/32
        )
        
        # 加载预训练权重
        if self.pretrained_path and os.path.exists(self.pretrained_path):
            print(f"加载预训练权重: {self.pretrained_path}")
            state_dict = torch.load(self.pretrained_path, map_location='cpu')
            
            # 处理权重键名转换
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('stages', 'stages_').replace('stem', 'stem_')
                new_state_dict[new_k] = v
            
            missing, unexpected = self.encoder.load_state_dict(new_state_dict, strict=False)
            if missing:
                print(f"  缺失的键 (忽略分类头): {len(missing)} 个")
            if unexpected:
                print(f"  多余的键: {len(unexpected)} 个")
        else:
            print("  未找到预训练权重，使用随机初始化")
        
        # 获取主干输出通道
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 64, 64)
            features = self.encoder(dummy_input)
            encoder_channels = [f.shape[1] for f in features]
        print(f"  主干输出通道: {encoder_channels}")
        
        # 创建适配层
        self._create_adapters(encoder_channels)
        
        # 获取 YOLO head 结构 (从 base_model)
        yolo_head = base_model.model.model[11:]  # head 部分
        
        # 构建 Hybrid 模型
        class HybridSegmentationModel(nn.Module):
            """ConvNeXt + YOLO Head Hybrid 模型"""
            
            def __init__(self, enc, adapt, head, nc=2, base_model=None):
                super().__init__()
                
                # 设置属性
                self.encoder = enc
                self.adapters = adapt
                self.head_layers = nn.ModuleList(list(head))
                self.nc = nc
                
                # 从 base_model 复制必要属性
                if base_model:
                    self.names = {i: str(i) for i in range(nc)}
                    self.yaml = base_model.model.yaml
                    self.save = base_model.model.save
                    self.stride = torch.tensor([8, 16, 32])
                    self.output_stride = 32  # for detection
                    self.yaml = base_model.model.yaml
                    # 复制 Detect 层的信息
                    self.yaml['nc'] = nc
                    
                    # 获取 Segment 层作为最后一层
                    self.last_layer = self.head_layers[-1] if self.head_layers else None

    def forward(self, x):
                # ConvNeXt backbone
        features = self.encoder(x)
                
                # 适配通道
                p3 = self.adapters['p3'](features[0])
                p4 = self.adapters['p4'](features[1])
                p5 = self.adapters['p5'](features[2])
                
                # YOLO head 前向传播
                # 手动实现 head 的前向传播
                y = [p3, p4, p5]
                
                # 从第 11 层开始的 head
                for i, m in enumerate(self.head_layers):
                    if isinstance(m, nn.Upsample):
                        y.append(m(y[-1]))
                    elif hasattr(m, 'f'):
                        # 处理 from 连接
                        if isinstance(m.f, int):
                            x = y[m.f] if m.f >= 0 else y[-1]
                        else:
                            x = [y[j] if j >= 0 else y[-1] for j in m.f]
                        y.append(m(x))
                    else:
                        x = m(x) if isinstance(x, torch.Tensor) else x
                
                return y[-1]  # 返回最后一层输出
        
        # 创建 Hybrid 模型
        self.model.model = HybridSegmentationModel(
            self.encoder, 
            self.adapters,
            yolo_head,
            nc=2,
            base_model=base_model
        )
        
        print("  Hybrid 模型构建完成")
        
        # 统计参数量
        total_params = sum(p.numel() for p in self.model.model.parameters()) / 1e6
        print(f"\n模型总参数量: {total_params:.1f}M")
    
    def _build_model(self):
        """构建带自定义主干的模型"""
        os.environ['HF_HUB_OFFLINE'] = '1'
        import timm
        
        config = BACKBONE_CONFIG[self.backbone]
        
        print(f"\n创建 {config['name']} 主干...")
        
        # 创建 timm 主干
        self.encoder = timm.create_model(
            config['timm_name'],
            pretrained=False,
            features_only=True,
            out_indices=[1, 2, 3],  # stride 8/16/32
        )
        
        # 加载预训练权重
        if self.pretrained_path and os.path.exists(self.pretrained_path):
            print(f"加载预训练权重: {self.pretrained_path}")
            state_dict = torch.load(self.pretrained_path, map_location='cpu')
            
            # 处理权重键名转换 (stem.0 -> stem_0)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('stages', 'stages_').replace('stem', 'stem_')
                new_state_dict[new_k] = v
            
            missing, unexpected = self.encoder.load_state_dict(new_state_dict, strict=False)
            if missing:
                print(f"  缺失的键 (忽略分类头): {len(missing)} 个")
            if unexpected:
                print(f"  多余的键: {len(unexpected)} 个")
        else:
            print("  未找到预训练权重，使用随机初始化")
        
        # 获取主干输出通道
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 64, 64)
            features = self.encoder(dummy_input)
            encoder_channels = [f.shape[1] for f in features]
        print(f"  主干输出通道: {encoder_channels}")
        
        # 创建适配层
        self._create_adapters(encoder_channels)
        
        # 替换 YOLO 主干
        self._replace_backbone()
        
        # 统计参数量
        total_params = sum(p.numel() for p in self.model.model.parameters()) / 1e6
        print(f"\n模型总参数量: {total_params:.1f}M")
    
    def _create_adapters(self, encoder_channels):
        """创建通道适配层 - 将 ConvNeXt 输出适配到 YOLO head 的期望通道"""
        # YOLO11-seg head 期望的通道: P3=256, P4=512, P5=1024
        # ConvNeXt-Base 输出: [256, 512, 1024]
        # 理想情况下直接相等，无需适配
        target_channels = [256, 512, 1024]  # YOLO head 期望
        
        self.adapters = nn.ModuleDict({
            'p3': nn.Conv2d(encoder_channels[0], target_channels[0], 1),
            'p4': nn.Conv2d(encoder_channels[1], target_channels[1], 1),
            'p5': nn.Conv2d(encoder_channels[2], target_channels[2], 1),
        })
        
        print(f"  适配层输出通道: {target_channels}")
    
    def _replace_backbone(self):
        """替换 YOLO 的 backbone"""
        # 获取 YOLO 模型结构
        yolo_model = self.model.model.model
        
        # 保存原始 backbone 结构用于替换
        backbone_blocks = []
        for i in range(7):  # YOLO11 backbone 层
            backbone_blocks.append(yolo_model[i])
        
        # 创建新的 backbone
        class NewBackbone(nn.Module):
            def __init__(self, encoder, adapters, stride=2, dwt=False):
                super().__init__()
                self.encoder = encoder
                self.adapters = adapters
                self.stride = stride
                self.dwt = dwt
                self.f = -1
                self.i = 0
                self.m = None
                self.is_multiscale_backbone = True  # 标记为多尺度 backbone
            
            @property
            def type(self):
                return "MultiBackbone"
            
            @property
            def T_destination(self):
                return None
            
            def forward(self, x):
                # 处理单通道输入
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                
                features = self.encoder(x)
                
                # 适配通道
                if self.adapters is not None:
                    p3 = self.adapters['p3'](features[0])
                    p4 = self.adapters['p4'](features[1])
                    p5 = self.adapters['p5'](features[2])
                else:
                    p3 = features[0]
                    p4 = features[1]
                    p5 = features[2]
                
                # 保存多尺度特征供后续层使用
                self.p3 = p3
                self.p4 = p4
                self.p5 = p5
                
                # 返回 P5 作为主要输出
                return p5
        
        # 替换 backbone
        new_backbone = NewBackbone(self.encoder, self.adapters)
        self.model.model.model[0] = new_backbone
        
        print("  主干替换完成")
    
    def freeze_backbone(self):
        """冻结backbone - 直接从YOLO模型中获取"""
        print("\n冻结 Backbone...")
        # 直接从YOLO模型中获取backbone
        yolo_model = self.model.model.model
        if hasattr(yolo_model[0], 'encoder'):
            for param in yolo_model[0].encoder.parameters():
                param.requires_grad = False
            print("  encoder 已冻结")
        if hasattr(yolo_model[0], 'adapters'):
            for param in yolo_model[0].adapters.parameters():
                param.requires_grad = False
            print("  adapters 已冻结")
        
        frozen = sum(p.numel() for p in self.model.model.parameters() if not p.requires_grad) / 1e6
        trainable = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad) / 1e6
        print(f"  冻结参数: {frozen:.1f}M")
        print(f"  可训练参数: {trainable:.1f}M")
    
    def unfreeze_backbone(self):
        """解冻backbone - 直接从YOLO模型中获取"""
        print("\n解冻 Backbone...")
        # 直接从YOLO模型中获取backbone
        yolo_model = self.model.model.model
        if hasattr(yolo_model[0], 'encoder'):
            for param in yolo_model[0].encoder.parameters():
                param.requires_grad = True
            print("  encoder 已解冻")
        if hasattr(yolo_model[0], 'adapters'):
            for param in yolo_model[0].adapters.parameters():
                param.requires_grad = True
            print("  adapters 已解冻")
        
        trainable = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad) / 1e6
        print(f"  可训练参数: {trainable:.1f}M")
    
    def train(self, freeze_epochs: int, total_epochs: int, skip_phase1: bool = False, phase1_weights: str = None, **config):
        """分阶段训练"""
            print(f"\n{'='*60}")
        print(f"阶段 1: Epoch 0-{freeze_epochs-1} | Backbone 冻结")
        print(f"阶段 2: Epoch {freeze_epochs}-{total_epochs-1} | Backbone 解冻")
            print(f"{'='*60}\n")

        project_name = config.get('project', 'runs')
        base_name = config.get('name', 'exp')
        phase1_name = base_name + '_phase1_frozen'
        phase2_name = base_name + '_phase2_all'
        
        results1 = None
        results2 = None
        phase1_weights = None
        
        # 确定阶段1权重路径
        if skip_phase1:
            # 用户明确要求跳过阶段1
            if phase1_weights and os.path.exists(phase1_weights):
                print(f"\n>>> 跳过阶段 1，加载指定权重: {phase1_weights}")
                self.model = YOLO(phase1_weights)
            else:
                # 自动查找最新的阶段1权重
                search_dir = f"{project_name}/segment/runs"
                if os.path.exists(search_dir):
                    phase1_dirs = [d for d in os.listdir(search_dir) if '_phase1_frozen' in d and os.path.isdir(f"{search_dir}/{d}")]
                    if phase1_dirs:
                        phase1_dirs.sort(reverse=True)
                        for d in phase1_dirs:
                            candidate = f"{search_dir}/{d}/weights/best.pt"
                            if os.path.exists(candidate):
                                print(f"\n>>> 跳过阶段 1，自动找到权重: {candidate}")
                                self.model = YOLO(candidate)
                                phase1_weights = candidate
                                break
                        else:
                            raise FileNotFoundError(f"在 {search_dir} 中未找到任何有效的 best.pt 权重文件")
                    else:
                        raise FileNotFoundError(f"在 {search_dir} 中未找到 _phase1_frozen 目录")
                else:
                    raise FileNotFoundError(f"目录不存在: {search_dir}")
        elif freeze_epochs > 0:
            # 运行阶段1
            print(f"\n>>> 开始阶段 1: 冻结训练 ({freeze_epochs} epochs)")
            self.freeze_backbone()
            
            phase1_config = config.copy()
            phase1_config['epochs'] = freeze_epochs
            phase1_config['name'] = phase1_name
            phase1_config['resume'] = False
            
            results1 = self.model.train(**phase1_config)
            phase1_weights = f"{project_name}/segment/runs/{phase1_name}/weights/best.pt"
            print(f"\n阶段 1 完成，权重保存至: {phase1_weights}")
        
        # 阶段 2: 解冻微调
        phase2_epochs = total_epochs - freeze_epochs
        
        if phase2_epochs <= 0:
            print(f"\n>>> 阶段 2 跳过: 0 epochs (freeze_epochs == total_epochs)")
            print(f">>> 训练完成，总计 {total_epochs} epochs")
            return results1, results2
        
        print(f"\n>>> 开始阶段 2: 解冻微调 ({phase2_epochs} epochs)")
        
        # 保存阶段1的overrides，确保阶段2有完整配置
        original_overrides = self.model.overrides.copy()
        
        self.unfreeze_backbone()
        
        # 确保有model路径
        if phase1_weights and os.path.exists(phase1_weights):
            # 重新加载阶段1权重，fresh start for phase2
            print(f">>> 使用阶段1权重: {phase1_weights}")
            self.model = YOLO(phase1_weights)
            # 更新overrides以包含必要配置
            self.model.overrides.update(original_overrides)
            # 确保model指向阶段1权重
            self.model.overrides['model'] = phase1_weights
        else:
            raise FileNotFoundError(f"阶段1权重不存在: {phase1_weights}")
        
        phase2_config = {
            'data': config.get('data'),
            'task': config.get('task'),
            'batch': config.get('batch'),
            'imgsz': config.get('imgsz'),
            'device': config.get('device'),
            'project': config.get('project'),
            'name': phase2_name,
            'epochs': phase2_epochs,
            'resume': False,
        }
        # 添加其他训练参数
        for key in ['optimizer', 'lr0', 'lrf', 'weight_decay', 'momentum', 'cos_lr',
                    'warmup_epochs', 'warmup_momentum', 'warmup_bias_lr', 'nbs',
                    'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale',
                    'shear', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup',
                    'copy_paste', 'overlap_mask', 'mask_ratio', 'save', 'save_period',
                    'cache', 'workers', 'verbose', 'seed', 'deterministic', 'amp',
                    'plots', 'val', 'iou']:
            if key in config:
                phase2_config[key] = config[key]
        
        results2 = self.model.train(**phase2_config)
        
        return results1, results2


def main():
    parser = argparse.ArgumentParser(
        description="YOLO11-seg + 多主干训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用主干:
  convnext_small   - ConvNeXt-Small (~50M 参数)
  convnext_base    - ConvNeXt-Base (87.6M 参数, 默认)
  convnext_large   - ConvNeXt-Large (~198M 参数)
  efficientnet_b3  - EfficientNet-B3 (~12M 参数)
  efficientnet_b4  - EfficientNet-B4 (~19M 参数)
  swin_tiny        - Swin-Tiny (~28M 参数)
  swin_small       - Swin-Small (~50M 参数)

示例:
  # 使用 ConvNeXt-Base (默认)
  python train_large_backbone.py --backbone convnext_base --pretrained convnext_base_in22k_ft_in1k.pth
  
  # 使用 ConvNeXt-Large，冻结更长
  python train_large_backbone.py --backbone convnext_large --freeze_epochs 65 --epochs 200
  
  # 使用 EfficientNet-B3
  python train_large_backbone.py --backbone efficientnet_b3 --pretrained efficientnet_b3.pth --freeze_epochs 35
"""
    )
    
    parser.add_argument("--backbone", type=str, default="convnext_base",
                        choices=list(BACKBONE_CONFIG.keys()),
                        help="主干网络类型 (默认: convnext_base)")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="预训练权重路径 (.pth 文件)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="总训练轮数 (默认: 200)")
    parser.add_argument("--freeze_epochs", type=int, default=None,
                        help="冻结主干 epoch 数 (默认: 根据主干自动设置)")
    parser.add_argument("--batch", type=int, default=2,
                        help="Batch size (默认: 2)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="图像尺寸 (默认: 640)")
    parser.add_argument("--device", type=str, default="0,2",
                        help="GPU 设备 (默认: 0,2)")
    parser.add_argument("--download", action="store_true",
                        help="自动下载预训练权重")
    parser.add_argument("--skip_phase1", action="store_true",
                        help="跳过阶段1，直接从已有权重开始阶段2")
    parser.add_argument("--phase1_weights", type=str, default=None,
                        help="阶段1权重路径 (用于跳过阶段1时指定)")
    
    args = parser.parse_args()
    
    # 打印主干信息
    print_backbone_info(args.backbone)
    
    # 设置默认冻结 epoch
    if args.freeze_epochs is None:
        args.freeze_epochs = BACKBONE_CONFIG[args.backbone]['default_freeze_epochs']
    
    # 处理预训练权重
    if args.download and args.pretrained is None:
        args.pretrained = download_backbone_weights(args.backbone)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n训练配置:")
    print(f"  主干: {args.backbone}")
    print(f"  预训练权重: {args.pretrained or '无'}")
    print(f"  总 epochs: {args.epochs}")
    print(f"  冻结 epochs: {args.freeze_epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  图像尺寸: {args.imgsz}")
    print(f"  设备: {args.device}")
    
    # 构建模型
    model_wrapper = MultiBackboneYOLO(
        backbone=args.backbone,
        pretrained_path=args.pretrained,
    )
    
    # 训练配置
    config = {
        "data": "dataset.yaml",
        "task": "segment",
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device.split(','),
        "project": "runs",
        "name": f"irstd_{args.backbone}_{timestamp}",
        "exist_ok": False,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 5e-4,
        "momentum": 0.937,
        "cos_lr": True,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.01,
        "nbs": 16,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.05,
        "overlap_mask": True,
        "mask_ratio": 4,
        "save": True,
        "save_period": 20,
        "cache": False,
        "workers": 8,
        "verbose": True,
        "seed": 0,
        "deterministic": True,
        "amp": True,
        "plots": True,
        "val": True,
        "iou": 0.5,
    }
    
    # 开始分阶段训练
    model_wrapper.train(
        freeze_epochs=args.freeze_epochs,
        total_epochs=args.epochs,
        skip_phase1=args.skip_phase1,
        phase1_weights=args.phase1_weights,
        **config
    )
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
