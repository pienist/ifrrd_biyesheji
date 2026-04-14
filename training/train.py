import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import yaml
import os
import argparse

from models import YOLOv11WithConvNeXt


class YOLOv11ConvNeXtTrainingPipeline:
    """
    完整的 YOLOv11 with ConvNeXt 训练管道
    """
    
    def __init__(self, config_path: str):
        """
        初始化训练管道
        
        参数：
            config_path: 配置文件路径
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.best_mAP = 0.0
        
    def setup_model(self):
        """
        设置模型
        
        返回：
            model: 完整的 YOLOv11-ConvNeXt 模型
        """
        print("\n" + "="*70)
        print("初始化 YOLOv11-ConvNeXt 模型".center(70))
        print("="*70)
        
        # 创建模型
        model = YOLOv11WithConvNeXt(
            convnext_model_type=self.config['backbone']['type'],
            num_classes=self.config['num_classes'],
            drop_path_rate=self.config['backbone']['drop_path_rate']
        )
        
        # 模型统计
        total_params, trainable_params = count_parameters(model)
        print(f"\n模型参数统计:")
        print(f"  总参数数: {total_params/1e6:.2f}M")
        print(f"  可训练参数: {trainable_params/1e6:.2f}M")
        print(f"  冻结参数: {(total_params-trainable_params)/1e6:.2f}M\n")
        
        return model.to(self.device)
    
    def setup_data(self):
        """
        设置数据加载器
        
        返回：
            (train_loader, val_loader): 训练和验证数据加载器
        """
        print("="*70)
        print("初始化数据加载器".center(70))
        print("="*70)
        
        # 数据增强
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                self.config['input_size'],
                scale=(0.08, 1.0),
                ratio=(3./4., 4./3.)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(int(self.config['input_size'] / 0.875)),
            transforms.CenterCrop(self.config['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 创建虚拟数据集（实际使用时替换为真实数据）
        # 这里仅作演示，实际需要加载 COCO 或自定义数据集
        print(f"\n数据配置:")
        print(f"  输入尺寸: {self.config['input_size']}×{self.config['input_size']}")
        print(f"  批大小: {self.config['batch_size']}")
        print(f"  工作进程: {self.config['num_workers']}\n")
        
        # 返回占位符（实际使用时需要真实的 DataLoader）
        return None, None
    
    def setup_optimizer(self, model: nn.Module):
        """
        设置优化器
        
        参数：
            model: 神经网络模型
            
        返回：
            (optimizer, scheduler): 优化器和学习率调度器
        """
        # 分离不同参数组
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # 使用不同的学习率
        param_groups = [
            {
                'params': backbone_params,
                'lr': self.config['optimizer']['lr'] * 0.1,
                'weight_decay': self.config['optimizer']['weight_decay']
            },
            {
                'params': head_params,
                'lr': self.config['optimizer']['lr'],
                'weight_decay': self.config['optimizer']['weight_decay']
            }
        ]
        
        # 创建优化器
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        print("="*70)
        print("优化器配置".center(70))
        print("="*70)
        print(f"\nOptimizer: AdamW")
        print(f"  Backbone LR: {self.config['optimizer']['lr'] * 0.1:.6f}")
        print(f"  Head LR: {self.config['optimizer']['lr']:.6f}")
        print(f"  Weight Decay: {self.config['optimizer']['weight_decay']}")
        print(f"\nScheduler: Cosine Annealing Warm Restarts")
        print(f"  T_0: 10")
        print(f"  T_mult: 2\n")
        
        return optimizer, scheduler
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch, total_epochs):
        """
        训练单个 epoch
        
        参数：
            model: 模型
            train_loader: 训练数据加载器
            optimizer: 优化器
            criterion: 损失函数
            epoch: 当前 epoch 编号
            total_epochs: 总 epoch 数
            
        返回：
            avg_loss: 平均损失
        """
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch+1}/{total_epochs}] Training",
            leave=True
        )
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # 移动数据到设备
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )
            
            # 更新参数
            optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            })
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def validate(self, model, val_loader, criterion):
        """
        验证模型
        
        参数：
            model: 模型
            val_loader: 验证数据加载器
            criterion: 损失函数
            
        返回：
            (avg_loss, mAP): 平均损失和 mAP
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validating", leave=False):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # 这里应该计算实际的 mAP
        # 由于篇幅限制，这里仅作演示
        mAP = 0.0
        
        return avg_loss, mAP
    
    def train(self):
        """
        完整训练流程
        """
        print("\n" + "█"*70)
        print("█" + " "*68 + "█")
        print("█" + "YOLOv11 with ConvNeXt 训练开始".center(68) + "█")
        print("█" + " "*68 + "█")
        print("█"*70 + "\n")
        
        # 设置模型和数据
        model = self.setup_model()
        train_loader, val_loader = self.setup_data()
        
        # 设置优化器
        optimizer, scheduler = self.setup_optimizer(model)
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        epochs = self.config['epochs']
        
        for epoch in range(epochs):
            print(f"\n{'='*70}")
            print(f"Epoch [{epoch+1}/{epochs}]".center(70))
            print(f"{'='*70}\n")
            
            # 训练
            if train_loader is not None:
                train_loss = self.train_epoch(
                    model, train_loader, optimizer, criterion, epoch, epochs
                )
                print(f"Train Loss: {train_loss:.4f}\n")
            
            # 验证
            if val_loader is not None:
                val_loss, mAP = self.validate(model, val_loader, criterion)
                print(f"Val Loss: {val_loss:.4f}")
                print(f"mAP: {mAP:.4f}\n")
                
                # 保存最佳模型
                if mAP > self.best_mAP:
                    self.best_mAP = mAP
                    torch.save(model.state_dict(), 'best_model.pth')
                    print("✓ 最佳模型已保存\n")
            
            # 更新学习率
            scheduler.step()
        
        print("\n" + "█"*70)
        print("█" + " "*68 + "█")
        print("█" + "训练完成！".center(68) + "█")
        print("█" + f"最佳 mAP: {self.best_mAP:.4f}".center(68) + "█")
        print("█" + " "*68 + "█")
        print("█"*70)


# 配置文件示例 (config.yaml)
"""
# YOLOv11-ConvNeXt 训练配置

# 骨干网络配置
backbone:
  type: 'small'  # 'tiny', 'small', 'base'
  drop_path_rate: 0.1

# 输入配置
input_size: 640
num_classes: 80

# 数据配置
batch_size: 16
num_workers: 4

# 训练配置
epochs: 100
optimizer:
  lr: 1e-3
  weight_decay: 5e-4

# 数据集路径
dataset:
  train_path: '/path/to/train'
  val_path: '/path/to/val'
  test_path: '/path/to/test'

# 输出配置
output:
  checkpoint_dir: './checkpoints'
  log_dir: './logs'
  model_name: 'yolov11_convnext'
"""

# ============================================================================
# 命令行入口
# ============================================================================

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='YOLOv11-ConvNeXt 训练')
    parser.add_argument('--model', type=str, default='convnext-small',
                       choices=['convnext-tiny', 'convnext-small', 'convnext-base'],
                       help='骨干网络类型')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--input-size', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--num-classes', type=int, default=80,
                       help='类别数')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='训练设备')
    parser.add_argument('--drop-path-rate', type=float, default=0.1,
                       help='DropPath 比率')
    
    args = parser.parse_args()
    
    # 从命令行参数构建配置
    config = {
        'backbone': {
            'type': args.model.replace('convnext-', ''),
            'drop_path_rate': args.drop_path_rate
        },
        'input_size': args.input_size,
        'num_classes': args.num_classes,
        'batch_size': args.batch_size,
        'num_workers': 4,
        'epochs': args.epochs,
        'optimizer': {
            'lr': args.lr,
            'weight_decay': 5e-4
        },
        'dataset': {
            'train_path': None,
            'val_path': None
        }
    }
    
    # 打印配置
    print("\n" + "="*60)
    print("训练配置".center(60))
    print("="*60)
    print(f"  骨干网络: {args.model}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  输入尺寸: {args.input_size}")
    print(f"  类别数: {args.num_classes}")
    print(f"  设备: {args.device}")
    print("="*60 + "\n")
    
    # 创建训练管道（使用配置字典代替配置文件）
    pipeline = DirectTrainingPipeline(config)
    pipeline.train()


class DirectTrainingPipeline:
    """直接使用配置字典的训练管道"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu'
        )
        self.best_mAP = 0.0
    
    def setup_model(self):
        """设置模型"""
        print("\n" + "="*70)
        print("初始化 YOLOv11-ConvNeXt 模型".center(70))
        print("="*70)
        
        model = YOLOv11WithConvNeXt(
            convnext_model_type=self.config['backbone']['type'],
            num_classes=self.config['num_classes'],
            drop_path_rate=self.config['backbone']['drop_path_rate']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n模型参数统计:")
        print(f"  总参数数: {total_params/1e6:.2f}M")
        print(f"  可训练参数: {trainable_params/1e6:.2f}M\n")
        
        return model.to(self.device)
    
    def setup_optimizer(self, model):
        """设置优化器"""
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        param_groups = [
            {
                'params': backbone_params,
                'lr': self.config['optimizer']['lr'] * 0.1,
                'weight_decay': self.config['optimizer']['weight_decay']
            },
            {
                'params': head_params,
                'lr': self.config['optimizer']['lr'],
                'weight_decay': self.config['optimizer']['weight_decay']
            }
        ]
        
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        print(f"Optimizer: AdamW")
        print(f"  Backbone LR: {self.config['optimizer']['lr'] * 0.1:.6f}")
        print(f"  Head LR: {self.config['optimizer']['lr']:.6f}\n")
        
        return optimizer, scheduler
    
    def train(self):
        """完整训练流程"""
        print("\n" + "█"*70)
        print("█" + " "*68 + "█")
        print("█" + "YOLOv11 with ConvNeXt 训练开始".center(68) + "█")
        print("█" + " "*68 + "█")
        print("█"*70 + "\n")
        
        model = self.setup_model()
        optimizer, scheduler = self.setup_optimizer(model)
        criterion = nn.CrossEntropyLoss()
        
        epochs = self.config['epochs']
        
        for epoch in range(epochs):
            print(f"\n{'='*70}")
            print(f"Epoch [{epoch+1}/{epochs}]".center(70))
            print(f"{'='*70}\n")
            
            # 模拟训练步骤
            model.train()
            print(f"  模拟训练中... (lr={optimizer.param_groups[0]['lr']:.6f})")
            
            # 保存示例权重
            if (epoch + 1) % 10 == 0:
                save_path = f'checkpoint_epoch_{epoch+1}.pth'
                torch.save(model.state_dict(), save_path)
                print(f"✓ Checkpoint 已保存: {save_path}\n")
            
            scheduler.step()
        
        # 保存最终模型
        torch.save(model.state_dict(), 'best_model.pth')
        
        print("\n" + "█"*70)
        print("█" + "训练完成！".center(68) + "█")
        print("█" + f"模型已保存: best_model.pth".center(68) + "█")
        print("█"*70)


if __name__ == '__main__':
    main()


# ============================================================================
# 代码解析
# ============================================================================
"""
YOLOv11ConvNeXtTrainingPipeline 的核心设计：

1. 分组学习率策略：
   
   为什么不同层使用不同学习率？
   ① 骨干网络已通过 ImageNet 预训练
   ② 使用较小学习率避免破坏已学知识
   ③ 检测头需要较大学习率快速学习任务特定知识
   
   推荐比例：
   Backbone LR: 0.1 × base_lr
   Head LR: 1.0 × base_lr

2. 学习率调度策略（Cosine Annealing with Warm Restarts）：
   
   为什么选择这个策略？
   ① 余弦退火提供平滑的衰减
   ② Warm Restarts 允许模型逃出局部最优
   ③ 相比固定学习率，收敛更稳定
   
   数学表达：
   lr_t = η_min + 0.5(η_0 - η_min) * (1 + cos(πt/T))
   其中 T 是周期，定期重启

3. 梯度裁剪（Gradient Clipping）：
   
   作用：
   ① 防止梯度爆炸
   ② 稳定训练过程
   ③ 特别重要的深度网络中
   
   建议范数：
   - 标准值：1.0
   - 保守值：0.5
   - 激进值：2.0

4. 分层冻结策略：
   
   第一阶段（冻结 Backbone）：
   - 前 10-20 个 epoch
   - 只训练检测头
   - 快速适应任务

"""
