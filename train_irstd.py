#!/usr/bin/env python3
"""
YOLO11-seg 红外小目标分割训练脚本（Day1 基线实验）
=================================================
红外小目标检测与分割多任务训练

Day1 实验目标：
- YOLOv11-seg 从零训练基线（不使用 ImageNet 预训练）
- CNN 基线，用于后续消融实验对比

使用方法:
    conda activate yolov11_seg
    python train_irstd.py
"""

from ultralytics import YOLO
from datetime import datetime

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ==========================================
    # 训练配置参数
    # ==========================================
    config = {
        # 模型配置 - 从头训练（不使用预训练权重）
        'model': 'yolo11s-seg.yaml',           # YOLO11-seg s规模 YAML 配置（从头训练）
        'pretrained': False,                    # 关闭预训练权重

        # 数据配置
        'data': 'dataset.yaml',                 # 数据集配置文件（YOLO格式）
        'task': 'segment',                      # 分割任务

        # 训练轮数
        'epochs': 200,                          # Day1 基线实验：200 epochs

        # Batch size 设置
        # 显存情况：
        #   GPU0 TITAN Xp  : 空闲  9.1 GB
        #   GPU1 TITAN X   : 空闲  7.9 GB
        #   GPU2 TITAN X   : 空闲  8.8 GB
        # batch=16 在 3 GPU 并行下每卡约 5-6 GB，满足显存约束
        'batch': 16,

        # 图像尺寸 - 红外小目标数据集原始图像尺寸为 640x640
        'imgsz': 640,

        # 设备配置
        'device': [0, 1, 2],                   # 使用 3 张 GPU 分布式训练

        # 输出配置
        'project': 'runs',                      # 项目目录
        'name': f'irstd_yolo11s_seg_{timestamp}',  # 实验名称（含时间戳）
        'exist_ok': False,                      # 不覆盖已有实验

        # 优化器配置
        # YOLO 默认使用 SGD，对比从头训练场景更稳定，收敛更好
        # lr0=0.01 是 YOLO 系列的经典设置，配合 cosine LR 和 warmup 效果可靠
        'optimizer': 'SGD',                     # SGD 优化器，从头训练更稳定
        'lr0': 0.005,                            # 初始学习率（从头训练保守设置，0.01 为预训练微调标准值）
        'lrf': 0.01,                             # 最终学习率比例（最低降至 lr0*0.01）
        'momentum': 0.937,                      # SGD 动量（YOLO 标准值）
        'weight_decay': 0.0005,                 # 权重衰减（标准值，防止过拟合）

        # 学习率调度
        # cos_lr=True + lrf=0.01：经典 YOLO 退火策略，200 epoch 下收敛充分
        'cos_lr': True,                         # 余弦退火学习率
        'warmup_epochs': 3.0,                   # 预热 3 epoch（标准值）
        'warmup_momentum': 0.8,                 # 预热动量
        'warmup_bias_lr': 0.1,                  # 预热偏置学习率

        # 数据增强 - 红外小目标适度增强
        'hsv_h': 0.015,                         # 色调增强
        'hsv_s': 0.7,                           # 饱和度增强
        'hsv_v': 0.4,                           # 亮度增强（红外图像主要依赖亮度）
        'degrees': 0.0,                         # 不旋转（红外目标方向敏感）
        'translate': 0.1,                       # 平移
        'scale': 0.5,                           # 缩放
        'shear': 0.0,                           # 不剪切
        'perspective': 0.0,                      # 不透视
        'flipud': 0.0,                          # 不上下翻转（方向敏感）
        'fliplr': 0.5,                          # 左右翻转
        'mosaic': 1.0,                          # Mosaic 增强（适合小目标）
        'mixup': 0.0,                           # 不使用 MixUp
        'copy_paste': 0.0,                      # 不使用 Copy-paste

        # 分割任务特定参数
        'overlap_mask': True,                   # 训练时合并实例 mask
        'mask_ratio': 4,                       # mask 下采样比例

        # 其他设置
        'save': True,                           # 保存检查点和预测结果
        'save_period': 20,                      # 每 20 epoch 保存一次
        'cache': False,                         # 不缓存图像到内存（显存有限）
        'workers': 8,                           # 数据加载线程数
        'verbose': True,                         # 打印详细日志
        'seed': 0,                               # 随机种子
        'deterministic': True,                   # 确定性操作
        'amp': True,                             # AMP 混合精度训练（节省显存）

        # 验证设置
        'val': True,                             # 训练时验证
        'plots': True,                           # 生成训练曲线图
    }

    # ==========================================
    # 加载模型并开始训练
    # ==========================================
    print("=" * 60)
    print("YOLO11-seg 红外小目标分割训练（Day1 基线实验）")
    print("=" * 60)
    print(f"模型: {config['model']} (从头训练，无预训练权重)")
    print(f"数据集: {config['data']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch Size: {config['batch']} (3 GPU, 每卡约 {config['batch']//3} 张)")
    print(f"图像尺寸: {config['imgsz']}")
    print(f"设备: {config['device']}")
    print(f"优化器: {config['optimizer']}, lr={config['lr0']}, cos_lr={config['cos_lr']}")
    print("=" * 60)

    model = YOLO(config['model'])
    results = model.train(**config)

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"模型保存位置: runs/segment/{config['name']}")
    print(f"最佳权重: runs/segment/{config['name']}/weights/best.pt")

    return results


if __name__ == '__main__':
    main()
