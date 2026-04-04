#!/usr/bin/env python3
"""
YOLO11-seg 红外小目标分割训练脚本（从头训练版本）
=================================================
红外小目标检测与分割多任务训练

不使用预训练权重，从头训练：
- 避免自然图像特征对红外小目标的干扰
- 让网络完全适配红外图像特征

使用方法:
    conda activate yolov11_seg
    python train_irstd.py

或使用命令行:
    yolo segment train model=yolo11s-seg.yaml data=dataset.yaml epochs=50 batch=24 imgsz=512 pretrained=False
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
        # 红外小目标与自然图像差异大，从头训练可能效果更好
        'model': 'yolo11s-seg.yaml',           # YOLO11-seg s规模 YAML 配置（从头训练）
        'pretrained': False,                    # 关闭预训练权重

        # 数据配置
        'data': 'dataset.yaml',                 # 数据集配置文件（YOLO格式）
        'task': 'segment',                      # 分割任务

        # 训练轮数
        'epochs': 50,                           # 训练50个epoch

        # Batch size 设置
        # 3 GPU × 8 = 24 (每张TITAN Xp 12GB足够)
        'batch': 24,

        # 图像尺寸 - 与你的数据集512x512一致
        'imgsz': 512,

        # 设备配置
        # 使用所有3张GPU进行分布式训练
        'device': [0, 1, 2],                   # 使用GPU 0, 1, 2

        # 输出配置
        'project': 'runs',                         # 项目目录（Ultralytics 会自动按 task 子目录）
        'name': f'irstd_yolo11s_seg_{timestamp}',  # 实验名称（含时间戳）
        'exist_ok': False,                      # 不覆盖已有实验

        # 优化器配置 - 从头训练推荐 AdamW
        'optimizer': 'AdamW',                   # AdamW 优化器，适合从头训练
        'lr0': 0.001,                           # 初始学习率（降低，0.01 太高）
        'lrf': 0.01,                            # 最终学习率比例
        'momentum': 0.937,                     # SGD动量（AdamW主要不看这个）
        'weight_decay': 0.05,                  # 权重衰减（增加正则化，防止过拟合）

        # 学习率调度
        'cos_lr': True,                         # 使用余弦退火调度
        'warmup_epochs': 5.0,                   # 预热5个epoch（延长预热）
        'warmup_momentum': 0.8,                 # 预热时的动量
        'warmup_bias_lr': 0.1,                 # 预热时的偏置学习率

        # 数据增强
        'hsv_h': 0.015,                       # 色调增强
        'hsv_s': 0.7,                         # 饱和度增强
        'hsv_v': 0.4,                         # 亮度增强
        'degrees': 0.0,                       # 旋转角度
        'translate': 0.1,                     # 平移
        'scale': 0.5,                         # 缩放
        'shear': 0.0,                         # 剪切
        'perspective': 0.0,                   # 透视变换
        'flipud': 0.0,                        # 上下翻转 (红外图像通常不需要)
        'fliplr': 0.5,                        # 左右翻转概率
        'mosaic': 1.0,                        # Mosaic增强
        'mixup': 0.0,                         # MixUp增强
        'copy_paste': 0.0,                    # Copy-paste增强 (分割任务)

        # 分割任务特定参数
        'overlap_mask': True,                 # 训练时合并实例mask
        'mask_ratio': 4,                      # mask下采样比例

        # 其他设置
        'save': True,                         # 保存检查点和预测结果
        'save_period': 10,                     # 每10个epoch保存一次
        'cache': False,                       # 不缓存图像到内存
        'workers': 8,                          # 数据加载线程数
        'verbose': True,                       # 打印详细日志
        'seed': 0,                             # 随机种子
        'deterministic': True,                # 确定性操作
        'amp': True,                          # 启用AMP混合精度训练

        # 验证设置
        'val': True,                           # 训练时验证
        'plots': True,                         # 生成训练曲线图
    }

    # ==========================================
    # 加载模型并开始训练
    # ==========================================
    print("=" * 60)
    print("YOLO11-seg 红外小目标分割训练（从头训练）")
    print("=" * 60)
    print(f"模型: {config['model']} (从头训练)")
    print(f"数据集: {config['data']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch Size: {config['batch']}")
    print(f"图像尺寸: {config['imgsz']}")
    print(f"设备: {config['device']}")
    print(f"优化器: {config['optimizer']}, lr={config['lr0']}")
    print("=" * 60)

    # 加载预训练模型
    model = YOLO(config['model'])

    # 开始训练
    results = model.train(**config)

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"模型保存位置: runs/segment/{config['name']}")
    print(f"最佳权重: runs/segment/{config['name']}/weights/best.pt")

    return results


if __name__ == '__main__':
    main()
