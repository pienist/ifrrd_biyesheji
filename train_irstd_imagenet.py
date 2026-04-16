#!/usr/bin/env python3
"""
YOLO11-seg 红外小目标分割训练脚本（Day2 COCO预训练对照实验）.
=================================================
红外小目标检测与分割多任务训练

Day2 实验目标：
- YOLOv11-seg + COCO 预训练权重（官方预训练模型）
- 与 Day1 从零训练基线形成对照，验证预训练对红外小目标任务的帮助

使用方法:
    conda activate yolov11_seg
    python train_irstd_imagenet.py
"""

from datetime import datetime

from ultralytics import YOLO


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ==========================================
    # 训练配置参数
    # ==========================================
    config = {
        # 模型配置 - 
        "model": "yolo11s-seg.pt",  # 加载预训练权重
        "pretrained": True,  # 开启预训练
        # 数据配置
        "data": "dataset.yaml",  # 数据集配置文件（YOLO格式）
        "task": "segment",  # 分割任务
        # 训练轮数
        "epochs": 200,  # 与 Day1 基线一致：200 epochs
        # Batch size 设置
        "batch": 16,
        # 图像尺寸 - 红外小目标数据集原始图像尺寸为 640x640
        "imgsz": 640,
        # 设备配置
        "device": [0],  # 单卡训练
        # 输出配置
        "project": "runs",  # 项目目录
        "name": f"irstd_yolo11s_seg_coco_{timestamp}",  # 实验名称（含时间戳）
        "exist_ok": False,  # 不覆盖已有实验
        # 优化器配置
        # lr0=0.005：与 Day1 基线完全一致，保证预训练实验对照的公平性
        "optimizer": "SGD",  # SGD 优化器
        "lr0": 0.005,  # 初始学习率（与 Day1 基线对齐）
        "lrf": 0.01,  # 最终学习率比例（最低降至 lr0*0.01）
        "momentum": 0.937,  # SGD 动量（YOLO 标准值）
        "weight_decay": 0.0005,  # 权重衰减（标准值，防止过拟合）
        # 学习率调度
        "cos_lr": True,  # 余弦退火学习率
        "warmup_epochs": 3.0,  # 预热 3 epoch（标准值）
        "warmup_momentum": 0.8,  # 预热动量
        "warmup_bias_lr": 0.1,  # 预热偏置学习率
        # 数据增强 - 与 Day1 基线完全一致
        "hsv_h": 0.015,  # 色调增强
        "hsv_s": 0.7,  # 饱和度增强
        "hsv_v": 0.4,  # 亮度增强（红外图像主要依赖亮度）
        "degrees": 0.0,  # 不旋转（红外目标方向敏感）
        "translate": 0.1,  # 平移
        "scale": 0.5,  # 缩放
        "shear": 0.0,  # 不剪切
        "perspective": 0.0,  # 不透视
        "flipud": 0.0,  # 不上下翻转（方向敏感）
        "fliplr": 0.5,  # 左右翻转
        "mosaic": 1.0,  # Mosaic 增强（适合小目标）
        "mixup": 0.0,  # 不使用 MixUp
        "copy_paste": 0.0,  # 不使用 Copy-paste
        # 分割任务特定参数
        "overlap_mask": True,  # 训练时合并实例 mask
        "mask_ratio": 4,  # mask 下采样比例
        # 其他设置
        "save": True,  # 保存检查点和预测结果
        "save_period": 20,  # 每 20 epoch 保存一次
        "cache": False,  # 不缓存图像到内存（显存有限）
        "workers": 8,  # 数据加载线程数
        "verbose": True,  # 打印详细日志
        "seed": 0,  # 随机种子
        "deterministic": True,  # 确定性操作
        "amp": True,  # AMP 混合精度训练（节省显存）
        # 验证设置
        "val": True,  # 训练时验证
        "plots": True,  # 生成训练曲线图
        "iou": 0.5,  # NMS IoU 阈值（平衡 Precision 与 Recall）
    }

    # ==========================================
    # 加载模型并开始训练
    # ==========================================
    print("=" * 60)
    print("YOLO11-seg 红外小目标分割训练脚本（Day2 COCO预训练对照）")
    print("=" * 60)
    print(f"模型: {config['model']} (COCO 预训练权重)")
    print(f"数据集: {config['data']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch Size: {config['batch']} (单 GPU)")
    print(f"图像尺寸: {config['imgsz']}")
    print(f"设备: {config['device']}")
    print(f"优化器: {config['optimizer']}, lr={config['lr0']}, cos_lr={config['cos_lr']}")
    print("=" * 60)

    model = YOLO(config["model"])
    results = model.train(**config)

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"模型保存位置: runs/segment/{config['name']}")
    print(f"最佳权重: runs/segment/{config['name']}/weights/best.pt")


if __name__ == "__main__":
    main()
