#!/usr/bin/env python3
"""
ConvNeXt-Seg 红外小目标分割训练脚本 - 实验B
=============================================
在 Baseline 基础上增加通道适配器 (Channel Adapter)

核心改进:
- 添加 1→3 通道适配器，将红外单通道转换为3通道
- 使用 ImageNet 权重均值初始化适配器
- 适配器可学习，允许微调适应红外特征

使用方法:
    conda activate yolov11_seg
    python train_convnext_seg_B.py

两阶段训练（冻结 + 解冻）:
    python train_convnext_seg_B.py --epochs 120 --batch 24 --device 0 --freeze 5 --freeze_epochs 30
"""

import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import ChannelAdapter


# ==========================================
# 修改后的权重加载函数（支持通道适配器）
# ==========================================

def load_convnext_pretrained(model, pretrained_path):
    """加载 ConvNeXt ImageNet-22k 预训练权重到 YOLO 模型。

    Args:
        model: YOLO 模型
        pretrained_path: 预训练权重路径

    Returns:
        model: 加载权重后的模型
    """
    print(f"\n{'='*60}")
    print(f"正在加载预训练权重: {pretrained_path}")
    print(f"{'='*60}")

    # 加载预训练权重
    state_dict = torch.load(pretrained_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']

    # 打印原始权重信息
    print(f"预训练权重总层数: {len(state_dict)}")

    # 获取第一个卷积层的权重（用于初始化通道适配器）
    first_conv_weight = None
    for key, value in state_dict.items():
        if 'stem.0.weight' in key:
            first_conv_weight = value
            print(f"  找到第一层卷积权重: {key}, shape={value.shape}")
            break

    # 初始化通道适配器（model.model[0] 是 ChannelAdapter）
    channel_adapter = None
    if hasattr(model.model, '0') and isinstance(model.model[0], ChannelAdapter):
        channel_adapter = model.model[0]
        if first_conv_weight is not None:
            channel_adapter.init_from_imagenet(first_conv_weight)
            print("  通道适配器初始化完成!")

    new_state_dict = {}
    loaded_count = 0
    skipped_count = 0

    # 预训练 stage 索引到 YOLO model 索引的映射
    # 注意: 实验B的模型在backbone之前有ChannelAdapter，所以索引需要+1
    stage_mapping = {
        0: 2,  # stages.0 -> model.2 (model.0是ChannelAdapter, model.1是stem)
        1: 3,  # stages.1 -> model.3
        2: 4,  # stages.2 -> model.4
        3: 5,  # stages.3 -> model.5
    }

    # 获取模型所有键（用于验证）
    model_keys = set(dict(model.model.named_parameters()).keys())

    for key, value in state_dict.items():
        # 跳过 head 相关的层（分类头）
        if 'head' in key or 'norm_pre' in key or 'cls' in key or 'fc_norm' in key:
            skipped_count += 1
            continue

        # 跳过原始 stem（YAML 使用 Conv+BN，不是 LayerNorm）
        if key.startswith('stem.'):
            skipped_count += 1
            continue

        # 跳过预训练权重中的 LayerNorm 层（downsample.0 是 LayerNorm）
        if '.downsample.0.' in key:
            skipped_count += 1
            continue

        new_key = key

        # 处理 stages: stages.{stage_idx}.blocks.{block_idx}.* -> model.{model_idx}.blocks.{block_idx}.*
        if key.startswith('stages.'):
            parts = key.split('.')
            if len(parts) >= 3:  # stages.{idx}.blocks.{i}...
                stage_idx = int(parts[1])
                block_type = parts[2]  # 'blocks' 或 'downsample'

                if stage_idx in stage_mapping and block_type == 'blocks':
                    model_idx = stage_mapping[stage_idx]
                    # 重新构建: stages.0.blocks.0.dwconv.weight -> model.2.blocks.0.dwconv.weight
                    new_key = 'model.' + str(model_idx) + '.' + '.'.join(parts[2:])

                    # 替换键名格式 (timm -> YOLO)
                    new_key = new_key.replace('mlp.fc1', 'pwconv1')
                    new_key = new_key.replace('mlp.fc2', 'pwconv2')
                    new_key = new_key.replace('conv_dw', 'dwconv')

                    # 验证键是否存在于目标模型
                    if new_key in model_keys:
                        new_state_dict[new_key] = value
                        loaded_count += 1
                    else:
                        skipped_count += 1
                        if skipped_count <= 5:
                            print(f"  跳过 (模型中不存在): {new_key}")
                else:
                    skipped_count += 1
            else:
                skipped_count += 1
        else:
            skipped_count += 1

    print(f"尝试加载权重层数: {loaded_count}")
    print(f"跳过层数: {skipped_count}")

    # 加载权重
    missing_keys, unexpected_keys = model.model.load_state_dict(new_state_dict, strict=False)

    if missing_keys:
        print(f"\n缺失的键 ({len(missing_keys)}): 这些键在预训练权重中不存在")
        for k in missing_keys[:5]:
            print(f"  {k}")
        if len(missing_keys) > 5:
            print(f"  ... 还有 {len(missing_keys) - 5} 个")

    if unexpected_keys:
        print(f"\n意外的键 ({len(unexpected_keys)}): 这些键在模型中不存在")
        for k in unexpected_keys[:5]:
            print(f"  {k}")
        if len(unexpected_keys) > 5:
            print(f"  ... 还有 {len(unexpected_keys) - 5} 个")

    print(f"\n✅ 预训练权重加载完成!")

    return model


def freeze_layers(model, n_freeze):
    """冻结模型的前 n 层。

    Args:
        model: YOLO 模型
        n_freeze: 要冻结的层数，0 表示全部解冻
    """
    # ChannelAdapter (model.0) 始终保持可训练
    if hasattr(model.model, '0') and isinstance(model.model[0], ChannelAdapter):
        for param in model.model[0].parameters():
            param.requires_grad = True
        print("  通道适配器始终保持可训练状态")

    if n_freeze == 0:
        print("解冻所有层（包括backbone）")
        for param in model.model.parameters():
            param.requires_grad = True
        return

    print(f"冻结前 {n_freeze} 层...")
    freeze_count = 0
    for i, (name, param) in enumerate(model.model.named_parameters()):
        if i < n_freeze:
            param.requires_grad = False
            freeze_count += 1
        else:
            param.requires_grad = True

    print(f"已冻结 {freeze_count} 层, 可训练参数: {sum(1 for p in model.model.parameters() if p.requires_grad)} 层")


def train_with_config(model, config, freeze_n=5):
    """使用给定配置训练模型。

    Args:
        model: YOLO 模型实例
        config: 训练配置字典
        freeze_n: 要冻结的层数（0=全部解冻）

    Returns:
        results: 训练结果
    """
    # 冻结/解冻层
    freeze_layers(model, freeze_n)

    # 打印训练信息
    freeze_status = "冻结" if freeze_n > 0 else "解冻全部"
    print(f"\n开始训练: epochs={config['epochs']}, batch={config['batch']}, {freeze_status} backbone")
    print(f"学习率: lr0={config['lr0']}, lrf={config.get('lrf', 0.1)}")
    print("=" * 60)

    results = model.train(**config)

    return results


def main():
    # ==========================================
    # 命令行参数解析
    # ==========================================
    parser = argparse.ArgumentParser(description="ConvNeXt-Seg 实验B: 通道适配器训练脚本")
    parser.add_argument("--epochs", type=int, default=120, help="总训练轮数")
    parser.add_argument("--batch", type=int, default=24, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="GPU 设备")
    parser.add_argument("--freeze", type=int, default=5, help="冻结前 N 层 (0=全部解冻)")
    parser.add_argument("--freeze_epochs", type=int, default=30, help="冻结阶段的 epoch 数")
    parser.add_argument("--name", type=str, default=None, help="实验名称")
    parser.add_argument("--weights", type=str, default=None, help="预训练权重路径")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复")

    cmd_args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = cmd_args.name or f"B_channel_adapter_{timestamp}"

    # ==========================================
    # 实验B配置说明
    # ==========================================
    print("=" * 60)
    print("实验B: 通道适配器 (Channel Adapter)")
    print("=" * 60)
    print("""
核心改进:
- 添加 1→3 通道适配器，将红外单通道转换为3通道
- 使用 ImageNet 权重均值初始化适配器
- 适配器可学习，允许微调适应红外特征

原理:
- ImageNet预训练权重基于3通道RGB图像
- 红外图像是单通道灰度图
- 使用1×1卷积进行通道投影
- 初始化: weight_1ch = mean(weight_3ch, dim=1)

配置:
- 阶段1 (冻结): 30 epochs, lr0=0.001 (Batch 24)
- 阶段2 (解冻): 90 epochs, lr0=0.0005 (Batch 24)
    """)

    # ==========================================
    # 训练配置参数
    # ==========================================
    config = {
        # 模型配置 - ConvNeXt-Seg with Channel Adapter
        "model": "ultralytics/cfg/models/convnext/convnext-seg-B.yaml",
        # 预训练权重路径
        "pretrained_path": "convnext_base_in22k_ft_in1k.pth",
        # 数据配置
        "data": "dataset_B.yaml",
        "task": "segment",
        # 训练轮数
        "epochs": cmd_args.epochs,
        # Batch size 设置
        "batch": cmd_args.batch,
        # 图像尺寸
        "imgsz": 640,
        # 设备配置
        "device": cmd_args.device,
        # 输出配置
        "project": "runs",
        "name": base_name,
        "exist_ok": False,
        # 优化器配置
        "optimizer": "AdamW",
        "lr0": 0.001,  # Batch 24 调整
        "lrf": 0.1,
        "weight_decay": 0.05,
        # 学习率调度
        "cos_lr": True,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        # 数据增强
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
        "mixup": 0.0,
        "copy_paste": 0.0,
        # 分割任务特定参数
        "overlap_mask": True,
        "mask_ratio": 4,
        # 其他设置
        "save": True,
        "save_period": 20,
        "cache": False,
        "workers": 8,
        "verbose": True,
        "seed": 0,
        "deterministic": True,
        "amp": True,
        # 验证设置
        "val": True,
        "plots": True,
        "iou": 0.5,
        # 冻结层数
        "freeze": cmd_args.freeze,
    }

    # ==========================================
    # 两阶段训练逻辑
    # ==========================================
    freeze_epochs = cmd_args.freeze_epochs
    total_epochs = cmd_args.epochs
    resume_path = cmd_args.resume

    # 确定是否需要两阶段训练
    use_two_stage = (freeze_epochs > 0) and (total_epochs > freeze_epochs) and (cmd_args.freeze > 0) and (not resume_path)

    if use_two_stage:
        print("\n" + "=" * 60)
        print("检测到两阶段训练模式")
        print(f"阶段 1: 冻结 {cmd_args.freeze} 层, 训练 {freeze_epochs} epochs")
        print(f"阶段 2: 解冻全部, 训练 {total_epochs - freeze_epochs} epochs")
        print("=" * 60)
        print("\n📊 学习率策略 (Batch 24):")
        print("  阶段1 (冻结): lr0=0.001, lrf=0.1  →  Neck+Head 快速收敛")
        print("  阶段2 (微调): lr0=0.0005, lrf=0.1 →  Backbone 温和调整")
        print("  (阶段2学习率为阶段1的 1/2，保护预训练权重)")

        # ========== 阶段 1: 冻结训练 ==========
        print("\n" + "=" * 60)
        print("阶段 1: 冻结 backbone 训练 (Neck+Head)")
        print("=" * 60)
        print(f"学习率: lr0={config['lr0']}, lrf={config.get('lrf', 0.1)}")

        # 阶段1配置
        stage1_config = config.copy()
        stage1_config["epochs"] = freeze_epochs
        stage1_config["name"] = base_name + "_freeze"
        stage1_config["exist_ok"] = False
        stage1_config["lr0"] = 0.001  # Batch 24 调整
        stage1_config["lrf"] = 0.1

        # 加载模型
        model = YOLO(config["model"])

        # 加载预训练权重
        pretrained_path = stage1_config.pop("pretrained_path", None)
        if pretrained_path and os.path.exists(pretrained_path):
            model = load_convnext_pretrained(model, pretrained_path)

        # 阶段1训练
        train_with_config(model, stage1_config, freeze_n=cmd_args.freeze)

        # 获取阶段1最佳权重
        # ultralytics 路径规则: runs/segment/runs/{name}/weights/best.pt
        stage1_weights = f"runs/segment/runs/{stage1_config['name']}/weights/best.pt"
        print(f"\n阶段 1 完成! 权重保存于: {stage1_weights}")

        # ========== 阶段 2: 解冻训练 ==========
        print("\n" + "=" * 60)
        print("阶段 2: 解冻全部参数，微调训练")
        print("=" * 60)
        print(f"学习率: lr0=0.0005, lrf={config.get('lrf', 0.1)} (backbone 使用 5e-4 学习率)")

        # 阶段2配置
        stage2_config = config.copy()
        stage2_config["epochs"] = total_epochs - freeze_epochs
        stage2_config["name"] = base_name + "_finetune"
        stage2_config["exist_ok"] = False
        stage2_config.pop("pretrained_path", None)
        stage2_config["lrf"] = 0.1
        stage2_config["lr0"] = 0.0005  # Batch 24 调整

        # 从阶段1加载权重
        if os.path.exists(stage1_weights):
            print(f"从阶段1加载权重: {stage1_weights}")
            stage2_config["model"] = stage1_weights
        else:
            print(f"警告: 阶段1权重不存在 {stage1_weights}，将从头开始训练阶段2")

        # 加载模型并解冻
        model2 = YOLO(stage2_config["model"])

        # 阶段2训练 (freeze=0 表示全部解冻)
        train_with_config(model2, stage2_config, freeze_n=0)

        print(f"\n阶段 2 完成! 权重保存于: runs/segment/{stage2_config['name']}/weights/best.pt")

        # 打印最终总结
        print("\n" + "=" * 60)
        print("🎉 两阶段训练全部完成!")
        print("=" * 60)
        print(f"阶段 1 (冻结): runs/segment/{stage1_config['name']}/weights/best.pt")
        print(f"阶段 2 (微调): runs/segment/{stage2_config['name']}/weights/best.pt")
        print(f"最终模型: runs/segment/{stage2_config['name']}/weights/best.pt")

    else:
        # ==========================================
        # 单阶段训练（普通模式）
        # ==========================================
        print("=" * 60)
        print("ConvNeXt-Seg 实验B: 通道适配器训练")
        print("=" * 60)
        print(f"模型: {config['model']}")
        print(f"数据集: {config['data']}")
        print(f"Epochs: {config['epochs']}")
        print(f"Batch Size: {config['batch']}")
        print(f"图像尺寸: {config['imgsz']}")
        print(f"设备: {config['device']}")
        print(f"冻结层数: {config['freeze']}")
        print(f"优化器: {config['optimizer']}, lr={config['lr0']}, cos_lr={config['cos_lr']}")

        # 加载预训练权重
        pretrained_path = config.pop("pretrained_path", None)

        # 处理 resume 或使用预训练权重
        if resume_path:
            print(f"从检查点恢复: {resume_path}")
            config["model"] = resume_path
        elif cmd_args.weights:
            config["model"] = cmd_args.weights

        model = YOLO(config["model"])

        # 加载 ConvNeXt ImageNet-22k 预训练权重
        if not resume_path and not cmd_args.weights:
            if pretrained_path and os.path.exists(pretrained_path):
                model = load_convnext_pretrained(model, pretrained_path)
            else:
                print(f"\n预训练权重文件不存在: {pretrained_path}")
                print("将从头开始训练...")

        print(f"{'='*60}\n")

        results = model.train(**config)

        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        print(f"模型保存位置: runs/segment/{config['name']}")
        print(f"最佳权重: runs/segment/{config['name']}/weights/best.pt")

        return results


if __name__ == "__main__":
    main()
