#!/usr/bin/env python3
"""
ConvNeXt 预训练权重格式转换脚本 v2
=================================
支持三种格式的转换：
1. 旧版 timm: downsample_layers.N.M → stages.N.M.xxx
2. 点号格式: stages.N.blocks.M.xxx → stages_N.blocks.M.xxx
3. 标准格式: 直接兼容
"""

import torch
import re
from pathlib import Path


def convert_state_dict(state_dict: dict) -> dict:
    """
    将任意 ConvNeXt 格式转换为 timm 标准格式
    
    转换规则:
    1. stages.N.xxx → stages_N.xxx (点号转下划线)
    2. stem.N.xxx → stem_N.xxx
    3. head.xxx → head_fc.xxx (仅限没有fc后缀的)
    """
    new_state_dict = {}
    
    for old_key, value in state_dict.items():
        new_key = old_key
        
        # 1. stages.N.xxx → stages_N.xxx
        # 例如: stages.0.blocks.0.conv_dw.weight → stages_0.blocks.0.conv_dw.weight
        #      stages.1.downsample.0.weight → stages_1.downsample.0.weight
        #      stages.3.blocks.2.mlp.fc1.weight → stages_3.blocks.2.mlp.fc1.weight
        new_key = re.sub(r'^stages\.(\d+)', r'stages_\1', new_key)
        
        # 2. stem.N.xxx → stem_N.xxx
        # 例如: stem.0.weight → stem_0.weight
        new_key = re.sub(r'^stem\.(\d+)', r'stem_\1', new_key)
        
        # 3. head.xxx → head_fc.xxx (当没有fc后缀时)
        # 例如: head.weight → head_fc.weight
        if new_key.startswith('head.') and not new_key.startswith('head_fc.'):
            new_key = new_key.replace('head.', 'head_fc.')
        
        new_state_dict[new_key] = value
    
    return new_state_dict


def load_and_convert(weight_path: str, output_path: str = None):
    """
    加载并转换权重文件
    """
    weight_path = Path(weight_path)
    if output_path is None:
        output_path = weight_path.parent / f"{weight_path.stem}_timm{weight_path.suffix}"

    print(f"加载权重: {weight_path}")
    ckpt = torch.load(weight_path, map_location='cpu')

    # 处理嵌套结构
    if 'model' in ckpt:
        state_dict = ckpt['model']
        print("提取 'model' 层")
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        print("提取 'state_dict' 层")
    else:
        state_dict = ckpt

    # 显示原始格式
    sample_keys = list(state_dict.keys())[:3]
    print(f"原始格式示例: {sample_keys}")
    print(f"原始权重层数: {len(state_dict)}")

    # 转换
    new_state_dict = convert_state_dict(state_dict)

    # 显示转换后格式
    new_sample_keys = list(new_state_dict.keys())[:3]
    print(f"转换后格式示例: {new_sample_keys}")
    print(f"转换后权重层数: {len(new_state_dict)}")

    # 保存
    print(f"\n保存到: {output_path}")
    torch.save(new_state_dict, output_path)

    import os
    size = os.path.getsize(output_path) / 1024 / 1024
    print(f"文件大小: {size:.1f} MB")

    return output_path


def verify_conversion(converted_path: str, model_name: str = 'convnext_base.fb_in22k_ft_in1k'):
    """
    验证转换后的权重是否可以被 timm 加载
    """
    print("\n" + "=" * 60)
    print("验证转换结果")
    print("=" * 60)

    try:
        import timm

        # 创建模型（无预训练）
        print(f"创建 timm 模型: {model_name}")
        model = timm.create_model(model_name, pretrained=False, features_only=True)
        model_keys = set(model.state_dict().keys())

        # 加载转换后的权重
        print(f"加载转换后的权重: {converted_path}")
        converted_weights = torch.load(converted_path, map_location='cpu')
        converted_keys = set(converted_weights.keys())

        # 对比
        print(f"\ntimm 模型层数: {len(model_keys)}")
        print(f"转换权重层数: {len(converted_keys)}")

        # 找出差异
        in_timm_not_conv = model_keys - converted_keys
        in_conv_not_timm = converted_keys - model_keys

        if in_timm_not_conv:
            print(f"\n⚠️ timm 需要但转换权重没有的层 ({len(in_timm_not_conv)} 个):")
            for k in sorted(list(in_timm_not_conv))[:5]:
                print(f"  {k}")
            if len(in_timm_not_conv) > 5:
                print(f"  ... 还有 {len(in_timm_not_conv) - 5} 个")

        if in_conv_not_timm:
            print(f"\n⚠️ 转换权重有多余的层 ({len(in_conv_not_timm)} 个):")
            for k in sorted(list(in_conv_not_timm))[:5]:
                print(f"  {k}")
            if len(in_conv_not_timm) > 5:
                print(f"  ... 还有 {len(in_conv_not_timm) - 5} 个")

        # 检查形状匹配
        mismatched_shapes = []
        for key in model_keys & converted_keys:
            model_shape = model.state_dict()[key].shape
            conv_shape = converted_weights[key].shape
            if model_shape != conv_shape:
                mismatched_shapes.append((key, model_shape, conv_shape))

        if mismatched_shapes:
            print(f"\n⚠️ 形状不匹配的层 ({len(mismatched_shapes)} 个):")
            for k, ms, cs in mismatched_shapes[:5]:
                print(f"  {k}: 模型 {ms} vs 权重 {cs}")

        if not in_timm_not_conv and not in_conv_not_timm and not mismatched_shapes:
            print("\n✅ 转换成功! 权重与 timm 模型完全兼容")
            return True
        else:
            print("\n⚠️ 转换后存在差异，可能需要手动调整")
            return False

    except Exception as e:
        print(f"\n验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='ConvNeXt 权重格式转换')
    parser.add_argument('input', help='输入权重文件路径')
    parser.add_argument('-o', '--output', help='输出权重文件路径')
    parser.add_argument('--verify', action='store_true', help='验证转换结果')
    parser.add_argument('--model', default='convnext_base.fb_in22k_ft_in1k', help='timm 模型名称')
    args = parser.parse_args()

    # 转换
    output_path = load_and_convert(args.input, args.output)

    # 验证
    if args.verify:
        verify_conversion(output_path, args.model)


if __name__ == '__main__':
    main()
