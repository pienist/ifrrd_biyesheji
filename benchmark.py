#!/usr/bin/env python3
"""YOLOv11-ConvNeXt 性能基准测试"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path


def count_parameters(model):
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_latency(model, input_size=640, warmup=10, iterations=100, device='cuda'):
    """
    测量单次推理延迟
    
    返回: 平均延迟(ms), 标准差, 最小值, 最大值
    """
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 测量
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)  # 转换为ms
    
    return np.mean(times), np.std(times), np.min(times), np.max(times)


def measure_throughput(model, input_size=640, warmup=10, iterations=50, batch_size=8, device='cuda'):
    """
    测量吞吐量
    
    返回: 吞吐量(img/s), 平均延迟(ms)
    """
    model.eval()
    dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup // 2):
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 测量
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    total_time = time.perf_counter() - start_time
    total_images = iterations * batch_size
    throughput = total_images / total_time
    avg_latency = (total_time / iterations) * 1000  # ms
    
    return throughput, avg_latency


def get_model_size(model_path):
    """获取模型文件大小"""
    if Path(model_path).exists():
        size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        return size_mb
    return None


def benchmark_model(model, model_name, model_path, input_size, device):
    """运行完整基准测试"""
    print("\n" + "="*60)
    print(f"基准测试: {model_name}".center(60))
    print("="*60)
    
    # 1. 模型参数统计
    total_params, trainable_params = count_parameters(model)
    print(f"\n[模型参数]")
    print(f"  总参数:   {total_params/1e6:.2f}M")
    print(f"  可训练:   {trainable_params/1e6:.2f}M")
    
    # 2. 模型大小
    if model_path:
        size = get_model_size(model_path)
        if size:
            print(f"  模型大小: {size:.2f}MB")
    
    # 3. 延迟测试
    print(f"\n[延迟测试] (输入尺寸: {input_size}x{input_size})")
    mean_lat, std_lat, min_lat, max_lat = measure_latency(
        model, input_size, warmup=10, iterations=100, device=device
    )
    print(f"  平均: {mean_lat:.2f}ms")
    print(f"  标准差: {std_lat:.2f}ms")
    print(f"  最小: {min_lat:.2f}ms")
    print(f"  最大: {max_lat:.2f}ms")
    
    # 4. 吞吐量测试
    print(f"\n[吞吐量测试]")
    throughput, batch_lat = measure_throughput(
        model, input_size, warmup=5, iterations=50, batch_size=8, device=device
    )
    print(f"  吞吐量: {throughput:.1f} img/s")
    print(f"  批处理延迟: {batch_lat:.2f}ms")
    
    # 5. 显存占用
    if device == 'cuda' and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"\n[显存占用]")
        print(f"  已分配: {allocated:.1f}MB")
        print(f"  已保留: {reserved:.1f}MB")
    
    print("\n" + "="*60)
    
    return {
        'model': model_name,
        'params_m': total_params / 1e6,
        'mean_latency_ms': mean_lat,
        'throughput_img_s': throughput,
        'model_size_mb': get_model_size(model_path) if model_path else None
    }


def main():
    parser = argparse.ArgumentParser(description='YOLOv11-ConvNeXt 性能基准测试')
    parser.add_argument('--model', type=str, default='convnext-small',
                       choices=['convnext-tiny', 'convnext-small', 'convnext-base'],
                       help='模型类型')
    parser.add_argument('--model-path', type=str, default=None,
                       help='模型权重路径 (可选)')
    parser.add_argument('--input-size', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--num-iterations', type=int, default=100,
                       help='测试迭代次数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备 (cuda/cpu)')
    args = parser.parse_args()
    
    print(f"\n设备: {args.device}")
    print(f"迭代次数: {args.num_iterations}")
    
    # 导入模型
    from models import YOLOv11WithConvNeXt
    
    # 创建模型
    model_type = args.model.replace('convnext-', '')
    model = YOLOv11WithConvNeXt(
        convnext_model_type=model_type,
        num_classes=80
    )
    
    if args.model_path and Path(args.model_path).exists():
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        print(f"已加载权重: {args.model_path}")
    
    model = model.to(args.device)
    model.eval()
    
    # 运行基准测试
    benchmark_model(
        model,
        model_name=args.model,
        model_path=args.model_path,
        input_size=args.input_size,
        device=args.device
    )


if __name__ == '__main__':
    main()
