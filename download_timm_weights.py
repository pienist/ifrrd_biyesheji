#!/usr/bin/env python3
"""下载 timm 预训练权重（使用国内镜像加速）"""

import os
import urllib.request

# 模型信息
MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"
OUTPUT_FILE = "convnext_base_in22k_ft_in1k.pth"

# 使用 hf-mirror 国内镜像
BASE_URL = "https://hf-mirror.com/timm/convnext_base.fb_in22k_ft_in1k/resolve/main"
FILE_URL = f"{BASE_URL}/pytorch_model.bin"

# 如果 safetensors 格式可用，改用这个（更快更安全）
# FILE_URL = f"{BASE_URL}/model.safetensors"
# OUTPUT_FILE = "convnext_base_in22k_ft_in1k.safetensors"

def download_with_progress(url, filepath):
    """带进度显示的下载函数"""
    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size)
            bar_length = 40
            filled = bar_length * percent // 100
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f'\r下载进度: [{bar}] {percent}% ({downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB)', end='', flush=True)
    
    if os.path.exists(filepath):
        print(f"文件已存在: {filepath}")
        response = input("是否重新下载？(y/n): ")
        if response.lower() != 'y':
            print("跳过下载")
            return False
    
    print(f"开始下载: {url}")
    print(f"保存至: {filepath}")
    print("-" * 60)
    
    urllib.request.urlretrieve(url, filepath, reporthook)
    print("\n" + "-" * 60)
    print(f"下载完成！文件大小: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
    return True

def verify_with_timm(filepath):
    """使用 timm 验证权重文件"""
    try:
        import timm
        
        print("\n正在验证权重文件...")
        
        # 检查文件是否存在
        ext = os.path.splitext(filepath)[1]
        if ext == '.safetensors':
            from safetensors.torch import load_file
            state_dict = load_file(filepath)
            print(f"  ✓ safetensors 格式加载成功，共 {len(state_dict)} 个参数")
        else:
            import torch
            state_dict = torch.load(filepath, map_location='cpu')
            if isinstance(state_dict, dict) and 'model' in state_dict:
                print(f"  ✓ checkpoint 格式加载成功")
            else:
                print(f"  ✓ state_dict 格式加载成功，共 {len(state_dict)} 个参数")
        
        # 尝试用 timm 加载
        print("\n尝试用 timm 加载模型...")
        model = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            checkpoint_path=filepath
        )
        print(f"  ✓ timm 模型加载成功!")
        print(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        return True
    except Exception as e:
        print(f"\n验证失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("  timm 权重下载工具 (hf-mirror 镜像)")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print()
    
    # 下载
    success = download_with_progress(FILE_URL, OUTPUT_FILE)
    
    if success:
        # 验证
        verify_with_timm(OUTPUT_FILE)
        
        print("\n" + "=" * 60)
        print("  下载并验证完成！")
        print("=" * 60)
        print("\n使用方法:")
        print("  model = timm.create_model(")
        print(f"      '{MODEL_NAME}',")
        print(f"      pretrained=True,")
        print(f"      checkpoint_path='{OUTPUT_FILE}'")
        print("  )")