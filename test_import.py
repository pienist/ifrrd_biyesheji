#!/usr/bin/env python
"""诊断脚本"""
import sys
import os

print("=== 诊断开始 ===")
print(f"Python: {sys.version}")
print(f"工作目录: {os.getcwd()}")

# 检查文件是否存在
for f in ['models/__init__.py', 'models/yolov11_head.py', 'models/convnext_backbone.py']:
    exists = os.path.exists(f)
    print(f"文件存在 {f}: {exists}")

print("\n=== 测试导入 ===")
try:
    print("导入 models.yolov11_head...")
    import models.yolov11_head
    print("成功!")
except Exception as e:
    print(f"失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
