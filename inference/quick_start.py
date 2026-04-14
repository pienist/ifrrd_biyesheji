# quick_start.py - 三分钟快速上手

import torch
from models import YOLOv11WithConvNeXt
from inference import YOLOv11ConvNeXtInference

# 1. 加载模型
model = YOLOv11WithConvNeXt(
    convnext_model_type='small',
    num_classes=80
)

# 2. 初始化推理引擎
detector = YOLOv11ConvNeXtInference(
    model_path='best_model.pth',
    device='cuda',
    half_precision=True
)

# 3. 运行推理
detections, inference_time = detector.inference('test_image.jpg')

# 4. 显示结果
print(f"检测到 {len(detections)} 个物体")
print(f"推理时间: {inference_time*1000:.2f}ms")

for detection in detections:
    print(f"  - {detection['class']}: {detection['confidence']:.2f}")
