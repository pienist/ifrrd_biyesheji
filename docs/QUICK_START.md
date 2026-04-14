# Quick Start Guide

## Training

Train a model on your dataset:

```bash
python training/train.py --data your_data.yaml --epochs 100 --batch-size 16
```

## Inference

Run inference on images:

```bash
python inference/inference.py --weights path/to/weights.pt --source path/to/image.jpg
```

## Export

Export model for deployment:

```bash
python deployment/export.py --weights path/to/weights.pt --format onnx
```

## Deployment

Start inference server:

```bash
python deployment/server.py --model path/to/weights.pt --port 8000
```

## Using Pre-trained Models

Download pre-trained weights and use:

```python
from inference.detector import ConvNeXtDetector

detector = ConvNeXtDetector('yolov11-convnext-tiny.pt')
results = detector.detect('image.jpg')
```

## Configuration

Edit `configs/training_config.yaml` to customize training parameters.
