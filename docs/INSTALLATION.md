# Installation Guide

## Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.1+ (for GPU support)

## Basic Installation

```bash
pip install -r requirements.txt
```

## Development Installation

```bash
git clone <repository-url>
cd yolov11-convnext
pip install -e .
```

## Docker Installation

```bash
cd deployment/docker
docker build -t yolov11-convnext .
docker run --gpus all yolov11-convnext
```

## Verification

To verify your installation:

```bash
python -c "from models import YOLOv11ConvNeXt; print('Installation successful!')"
```
