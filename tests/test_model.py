# Tests for YOLOv11-ConvNeXt models

import pytest
import torch
from models.yolov11_convnext import YOLOv11ConvNeXt


def test_model_creation():
    """Test model creation."""
    model = YOLOv11ConvNeXt(num_classes=80)
    assert model is not None
    assert model.num_classes == 80


def test_model_forward():
    """Test model forward pass."""
    model = YOLOv11ConvNeXt(num_classes=80)
    x = torch.randn(1, 3, 640, 640)
    # Add your forward assertion here
    # output = model(x)
    # assert output.shape[0] == 1


def test_model_output_shape():
    """Test model output shape."""
    model = YOLOv11ConvNeXt(num_classes=80)
    x = torch.randn(1, 3, 640, 640)
    # Add shape validation
    pass


if __name__ == '__main__':
    pytest.main([__file__])
