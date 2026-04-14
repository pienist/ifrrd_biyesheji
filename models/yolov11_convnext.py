#!/usr/bin/env python3
"""YOLOv11 with ConvNeXt backbone complete model."""

import torch
import torch.nn as nn
from .convnext_backbone import ConvNeXtBackbone


class YOLOv11ConvNeXt(nn.Module):
    """YOLOv11 model with ConvNeXt backbone."""
    
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = ConvNeXtBackbone()
        self.num_classes = num_classes
        
    def forward(self, x):
        raise NotImplementedError
