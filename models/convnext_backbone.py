#!/usr/bin/env python3
"""ConvNeXt backbone for YOLOv11 adaptation."""

import torch
import torch.nn as nn


class ConvNeXtBackbone(nn.Module):
    """ConvNeXt backbone module."""
    
    def __init__(self, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()
        self.depths = depths
        self.dims = dims
        
    def forward(self, x):
        raise NotImplementedError
