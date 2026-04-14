#!/usr/bin/env python3
"""Detector class for YOLOv11-ConvNeXt."""

import torch
import cv2
from visualizer import Visualizer


class ConvNeXtDetector:
    """Object detector using YOLOv11-ConvNeXt model."""
    
    def __init__(self, weights_path, conf_threshold=0.25, iou_threshold=0.45):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.visualizer = Visualizer()
        
    def load_model(self):
        """Load model from weights."""
        raise NotImplementedError
        
    def detect(self, source):
        """Run detection on source."""
        raise NotImplementedError
