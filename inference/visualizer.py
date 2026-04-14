#!/usr/bin/env python3
"""Visualization tools for detection results."""

import cv2
import numpy as np


class Visualizer:
    """Visualizer for detection results."""
    
    def __init__(self):
        self.colors = self._generate_colors(80)
        
    def _generate_colors(self, n):
        """Generate n distinct colors."""
        return [(np.random.randint(0, 255) for _ in range(3)) for _ in range(n)]
    
    def draw_boxes(self, image, boxes, labels, scores):
        """Draw bounding boxes on image."""
        raise NotImplementedError
        
    def save_result(self, image, output_path):
        """Save visualization result."""
        raise NotImplementedError
