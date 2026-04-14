#!/usr/bin/env python3
"""Performance metrics for evaluation."""

import numpy as np


def calculate_map(predictions, ground_truths):
    """Calculate mean Average Precision."""
    raise NotImplementedError


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes."""
    raise NotImplementedError


def calculate_precision_recall(tp, fp, fn):
    """Calculate precision and recall."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall
