#!/usr/bin/env python3
"""Inference script for YOLOv11-ConvNeXt."""

import argparse
from detector import ConvNeXtDetector


def main():
    parser = argparse.ArgumentParser(description='Run inference with YOLOv11-ConvNeXt')
    parser.add_argument('--weights', type=str, required=True, help='Model weights path')
    parser.add_argument('--source', type=str, required=True, help='Input image/video path')
    parser.add_argument('--output', type=str, default='results/predictions', help='Output directory')
    args = parser.parse_args()
    
    detector = ConvNeXtDetector(args.weights)
    results = detector.detect(args.source)
    print(f"Detected {len(results)} objects")


if __name__ == '__main__':
    main()
