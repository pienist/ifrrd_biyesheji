#!/usr/bin/env python3
"""Training script for YOLOv11-ConvNeXt."""

import argparse
import torch
from trainer import YOLOv11Trainer


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11-ConvNeXt')
    parser.add_argument('--data', type=str, required=True, help='Dataset yaml path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size')
    args = parser.parse_args()
    
    trainer = YOLOv11Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
