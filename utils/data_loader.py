#!/usr/bin/env python3
"""Data loading utilities for YOLOv11-ConvNeXt."""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2


class YOLODataset(Dataset):
    """Dataset loader for YOLO format data."""
    
    def __init__(self, img_dir, label_dir=None, img_size=640, augment=False):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        self.img_size = img_size
        self.augment = augment
        self.img_files = sorted(list(self.img_dir.glob('*.[jp][pn][g]')))
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        raise NotImplementedError
        
    def load_image(self, idx):
        """Load image from file."""
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        return img


def create_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    """Create data loader from dataset."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
