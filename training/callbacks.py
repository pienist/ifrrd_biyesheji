#!/usr/bin/env python3
"""Training callbacks for YOLOv11-ConvNeXt."""


class TrainerCallback:
    """Callback handler for training events."""
    
    def on_train_start(self, trainer):
        pass
    
    def on_train_epoch_start(self, trainer, epoch):
        pass
    
    def on_train_batch_end(self, trainer, batch, output):
        pass
    
    def on_train_epoch_end(self, trainer, epoch, metrics):
        pass
    
    def on_train_end(self, trainer):
        pass
