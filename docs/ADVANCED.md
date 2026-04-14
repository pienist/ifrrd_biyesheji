# Advanced Usage Guide

## Custom Model Configuration

Create a custom model configuration in `configs/`:

```yaml
model:
  backbone:
    type: convnext_custom
    depths: [2, 2, 6, 2]
    dims: [64, 128, 256, 512]
```

## Mixed Precision Training

Enable FP16 training for faster training:

```python
trainer = YOLOv11Trainer(args)
trainer.use_amp = True
trainer.train()
```

## Multi-GPU Training

```bash
python -m torch.distributed.launch --nproc_per_node=4 training/train.py --data your_data.yaml
```

## Model Quantization

For faster inference on edge devices:

```python
from deployment.quantize import quantize_dynamic

quantized_model = quantize_dynamic(model)
```

## Advanced Data Augmentation

Configure augmentation in your config:

```yaml
augmentation:
  mosaic: 1.0
  mixup: 0.2
  copy_paste: 0.1
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
```

## Custom Callbacks

```python
from training.callbacks import TrainerCallback

class MyCallback(TrainerCallback):
    def on_train_epoch_end(self, trainer, epoch, metrics):
        # Custom logic
        pass
```
