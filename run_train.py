#!/usr/bin/env python
"""Wrapper to run training with local ultralytics"""
import sys

# Add local ultralytics to path FIRST
_local_ultralytics = '/data1/undergraduate/ultralytics'
if _local_ultralytics not in sys.path:
    sys.path.insert(0, _local_ultralytics)

# Now run the actual training script
if __name__ == '__main__':
    # Use exec to run the script in the same namespace
    import os
    os.chdir('/data1/undergraduate/ultralytics')
    exec(open('/data1/undergraduate/ultralytics/train_convnext_seg_B.py').read().replace(
        "from ultralytics.nn.tasks import SegmentationModel",
        "from ultralytics.nn.tasks import SegmentationModel"
    ).replace(
        "from ultralytics import YOLO",
        "from ultralytics.models.yolo import YOLO"
    ))
