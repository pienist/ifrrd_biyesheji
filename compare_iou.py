from ultralytics import YOLO

model = YOLO('runs/segment/irstd_yolo11s_seg_imagenet_pretained_200epoch/weights/best.pt')

# 寻找合适的NMS IoU阈值
iou_values = [0.35, 0.45, 0.5, 0.55, 0.65]

for iou in iou_values:
    metrics = model.val(iou=iou, split='test', verbose=False)
    print(f"IoU={iou}: Recall={float(metrics.box.r):.4f}, Precision={float(metrics.box.p):.4f}, mAP50={float(metrics.box.map50):.4f}, mAP50-95={float(metrics.box.map):.4f}")