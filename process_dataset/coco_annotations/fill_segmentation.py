#!/usr/bin/env python3
"""
根据 stage1_640x640/masks 中的二值掩膜图像，为 coco_annotations/instances_train.json
补充 segmentation（多边形坐标）、更新 area 字段。

工作流程：
  1. 遍历每张有标注的图像，从对应掩膜中提取所有前景连通域
  2. 用 cv2.findContours 获取每个连通域的轮廓 (polygon)
  3. 将每个轮廓的坐标填入对应 annotation 的 segmentation
  4. 用掩膜像素数重新计算 area
  5. 输出新的 JSON 文件（inplace 覆盖，或保存为新文件）
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

# ============================================================
# 配置
# ============================================================
COCO_JSON_PATH = "/data1/undergraduate/ultralytics/coco_annotations/instances_train.json"
MASK_DIR = "/data1/undergraduate/ultralytics/stage1_640x640/masks"
# 过滤掉大于此面积阈值的连通域（sirstv2 数据集中掩膜边缘的伪影区域）
AREA_THRESHOLD = 2000
OUTPUT_JSON_PATH = "/data1/undergraduate/ultralytics/coco_annotations/instances_train_segmented.json"
# 设置为 True 则覆盖原文件，为 False 则保存为新文件
INPLACE = False

# ============================================================
# 辅助函数
# ============================================================

def mask_to_polygons(mask: np.ndarray) -> list[tuple[list[float], int]]:
    """
    将二值掩膜转换为 COCO segmentation 列表。

    Returns
    -------
    list of (polygon_coords, area)
        polygon_coords: [x1, y1, x2, y2, ...] (COCO RLE alternative: polygon)
        area: 前景像素数量（等同于 mask 面积）
    """
    foreground = (mask == 255).astype(np.uint8)  # 只取 target=255，忽略 128 的 ignore 区域
    area = int(foreground.sum())

    if area == 0:
        return []

    # 找连通域
    labels, n_components = ndimage.label(foreground)
    polygons = []

    for component_id in range(1, n_components + 1):
        component_mask = (labels == component_id).astype(np.uint8) * 255
        comp_area = int(np.sum(component_mask > 0))

        if comp_area == 0:
            continue

        # cv2.findContours 返回 (contours, hierarchy)
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            n_pts = contour.shape[0]
            if n_pts == 0:
                continue
            if n_pts < 3:
                # 超小目标（1-2 像素）：用最小外包矩形模拟一个矩形多边形
                x, y, w, h = cv2.boundingRect(contour)
                coords = [float(x), float(y),
                          float(x + w), float(y),
                          float(x + w), float(y + h),
                          float(x), float(y + h)]
                polygons.append((coords, comp_area))
            else:
                coords = contour.squeeze().flatten().tolist()
                polygons.append((coords, comp_area))

    # 过滤掉大面积伪影区域（仅对 sirstv2 等含边缘掩影的数据集必要）
    polygons = [(c, a) for c, a in polygons if a <= AREA_THRESHOLD]

    # 按面积从小到大排序，保证与 JSON annotations 按 area 升序的顺序一致
    polygons.sort(key=lambda p: p[1])

    return polygons


def compute_bbox_from_polygon(polygon: list[float]) -> tuple[float, float, float, float]:
    """
    从 polygon 坐标序列计算 axis-aligned bbox。
    返回 (x_min, y_min, width, height)，COCO bbox 格式。
    """
    xs = polygon[0::2]
    ys = polygon[1::2]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)


# ============================================================
# 主流程
# ============================================================

def main():
    # 1. 加载原始 JSON
    print(f"[1] 加载 {COCO_JSON_PATH} ...")
    with open(COCO_JSON_PATH, "r") as f:
        coco = json.load(f)

    original_images = coco["images"]
    original_annotations = coco["annotations"]

    print(f"    images: {len(original_images)}")
    print(f"    annotations (before): {len(original_annotations)}")

    # 2. 按 image_id 建立 original annotations 的索引
    # key: image_id -> list of annotation indices
    ann_idx_by_image = {}
    for idx, ann in enumerate(original_annotations):
        ann_idx_by_image.setdefault(ann["image_id"], []).append(idx)

    # 3. 建立 image_id -> file_name 的映射
    img_id_to_fname = {img["id"]: img["file_name"] for img in original_images}
    fname_to_img_id = {v: k for k, v in img_id_to_fname.items()}

    # 4. 收集所有掩膜文件，验证覆盖
    mask_files = set(os.listdir(MASK_DIR))
    json_files = set(fname_to_img_id.keys())
    missing_masks = json_files - mask_files
    missing_files = mask_files - json_files

    if missing_masks:
        print(f"\n[WARN] JSON 中有 {len(missing_masks)} 张图像在掩膜目录中找不到对应文件:")
        for f in sorted(missing_masks)[:5]:
            print(f"         {f}")
        print(f"         ... (共 {len(missing_masks)} 个)")
    if missing_files:
        print(f"\n[INFO] 掩膜目录中有 {len(missing_files)} 张文件未在 JSON 中注册 (可能是无标注图像):")
        for f in sorted(missing_files)[:5]:
            print(f"         {f}")
        print(f"         ... (共 {len(missing_files)} 个)")

    # 5. 遍历所有有标注的图像，处理掩膜
    updated_annotations = []
    stats = {
        "processed_images": 0,
        "skipped_no_ann": 0,
        "skipped_no_mask": 0,
        "no_polygons": 0,
        "total_polygons": 0,
        "filtered_large": 0,
        "mismatch": 0,
    }

    for img_id, file_name in img_id_to_fname.items():
        ann_indices = ann_idx_by_image.get(img_id, [])

        # 无标注的图像 — 跳过
        if not ann_indices:
            stats["skipped_no_ann"] += 1
            continue

        # 无掩膜的图像 — 跳过（保留原 annotation）
        mask_path = os.path.join(MASK_DIR, file_name)
        if not os.path.exists(mask_path):
            stats["skipped_no_mask"] += 1
            for idx in ann_indices:
                updated_annotations.append(original_annotations[idx])
            continue

        # 加载掩膜
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]  # 取第一个通道

        # 提取多边形（已过滤大面积伪影、按面积升序排序）
        polygons = mask_to_polygons(mask)
        stats["total_polygons"] += len(polygons)

        if len(polygons) == 0:
            # 有标注但掩膜全黑 — 保留原 annotation，仅标记
            stats["no_polygons"] += 1
            for idx in ann_indices:
                updated_annotations.append(original_annotations[idx])
            continue

        # 连通域数量与 annotation 数量是否一致
        # 排序后再比对：polygons 按 area 升序；annotations 也按 area 升序，保证配对正确
        if len(polygons) != len(ann_indices):
            stats["mismatch"] += 1
            if stats["mismatch"] <= 5:
                print(f"[MISMATCH] {file_name}: polygons={len(polygons)} vs annotations={len(ann_indices)}")

        stats["processed_images"] += 1

        # 将该图的 annotation 按 area 升序排序，再与 polygon 配对
        sorted_ann_indices = sorted(ann_indices, key=lambda idx: original_annotations[idx]["area"])

        for i, idx in enumerate(sorted_ann_indices):
            orig_ann = original_annotations[idx]

            if i < len(polygons):
                poly_coords, poly_area = polygons[i]
                x, y, w, h = compute_bbox_from_polygon(poly_coords)
                updated_ann = {
                    **orig_ann,
                    "segmentation": [poly_coords],
                    "bbox": [x, y, w, h],
                    "area": float(poly_area),
                }
            else:
                # polygon 不够用，保留原 annotation（不应该发生）
                updated_ann = orig_ann

            updated_annotations.append(updated_ann)

    # 6. 写回 JSON
    output_path = OUTPUT_JSON_PATH if not INPLACE else COCO_JSON_PATH
    if INPLACE:
        backup_path = COCO_JSON_PATH + ".backup"
        with open(backup_path, "w") as f:
            json.dump(coco, f)
        print(f"\n[BACKUP] 原文件已备份为 {backup_path}")

    coco["annotations"] = updated_annotations

    print(f"\n[2] 写入 {output_path} ...")
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    # 7. 打印统计
    print("\n" + "=" * 60)
    print("处理完成！统计摘要：")
    print("=" * 60)
    print(f"  处理图像数（含标注）: {stats['processed_images']}")
    print(f"  跳过（无标注）:       {stats['skipped_no_ann']}")
    print(f"  跳过（无掩膜）:       {stats['skipped_no_mask']}")
    print(f"  有标注但掩膜全黑:     {stats['no_polygons']}")
    print(f"  polygon/annotation 数量不匹配: {stats['mismatch']}")
    print(f"  过滤后 polygon 总数:  {stats['total_polygons']}")
    print(f"  更新后 annotation 数:  {len(updated_annotations)}")

    # 8. 验证输出
    print("\n" + "=" * 60)
    print("快速验证（抽样前 5 条有 segmentation 的 annotation）:")
    print("=" * 60)
    count = 0
    for ann in updated_annotations:
        if ann.get("segmentation") and ann["segmentation"]:
            poly = ann["segmentation"][0]
            print(f"  ann id={ann['id']:5d}  image_id={ann['image_id']:5d}  "
                  f"bbox={[round(v,1) for v in ann['bbox']]}  "
                  f"area={ann['area']:7.1f}  "
                  f"seg_len={len(poly)//2} pts")
            count += 1
            if count >= 5:
                break

    print(f"\n✅ 完成！输出文件: {output_path}")


if __name__ == "__main__":
    main()
