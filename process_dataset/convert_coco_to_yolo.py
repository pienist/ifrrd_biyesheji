#!/usr/bin/env python3
"""
COCO RLE 分割格式转 YOLO 分割格式.
===================================
将 COCO JSON 标注（RLE 格式）转换为 YOLO txt 格式
使用 OpenCV 提取轮廓

使用方法:
    python convert_coco_to_yolo.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util


def rle_to_polygon(rle, img_height, img_width):
    """将 RLE 格式转换为 polygon 点列表 使用 OpenCV 提取轮廓."""
    # 解码 RLE 为二进制 mask
    binary_mask = mask_util.decode(rle).astype(np.uint8)

    # 找到轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        # 简化轮廓点（减少点数）
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 转换为 [x1,y1,x2,y2,...] 格式
        # 注意：OpenCV 的点是 (x, y) 格式
        points = []
        for point in approx:
            x = point[0][0] / img_width
            y = point[0][1] / img_height
            points.extend([x, y])

        if len(points) >= 6:  # 至少3个点
            polygons.append(points)

    return polygons


def polygon_to_yolo(polygon, img_height, img_width):
    """将 COCO polygon 点列表转换为 YOLO 归一化格式 polygon: [x1, y1, x2, y2, ...] 像素绝对坐标 返回: [x1, y1, x2, y2, ...] 归一化坐标 (0~1)."""
    coords = []
    for i in range(0, len(polygon), 2):
        x = polygon[i] / img_width
        y = polygon[i + 1] / img_height
        coords.extend([x, y])
    return coords


def convert_coco_to_yolo(coco_json_path, output_dir):
    """将 COCO polygon 格式标注转换为 YOLO 分割格式."""
    # 加载 COCO JSON
    with open(coco_json_path) as f:
        coco = json.load(f)

    print(f"  加载 {len(coco['images'])} 张图像, {len(coco['annotations'])} 个标注")

    # 创建图像 id -> (文件名, 宽, 高) 映射
    img_info = {img["id"]: (img["file_name"], img["width"], img["height"]) for img in coco["images"]}

    # 按图像分组标注
    annotations_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    error_count = 0
    skipped_rle = 0  # 遇到的 dict 类型（RLE）数量

    # 转换每个图像的标注
    for img_id, annotations in annotations_by_image.items():
        file_name, img_width, img_height = img_info[img_id]

        # YOLO 标签文件路径
        label_file = output_dir / (Path(file_name).stem + ".txt")

        with open(label_file, "w") as f:
            for ann in annotations:
                category_id = ann["category_id"] - 1  # COCO 从 1 开始，YOLO 从 0 开始

                segmentation = ann.get("segmentation", [])

                # polygon 格式: list of [x1,y1,x2,y2,...]
                if isinstance(segmentation, list):
                    for polygon in segmentation:
                        if len(polygon) < 6:
                            continue
                        yolo_coords = polygon_to_yolo(polygon, img_height, img_width)
                        coords_str = " ".join([f"{c:.6f}" for c in yolo_coords])
                        f.write(f"{category_id} {coords_str}\n")

                # RLE 格式: dict（使用 pycocotools 解码）
                elif isinstance(segmentation, dict):
                    try:
                        polygons = rle_to_polygon(segmentation, img_height, img_width)
                        for polygon in polygons:
                            if len(polygon) < 6:
                                continue
                            coords_str = " ".join([f"{c:.6f}" for c in polygon])
                            f.write(f"{category_id} {coords_str}\n")
                    except Exception:
                        error_count += 1
                        skipped_rle += 1
                        continue
                else:
                    error_count += 1
                    continue

        converted_count += 1

    print(f"  转换完成: {converted_count} 个文件, {error_count} 个错误（其中 RLE 格式 {skipped_rle} 个）")
    return converted_count


def main():
    # 配置路径
    base_dir = Path("/data1/undergraduate/ultralytics/dataset_coco")
    annotations_dir = base_dir / "annotations"

    # 输出目录
    output_base = base_dir / "labels"

    splits = ["train", "val", "test"]

    total_converted = 0

    for split in splits:
        print(f"\n{'=' * 50}")
        print(f"转换 {split} 数据集...")

        coco_json = annotations_dir / f"{split}.json"
        if not coco_json.exists():
            print(f"  跳过: {coco_json} 不存在")
            continue

        output_dir = output_base / split
        count = convert_coco_to_yolo(coco_json, output_dir)
        total_converted += count

    print(f"\n{'=' * 50}")
    print(f"总共转换 {total_converted} 个标注文件")
    print(f"标签保存在: {output_base}")

    # 验证转换结果
    print("\n验证转换结果:")
    for split in splits:
        label_dir = output_base / split
        if label_dir.exists():
            label_files = list(label_dir.glob("*.txt"))
            label_count = len(label_files)
            # 检查是否有内容
            non_empty = 0
            for lf in label_files[:10]:
                if lf.stat().st_size > 0:
                    non_empty += 1
            print(f"  {split}: {label_count} 个标签文件 (前10个中{non_empty}个非空)")


if __name__ == "__main__":
    main()
