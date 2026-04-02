#!/usr/bin/env python3
"""
从 stage1_640x640 的 masks 生成 COCO 格式 bbox 标注。.

mask 像素值约定：
  sirstv2 : target=255,  ignore=128（真目标=255，填充/忽略区域=128）
  irstd1k: target=255,  ignore=0（无 ignore）
  nudt   : target=255,  ignore=0（无 ignore）

COCO bbox 格式：[x_min, y_min, width, height]（绝对像素，非归一化）
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
STAGE1_DIR = Path("/data1/undergraduate/ultralytics/stage1_640x640")
IMG_DIR = STAGE1_DIR / "images"
MASK_DIR = STAGE1_DIR / "masks"
OUTPUT_DIR = Path("/data1/undergraduate/ultralytics/coco_annotations")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

CATEGORY_NAME = "infrared_target"
SPLIT = "train"

# ─────────────────────────────────────────────
# mask 值映射：每个子数据集的 target 和 ignore 值
# ─────────────────────────────────────────────
# TARGET_VALUE  → 用于计算 bbox 的像素值（连通域分析）
# IGNORE_VALUE  → 忽略区域（不参与 bbox 计算，也不作为目标）
DATASET_MASK_CONFIG = {
    "sirstv2": {"target": 255, "ignore": 128},  # target=真目标, ignore=填充/忽略区域
    "irstd1k": {"target": 255, "ignore": None},
    "nudt": {"target": 255, "ignore": None},
}


def get_mask_config(fname: str) -> dict:
    """根据文件名确定 mask 值配置。."""
    for prefix, cfg in DATASET_MASK_CONFIG.items():
        if fname.startswith(prefix):
            return cfg
    return {"target": 255, "ignore": None}  # 默认


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────


def find_instances(mask: np.ndarray, target_val: int, ignore_val: int | None = None):
    """在 mask == target_val 的像素中寻找所有连通域（实例）。 ignore_val 像素被当作背景排除（即使值非零）。 返回 list of (inst_mask, rmin, cmin, rmax,
    cmax)。.
    """
    target_pixels = mask == target_val
    if ignore_val is not None:
        target_pixels = target_pixels & (mask != ignore_val)
    labeled, num = ndimage.label(target_pixels)
    instances = []
    for i in range(1, num + 1):
        inst_mask = labeled == i
        rows = np.any(inst_mask, axis=1)
        cols = np.any(inst_mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            continue
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        instances.append((inst_mask, rmin, cmin, rmax, cmax))
    return instances


def mask_to_bbox(rmin: int, cmin: int, rmax: int, cmax: int):
    """[x_min, y_min, width, height]，x=cmin, y=rmin."""
    return float(cmin), float(rmin), float(cmax - cmin + 1), float(rmax - rmin + 1)


# ─────────────────────────────────────────────
# 主逻辑
# ─────────────────────────────────────────────


def build_coco_annotations():
    img_files = sorted(IMG_DIR.glob("*.png"))
    print(f"共找到 {len(img_files)} 张图片")

    empty_files = []
    stats = defaultdict(int)

    # ── 1. 构建 COCO 结构 ──
    info = {
        "year": 2026,
        "version": "1.0",
        "description": "Infrared small target detection dataset (stage1_640x640)",
        "contributor": "Ultralytics",
        "url": "",
        "date_created": "2026-04-02",
    }
    licenses = [{"id": 1, "name": "Unknown", "url": ""}]
    categories = [{"id": 1, "name": CATEGORY_NAME, "supercategory": "object"}]

    images_list = []
    annotations_list = []
    ann_id = 1

    for img_idx, img_path in enumerate(img_files):
        fname = img_path.name
        cfg = get_mask_config(fname)

        # ── 加载 mask ──
        mask = np.array(Image.open(MASK_DIR / fname))
        h, w = mask.shape
        coco_img_id = img_idx + 1

        images_list.append(
            {
                "id": coco_img_id,
                "file_name": fname,
                "width": w,
                "height": h,
                "date_captured": "",
                "license": 1,
                "coco_url": "",
                "flickr_url": "",
                "seg_file_name": fname,
            }
        )

        # ── 找所有目标连通域（只从 target_val 像素计算 bbox，忽略 ignore_val）──
        instances = find_instances(mask, cfg["target"], cfg.get("ignore"))

        if not instances:
            empty_files.append(fname)
            stats["empty"] += 1
        else:
            stats["has_target"] += 1

        # ── 写 annotation ──
        for inst_mask, rmin, cmin, rmax, cmax in instances:
            x, y, bw, bh = mask_to_bbox(rmin, cmin, rmax, cmax)
            area = float(np.sum(inst_mask))
            if area <= 0:
                continue
            annotations_list.append(
                {
                    "id": ann_id,
                    "image_id": coco_img_id,
                    "category_id": 1,
                    "bbox": [x, y, bw, bh],
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            ann_id += 1

    # ── 保存 ──
    coco = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": images_list,
        "annotations": annotations_list,
    }
    out_path = OUTPUT_DIR / f"instances_{SPLIT}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    # ── 统计 ──
    print(f"\n✓ 已保存: {out_path}")
    print(f"  images       : {len(images_list)}")
    print(f"  annotations : {len(annotations_list)}")
    print(f"  空掩码图    : {len(empty_files)}")
    print(f"  有目标图    : {stats['has_target']}")

    if annotations_list:
        areas = [a["area"] for a in annotations_list]
        bboxes = [a["bbox"] for a in annotations_list]
        ws = [b[2] for b in bboxes]
        hs = [b[3] for b in bboxes]
        print("\n=== 标注统计 ===")
        print(f"area   min={min(areas):.1f}  max={max(areas):.1f}  mean={np.mean(areas):.1f}")
        print(f"bbox w min={min(ws):.1f}  max={max(ws):.1f}  mean={np.mean(ws):.1f}")
        print(f"bbox h min={min(hs):.1f}  max={max(hs):.1f}  mean={np.mean(hs):.1f}")
        print(f"空掩码样例: {empty_files[:5]}{'...' if len(empty_files) > 5 else ''}")


if __name__ == "__main__":
    build_coco_annotations()
