#!/usr/bin/env python3
"""
根据 instances_train.json 生成预览图片（左右分屏）：
  左侧：原图 + 红色 bbox 框（来自 JSON）
  右侧：原图 + 绿色 mask 轮廓（来自 mask 图像）

mask 像素值约定（与 generate_coco_bbox.py 一致）：
  sirstv2 : target=255,  ignore=128（真目标=255，填充/忽略区域=128）
  irstd1k: target=255,  ignore=0（无 ignore）
  nudt   : target=255,  ignore=0（无 ignore）

用法：
  python3 generate_preview.py                    # 重新生成全部 preview
  python3 generate_preview.py sirstv2_000738     # 只生成指定图片
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ── 路径配置 ──────────────────────────────────────────────
STAGE1_DIR   = Path("/data1/undergraduate/ultralytics/stage1_640x640")
IMG_DIR      = STAGE1_DIR / "images"
MASK_DIR     = STAGE1_DIR / "masks"
PREVIEW_DIR  = Path("/data1/undergraduate/ultralytics/coco_annotations/preview")
COCO_JSON    = Path("/data1/undergraduate/ultralytics/coco_annotations/instances_train.json")

# ── 颜色配置 ──────────────────────────────────────────────
COLOR_BBOX = (0, 0, 255)       # 红  – bbox（来自 JSON）
COLOR_MASK = (0, 255, 0)       # 绿  – mask 轮廓（来自 mask 图像）

# ── mask 值映射（与 generate_coco_bbox.py 保持一致）────────
DATASET_MASK_CONFIG = {
    "sirstv2": {"target": 255},
    "irstd1k": {"target": 255},
    "nudt":    {"target": 255},
}


def get_target_value(fname: str) -> int:
    """根据文件名返回该数据集的 target 像素值。"""
    for prefix, cfg in DATASET_MASK_CONFIG.items():
        if fname.startswith(prefix):
            return cfg["target"]
    return 255  # 默认


# ─────────────────────────────────────────────────────────
# 加载 COCO JSON
# ─────────────────────────────────────────────────────────

def load_coco():
    with open(COCO_JSON, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────
# 绘制 bbox（红框）
# ─────────────────────────────────────────────────────────

def draw_bboxes(overlay: np.ndarray, annotations: list):
    """在 overlay 上绘制所有 bbox（红框），标注类别名。"""
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w <= 0 or h <= 0:
            continue
        cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_BBOX, 2)

        label = "infrared_target"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(overlay, (x, y - th - 6), (x + tw, y), COLOR_BBOX, -1)
        cv2.putText(overlay, label, (x, y - 2),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


# ─────────────────────────────────────────────────────────
# 绘制 mask 轮廓（绿框）
# ─────────────────────────────────────────────────────────

def draw_mask_contour(overlay: np.ndarray, mask: np.ndarray, target_val: int):
    """
    在 overlay 上绘制 mask 目标区域的绿色轮廓（只绘制 target_val 像素）。
    mask 轮廓用 cv2.findContours 提取，再画到 overlay 上。
    """
    binary = ((mask == target_val) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    cv2.drawContours(overlay, contours, -1, COLOR_MASK, 2)


# ─────────────────────────────────────────────────────────
# 合成左右分屏预览图
# ─────────────────────────────────────────────────────────

def generate_preview(orig_fname: str, coco: dict) -> bool:
    """
    为一张图片生成分屏 preview：
      左半 = 原图 + 红 bbox
      右半 = 原图 + 绿 mask 轮廓
    """
    if not orig_fname.endswith(".png"):
        orig_fname += ".png"

    # ── 1. 找 JSON entry ──
    img_entry = next(
        (im for im in coco["images"] if im["file_name"] == orig_fname), None)
    if img_entry is None:
        print(f"  ⚠  {orig_fname}: 未在 instances_train.json 中找到，跳过")
        return False

    img_id = img_entry["id"]
    anns = [a for a in coco["annotations"] if a["image_id"] == img_id]

    # ── 2. 加载原图 ──
    img_path = IMG_DIR / orig_fname
    if not img_path.exists():
        print(f"  ⚠  {orig_fname}: 找不到原图 {img_path}，跳过")
        return False

    orig = np.array(Image.open(img_path).convert("RGB"))
    H, W = orig.shape[:2]

    # ── 3. 加载 mask，确定 target 值 ──
    mask_path = MASK_DIR / orig_fname
    target_val = get_target_value(orig_fname)
    if mask_path.exists():
        mask = np.array(Image.open(mask_path))
    else:
        mask = np.zeros((H, W), dtype=np.uint8)

    # ── 4. 构建左图（红 bbox）──
    left = orig.copy()
    draw_bboxes(left, anns)

    # ── 5. 构建右图（绿 mask 轮廓）──
    right = orig.copy()
    draw_mask_contour(right, mask, target_val)

    # ── 6. 水平拼接，添加分隔线 ──
    canvas = np.hstack([left, right])
    separator = np.full((H, 4, 3), 200, dtype=np.uint8)  # 灰色分隔线
    canvas = np.hstack([left, separator, right])

    # ── 7. 保存 ──
    preview_name = f"preview_{orig_fname}"
    out_path = PREVIEW_DIR / preview_name
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    print(f"  ✓ {orig_fname}: bbox={len(anns)} → {preview_name}")
    return True


# ─────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────

def main():
    coco = load_coco()

    if len(sys.argv) > 1:
        # 命令行指定文件名（支持 preview_ 前缀或不带 .png）
        names = []
        for a in sys.argv[1:]:
            name = a
            if name.startswith("preview_"):
                name = name[len("preview_"):]
            if not name.endswith(".png"):
                name += ".png"
            names.append(name)
    else:
        # 默认：所有已存在的 preview 文件
        names = [p.name[len("preview_"):] for p in sorted(PREVIEW_DIR.glob("preview_*.png"))]

    print(f"将处理 {len(names)} 张 preview 图片...\n")
    success = sum(generate_preview(n, coco) for n in names)
    print(f"\n完成！成功 {success}/{len(names)} 张")


if __name__ == "__main__":
    main()
