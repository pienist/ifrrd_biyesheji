#!/usr/bin/env python3
"""
四象限预览：验证 instances_train.json 的标注质量（不看 masks 文件夹，仅交叉对比）。.

  左上：从 JSON segmentation 多边形推理出的掩膜（绿），叠在原图上
  左下：从 JSON bbox（蓝框），叠在原图上
  右上：masks 文件夹里的真实掩膜（灰阶），叠在原图上
  右下：原图

用法：
  python3 generate_preview_coco.py                    # 抽样 20 张
  python3 generate_preview_coco.py sirstv2_000001   # 指定图片
  python3 generate_preview_coco.py --all            # 全部图片
  python3 generate_preview_coco.py --stats           # 仅统计，不生成图片
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ── 路径配置 ──────────────────────────────────────────────
STAGE1_DIR = Path("/data1/undergraduate/ultralytics/stage1_640x640")
IMG_DIR = STAGE1_DIR / "images"
MASK_DIR = STAGE1_DIR / "masks"
OUTPUT_DIR = Path("/data1/undergraduate/ultralytics/coco_annotations/preview_coco")
COCO_JSON = Path("/data1/undergraduate/ultralytics/coco_annotations/instances_train_segmented.json")

# ── 颜色配置 ──────────────────────────────────────────────
COLOR_SEG = (0, 255, 0)  # 绿  – segmentation 推理掩膜
COLOR_BBOX = (255, 0, 0)  # 蓝  – bbox


# ─────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────


def polygons_to_mask(polygons, H, W):
    """将 COCO segmentation 多边形列表转换为二值掩膜图像。 polygons: list of [x1,y1,x2,y2,...] 返回 HxW np.uint8，255 表示目标区域."""
    mask = np.zeros((H, W), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def draw_bboxes(overlay, annotations, color=COLOR_BBOX):
    """在图像上绘制所有 bbox。."""
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w <= 0 or h <= 0:
            continue
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)


# ─────────────────────────────────────────────────────────
# 单图生成
# ─────────────────────────────────────────────────────────


def generate_one(fname: str, coco: dict) -> bool:
    """生成一张四象限预览图。."""
    if not fname.endswith(".png"):
        fname += ".png"

    # 1. 找 JSON 条目
    img_entry = next((im for im in coco["images"] if im["file_name"] == fname), None)
    if img_entry is None:
        print(f"  ⚠  {fname}: 未在 instances_train.json 中找到，跳过")
        return False

    img_id = img_entry["id"]
    img_h, img_w = img_entry["height"], img_entry["width"]
    anns = [a for a in coco["annotations"] if a["image_id"] == img_id]

    # 2. 加载原图
    img_path = IMG_DIR / fname
    if not img_path.exists():
        print(f"  ⚠  {fname}: 找不到原图 {img_path}，跳过")
        return False
    orig = np.array(Image.open(img_path).convert("RGB"))

    # 3. 加载真实 mask
    mask_path = MASK_DIR / fname
    if mask_path.exists():
        real_mask = np.array(Image.open(mask_path))
        if real_mask.ndim == 3:
            real_mask = real_mask[:, :, 0]
    else:
        real_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # 4. 从 JSON 推理 segmentation 掩膜（绿）
    seg_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for ann in anns:
        segs = ann.get("segmentation", [])
        for poly in segs:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(seg_mask, [pts], 255)

    # 5. 绘制四象限
    H, W = orig.shape[:2]
    CW = W * 2 + 4  # 含分隔线的整图宽度
    H * 2 + 4  # 含分隔线的整图高度

    # 左上：原图 + 绿 segmentation 掩膜
    tl = orig.copy()
    mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
    mask_vis[:, :, 1] = seg_mask  # 绿通道
    mask_contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_vis, mask_contours, -1, (0, 255, 0), 1)
    tl = cv2.addWeighted(tl, 0.6, mask_vis, 0.4, 0)

    # 左下：原图 + 蓝 bbox
    bl = orig.copy()
    draw_bboxes(bl, anns, color=COLOR_BBOX)

    # 右上：纯 mask 原图（0/128/255）
    tr = np.zeros((H, W, 3), dtype=np.uint8)
    tr[:, :, 1] = real_mask  # 绿色通道映射灰度值

    # 右下：原图
    br = orig.copy()

    # ── 组装四象限 ──
    sep_v = np.full((H, 4, 3), 180, dtype=np.uint8)  # 垂直分隔线
    sep_h = np.full((4, CW, 3), 180, dtype=np.uint8)  # 水平分隔线

    row_top = np.hstack([tl, sep_v, tr])
    row_bottom = np.hstack([bl, sep_v, br])
    canvas = np.vstack([row_top, sep_h, row_bottom])

    # 文本标注（坐标基于拼接后的画布）
    cv2.putText(canvas, "seg from JSON (green)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_SEG, 1)
    cv2.putText(canvas, "mask from folder", (W + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(canvas, f"bbox (n={len(anns)})", (8, H + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BBOX, 1)
    cv2.putText(canvas, "original image", (W + 12, H + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── 保存 ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"coco_preview_{fname}"
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"  ✓ {fname} → {out_path.name}")
    return True


# ─────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="生成 COCO 标注四象限预览")
    parser.add_argument("--stats", action="store_true", help="仅打印统计，不生成图片")
    parser.add_argument("--all", action="store_true", help="处理全部图片")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--n", type=int, default=20, help="抽样数量（默认 20）")
    parser.add_argument("names", nargs="*", help="指定文件名（不含 .png）")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(COCO_JSON, encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    anns = coco["annotations"]

    # ── 统计 ──
    ann_by_img = {}
    for a in anns:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    # 检查 images / masks / images 目录一致性
    img_ids = {im["id"] for im in images}
    json_files = {im["file_name"] for im in images}
    mask_files = {f.name for f in MASK_DIR.glob("*.png")} if MASK_DIR.exists() else set()
    img_files = {f.name for f in IMG_DIR.glob("*.png")} if IMG_DIR.exists() else set()

    print("=" * 60)
    print("COCO JSON vs stage1_640x640 一致性检查")
    print("=" * 60)
    print(f"  JSON images:   {len(json_files)}")
    print(f"  IMG files:     {len(img_files)}")
    print(f"  MASK files:    {len(mask_files)}")

    only_json = json_files - mask_files
    only_mask = mask_files - json_files
    only_img = img_files - json_files

    print(f"\n  JSON 有但 mask 没有: {len(only_json)} 个")
    if only_json:
        for f in sorted(only_json)[:5]:
            print(f"    {f}")
        if len(only_json) > 5:
            print(f"    ... 共 {len(only_json)} 个")

    print(f"\n  mask 有但 JSON 没有: {len(only_mask)} 个")
    if only_mask:
        for f in sorted(only_mask)[:5]:
            print(f"    {f}")
        if len(only_mask) > 5:
            print(f"    ... 共 {len(only_mask)} 个")

    print(f"\n  IMG 有但 JSON 没有: {len(only_img)} 个")
    if only_img:
        for f in sorted(only_img)[:5]:
            print(f"    {f}")
        if len(only_img) > 5:
            print(f"    ... 共 {len(only_img)} 个")

    # 无标注的图像
    annotated_ids = set(ann_by_img.keys())
    unannotated = img_ids - annotated_ids
    print(f"\n  JSON images 无任何 annotation: {len(unannotated)} 个")

    # 无 segmentation 的 annotation
    no_seg = sum(1 for a in anns if not a.get("segmentation"))
    print(f"  annotation 无 segmentation: {no_seg}/{len(anns)} 个")

    # 标注数量分布
    cnt_by_n = {}
    for im in images:
        n = len(ann_by_img.get(im["id"], []))
        cnt_by_n[n] = cnt_by_n.get(n, 0) + 1
    print("\n  每张图的 annotation 数量分布:")
    for n in sorted(cnt_by_n):
        print(f"    {n} 个目标: {cnt_by_n[n]} 张图")

    print("=" * 60)

    if args.stats:
        return

    # ── 确定要处理的图片 ──
    if args.names:
        names = args.names
    elif args.all:
        names = [im["file_name"] for im in images]
    else:
        # 抽样：每种目标数量均匀采样
        random.seed(args.seed)
        buckets = {}
        for im in images:
            n = len(ann_by_img.get(im["id"], []))
            buckets.setdefault(n, []).append(im["file_name"])

        sampled = []
        per_bucket = max(1, args.n // max(len(buckets), 1))
        for n in sorted(buckets):
            pool = buckets[n]
            k = min(per_bucket, len(pool))
            sampled.extend(random.sample(pool, k))

        names = sampled[: args.n]

    print(f"\n将生成 {len(names)} 张预览图...\n")
    success = sum(generate_one(n, coco) for n in names)
    print(f"\n完成！成功 {success}/{len(names)} 张 → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
