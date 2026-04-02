#!/usr/bin/env python3
"""
将 COCO 标注数据集按 split_record.csv 拆分为 dataset_coco/
├── annotations/
│   ├── test.json
│   ├── train.json
│   └── val.json
└── images/
    ├── test/       (软链接 → stage1_640x640/images/)
    ├── train/      (软链接 → stage1_640x640/images/)
    └── val/        (软链接 → stage1_640x640/images/)
"""

import json
import os
import csv
import shutil
from pathlib import Path
from collections import defaultdict

# ── 路径配置 ──────────────────────────────────────────────
ROOT        = Path("/data1/undergraduate/ultralytics")
SRC_JSON    = ROOT / "coco_annotations" / "instances_train_segmented.json"
SRC_IMG_DIR = ROOT / "stage1_640x640"   / "images"
CSV_PATH    = ROOT / "split_record.csv"
DST_ROOT    = ROOT / "dataset_coco"
DST_IMG     = DST_ROOT / "images"
DST_ANN     = DST_ROOT / "annotations"

# ── 1. 读取 CSV，按 split 分组 ─────────────────────────────
print("[" + "=" * 60)
print("Step 1: 读取 split_record.csv")
print("=" * 60 + "]")

with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

# {split: [file_name, ...]}
split_files = defaultdict(list)
for row in rows:
    fname = row["file_name"]
    split = row["split"]
    split_files[split].append(fname)

for split, names in split_files.items():
    print(f"  [{split}] {len(names)} 张")

# ── 2. 读取 COCO JSON ──────────────────────────────────────
print("\n[" + "=" * 60)
print("Step 2: 读取 COCO JSON")
print("=" * 60 + "]")

with open(SRC_JSON) as f:
    coco = json.load(f)

images     = coco["images"]
annotations = coco["annotations"]
categories  = coco["categories"]

# 建立 file_name → image_entry 映射（用于构建 image_id 集合）
fname_to_img = {im["file_name"]: im for im in images}
print(f"  images: {len(images)}, annotations: {len(annotations)}, categories: {len(categories)}")

# 建立 image_id → annotation列表 映射
img_id_to_anns = defaultdict(list)
for ann in annotations:
    img_id_to_anns[ann["image_id"]].append(ann)

# ── 3. 拆分 COCO JSON ─────────────────────────────────────
print("\n[" + "=" * 60)
print("Step 3: 拆分 COCO JSON → test.json / train.json / val.json")
print("=" * 60 + "]")

# 新 id 计数器（跨 split 独立重排，保证全局唯一且连续）
next_ann_id = 1

for split in ("test", "train", "val"):
    fnames = set(split_files[split])

    # 该 split 包含的 image entries
    split_images = []
    split_img_ids = set()
    for im in images:
        if im["file_name"] in fnames:
            split_images.append(im)
            split_img_ids.add(im["id"])

    # 该 split 包含的 annotation entries（重排 id）
    split_anns = []
    for ann in annotations:
        if ann["image_id"] in split_img_ids:
            ann_copy = dict(ann)
            ann_copy["id"] = next_ann_id
            next_ann_id += 1
            split_anns.append(ann_copy)

    split_coco = {
        "images":     split_images,
        "annotations": split_anns,
        "categories":  categories,
    }

    out_path = DST_ANN / f"{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_coco, f, indent=2, ensure_ascii=False)

    print(f"  {split:5s}: {len(split_images):4d} images, {len(split_anns):4d} annotations → {out_path.name}")

# ── 4. 建立软链接 ──────────────────────────────────────────
print("\n[" + "=" * 60)
print("Step 4: 建立软链接 → images/{test,train,val}/")
print("=" * 60 + "]")

for split in ("test", "train", "val"):
    link_dir = DST_IMG / split
    link_dir.mkdir(parents=True, exist_ok=True)

    for fname in split_files[split]:
        src = SRC_IMG_DIR / fname
        dst = link_dir / fname
        if not dst.exists():
            os.symlink(src, dst)

    actual = sum(1 for _ in link_dir.iterdir())
    print(f"  [{split}] 软链接 {len(split_files[split])} 张 → {actual} 个条目")

# ── 5. 验证 ───────────────────────────────────────────────
print("\n[" + "=" * 60)
print("Step 5: 验证")
print("=" * 60 + "]")

all_ok = True
for split in ("test", "train", "val"):
    link_dir  = DST_IMG / split
    json_path = DST_ANN / f"{split}.json"

    with open(json_path) as f:
        sc = json.load(f)

    json_fnames  = {im["file_name"] for im in sc["images"]}
    csv_fnames   = set(split_files[split])
    missing_json = csv_fnames - json_fnames
    extra_json   = json_fnames - csv_fnames

    link_names   = {p.name for p in link_dir.iterdir()}
    missing_link = csv_fnames - link_names

    # 检查 ann 的 image_id 引用完整性
    valid_img_ids = {im["id"] for im in sc["images"]}
    bad_img_ref   = [a["id"] for a in sc["annotations"] if a["image_id"] not in valid_img_ids]

    # 检查 segmentation 非空
    empty_seg = [a["id"] for a in sc["annotations"] if not a["segmentation"]]

    print(f"\n  [{split}]")
    print(f"    JSON images:      {len(json_fnames)}  |  CSV 记录:  {len(csv_fnames)}")
    print(f"    软链接条目:        {len(link_names)}")
    print(f"    JSON 多余:        {extra_json if extra_json else '无'}")
    print(f"    软链缺失:         {missing_link if missing_link else '无'}")
    print(f"    image_id 引用错误: {bad_img_ref if bad_img_ref else '无'}")
    print(f"    segmentation 为空: {empty_seg if empty_seg else '无'}")

    if missing_json or missing_link or bad_img_ref or empty_seg:
        all_ok = False

print("\n" + "=" * 60)
if all_ok:
    print("✓ 所有验证通过，数据集构建完成！")
else:
    print("✗ 存在上述问题，请检查")
print("=" * 60)

print(f"\n最终目录结构:")
for p in sorted((DST_ROOT).rglob("*")):
    indent = "  " * str(p).count("/")
    if p.is_dir():
        print(f"  {indent}{p.name}/")
    else:
        size_kb = p.stat().st_size / 1024
        print(f"  {indent}{p.name}  ({size_kb:.0f} KB)")
