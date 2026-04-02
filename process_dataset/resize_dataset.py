"""
将 stage1_cleaned 中的图片和掩码 resize 到 640x640
- 保持文件名不变
- 非正方形图片用黑色(0)填充图片，忽略标签(128)填充掩码
- 输出到 stage1_640x640 文件夹.
"""

from pathlib import Path

from PIL import Image

# 配置
INPUT_DIR = Path("/data1/undergraduate/ultralytics/stage1_cleaned")
OUTPUT_DIR = Path("/data1/undergraduate/ultralytics/stage1_640x640")
TARGET_SIZE = (640, 640)


def resize_with_padding(input_path: Path, output_path: Path, target_size: tuple):
    """将图片 resize 并用黑色填充到目标尺寸."""
    img = Image.open(input_path)

    # 计算缩放比例（保持宽高比）
    orig_w, orig_h = img.size
    target_w, target_h = target_size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # 缩放图片
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # 创建黑色画布
    canvas = Image.new(img.mode, target_size, 0)

    # 计算居中偏移
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2

    # 将缩放后的图片粘贴到画布中央
    canvas.paste(img_resized, (offset_x, offset_y))

    # 保存
    canvas.save(output_path, optimize=False)


def main():
    # 创建输出目录
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "masks").mkdir(parents=True, exist_ok=True)

    img_dir = INPUT_DIR / "images"
    mask_dir = INPUT_DIR / "masks"

    # 获取所有图片文件
    img_files = {}
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for p in img_dir.glob(ext):
            img_files[p.stem] = p

    mask_files = {}
    for p in mask_dir.glob("*.png"):
        mask_files[p.stem] = p

    # 检查对应关系
    img_keys = set(img_files.keys())
    mask_keys = set(mask_files.keys())

    common = img_keys & mask_keys
    only_img = img_keys - mask_keys
    only_mask = mask_keys - img_keys

    print(f"图片总数: {len(img_files)}")
    print(f"掩码总数: {len(mask_files)}")
    print(f"一一对应: {len(common)}")
    print(f"只有图片: {len(only_img)}")
    print(f"只有掩码: {len(only_mask)}")

    if only_img:
        print("\n警告: 以下图片没有对应掩码，将跳过:")
        for k in sorted(only_img)[:5]:
            print(f"  - {k}")
        if len(only_img) > 5:
            print(f"  ... 还有 {len(only_img) - 5} 个")

    if only_mask:
        print("\n警告: 以下掩码没有对应图片，将跳过:")
        for k in sorted(only_mask)[:5]:
            print(f"  - {k}")
        if len(only_mask) > 5:
            print(f"  ... 还有 {len(only_mask) - 5} 个")

    # 处理每对图片和掩码
    print(f"\n开始处理 {len(common)} 对图片和掩码...")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"目标尺寸: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print("-" * 50)

    success = 0
    failed = []

    for i, stem in enumerate(sorted(common), 1):
        img_path = img_files[stem]
        mask_path = mask_files[stem]

        out_img = OUTPUT_DIR / "images" / img_path.name
        out_mask = OUTPUT_DIR / "masks" / mask_path.name

        try:
            # 处理图片
            resize_with_padding(img_path, out_img, TARGET_SIZE)

            # 处理掩码（使用最近邻插值，保持掩码值）
            # 填充区用 128 标记为忽略区域
            mask = Image.open(mask_path)
            orig_w, orig_h = mask.size
            target_w, target_h = TARGET_SIZE

            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            mask_resized = mask.resize((new_w, new_h), Image.NEAREST)
            canvas = Image.new(mask.mode, TARGET_SIZE, 128)  # 填充区=忽略标签
            offset_x = (target_w - new_w) // 2
            offset_y = (target_h - new_h) // 2
            canvas.paste(mask_resized, (offset_x, offset_y))
            canvas.save(out_mask, optimize=False)

            success += 1

            if i % 500 == 0 or i == len(common):
                print(f"进度: {i}/{len(common)} ({i / len(common) * 100:.1f}%)")

        except Exception as e:
            failed.append((stem, str(e)))
            print(f"错误: {stem} - {e}")

    print("-" * 50)
    print("处理完成!")
    print(f"成功: {success}/{len(common)}")
    if failed:
        print(f"失败: {len(failed)}")
        for stem, err in failed[:10]:
            print(f"  - {stem}: {err}")


if __name__ == "__main__":
    main()
