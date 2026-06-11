import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from prepare_labelme_patches import (
    IMAGE_EXTS,
    crop_with_padding,
    labelme_to_mask,
    sample_background_centers,
    sample_class_centers,
    write_patch,
)


def list_images(root: Path):
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def copy_dataset(src: Path, dst: Path, clean: bool):
    if clean and dst.exists():
        shutil.rmtree(dst)
    for split in ("train", "val"):
        for sub in ("images", "masks"):
            out_dir = dst / split / sub
            out_dir.mkdir(parents=True, exist_ok=True)
            for p in sorted((src / split / sub).glob("*")):
                if p.is_file():
                    shutil.copy2(p, out_dir / p.name)


def split_items(items, val_ratio, rng):
    items = list(items)
    rng.shuffle(items)
    val_count = round(len(items) * val_ratio)
    return {"val": items[:val_count], "train": items[val_count:]}


def class_hist(root: Path, split: str):
    counts = {0: 0, 1: 0, 2: 0}
    for p in (root / split / "masks").glob("*.png"):
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        vals, nums = np.unique(mask, return_counts=True)
        for v, n in zip(vals.tolist(), nums.tolist()):
            counts[int(v)] = counts.get(int(v), 0) + int(n)
    total = sum(counts.values()) or 1
    return {
        "pixels": counts,
        "percent": {str(k): round(v * 100.0 / total, 4) for k, v in counts.items()},
    }


def write_review_patches(image_path, mask, out_root, split, args, rng, prefix):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return 0
    h, w = image.shape[:2]
    centers = []
    centers.extend(sample_class_centers(mask, 1, args.crack_per_image, rng))
    centers.extend(sample_class_centers(mask, 2, args.spall_per_image, rng))
    centers.extend(
        sample_background_centers(
            mask,
            args.background_per_positive_image,
            rng,
            args.patch,
            args.max_background_damage_ratio,
        )
    )
    if not centers:
        centers.extend(sample_background_centers(mask, args.negative_per_image, rng, args.patch, 0.0))

    img_dir = out_root / split / "images"
    mask_dir = out_root / split / "masks"
    patch_count = 0
    for i, (cx, cy) in enumerate(centers):
        jitter = args.patch // 3
        x = cx - args.patch // 2 + rng.randint(-jitter, jitter)
        y = cy - args.patch // 2 + rng.randint(-jitter, jitter)
        stem = f"{prefix}_{image_path.stem}_{i:02d}"
        write_patch(
            image,
            mask,
            img_dir / f"{stem}.jpg",
            mask_dir / f"{stem}.png",
            x,
            y,
            args.patch,
        )
        patch_count += 1
    return patch_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--src", default=r"E:\1\xiuzheng1")
    parser.add_argument("--out", required=True)
    parser.add_argument("--patch", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--crack-per-image", type=int, default=8)
    parser.add_argument("--spall-per-image", type=int, default=8)
    parser.add_argument("--background-per-positive-image", type=int, default=4)
    parser.add_argument("--negative-per-image", type=int, default=4)
    parser.add_argument("--max-background-damage-ratio", type=float, default=0.01)
    parser.add_argument("--max-negative-images", type=int, default=91)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    src = Path(args.src)
    out = Path(args.out)
    copy_dataset(Path(args.base), out, args.clean)

    images = list_images(src)
    json_by_stem = {p.stem: p for p in src.glob("*.json")}
    positive = [(p, json_by_stem[p.stem]) for p in images if p.stem in json_by_stem]
    negative = [p for p in images if p.stem not in json_by_stem]
    rng.shuffle(negative)
    negative = negative[: args.max_negative_images]

    pos_splits = split_items(positive, args.val_ratio, rng)
    neg_splits = split_items(negative, args.val_ratio, rng)

    added = {"train": {"positive_patches": 0, "negative_patches": 0}, "val": {"positive_patches": 0, "negative_patches": 0}}
    for split, items in pos_splits.items():
        for image_path, json_path in items:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            h, w = image.shape[:2]
            mask = labelme_to_mask(json_path, h, w)
            added[split]["positive_patches"] += write_review_patches(image_path, mask, out, split, args, rng, "xz1_pos")

    for split, items in neg_splits.items():
        for image_path in items:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            added[split]["negative_patches"] += write_review_patches(image_path, mask, out, split, args, rng, "xz1_hn")

    summary = {
        "base": str(Path(args.base)),
        "src": str(src),
        "out": str(out),
        "images": len(images),
        "positive_json": len(positive),
        "negative_images_used": len(negative),
        "added": added,
        "train_images": len(list((out / "train" / "images").glob("*.jpg"))),
        "val_images": len(list((out / "val" / "images").glob("*.jpg"))),
        "train_hist": class_hist(out, "train"),
        "val_hist": class_hist(out, "val"),
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
