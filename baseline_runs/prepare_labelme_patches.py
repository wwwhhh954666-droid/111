import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_MAP = {"creak": 1, "crack": 1, "spall": 2}


def find_pairs(root: Path):
    jsons = sorted(root.rglob("*.json"))
    pairs = []
    for jp in jsons:
        image_path = None
        for ext in IMAGE_EXTS:
            candidate = jp.with_suffix(ext)
            if candidate.exists():
                image_path = candidate
                break
            candidate = jp.with_suffix(ext.upper())
            if candidate.exists():
                image_path = candidate
                break
        if image_path:
            pairs.append((image_path, jp))
    return pairs


def labelme_to_mask(json_path: Path, h: int, w: int):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in data.get("shapes", []):
        cls = LABEL_MAP.get(str(shape.get("label", "")).strip().lower())
        if cls is None:
            continue
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if len(points) < 3:
            continue
        points[:, 0] = np.clip(points[:, 0], 0, w - 1)
        points[:, 1] = np.clip(points[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [points.astype(np.int32)], int(cls))
    return mask


def crop_with_padding(arr, x, y, size, pad_value=0):
    h, w = arr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + size), min(h, y + size)
    crop = arr[y0:y1, x0:x1]
    top, left = y0 - y, x0 - x
    bottom, right = y + size - y1, x + size - x1
    if top or bottom or left or right:
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)
    return crop


def sample_class_centers(mask, cls, max_patches, rng):
    ys, xs = np.where(mask == cls)
    if len(xs) == 0:
        return []
    count = min(max_patches, max(1, len(xs) // 18000 + 1))
    idxs = rng.sample(range(len(xs)), min(count, len(xs)))
    return [(int(xs[i]), int(ys[i])) for i in idxs]


def sample_background_centers(mask, max_patches, rng, patch_size=None, max_damage_ratio=None):
    h, w = mask.shape
    centers = []
    for _ in range(max_patches * 20):
        if len(centers) >= max_patches:
            break
        x, y = rng.randrange(w), rng.randrange(h)
        if mask[y, x] != 0:
            continue
        if patch_size and max_damage_ratio is not None:
            patch = crop_with_padding(mask, x - patch_size // 2, y - patch_size // 2, patch_size, 0)
            if float(np.count_nonzero(patch)) / float(patch.size) > max_damage_ratio:
                continue
            centers.append((x, y))
        else:
            centers.append((x, y))
    while len(centers) < max_patches:
        centers.append((rng.randrange(w), rng.randrange(h)))
    return centers


def write_patch(image, mask, out_img, out_mask, x, y, size):
    img_crop = crop_with_padding(image, x, y, size, (0, 0, 0))
    mask_crop = crop_with_padding(mask, x, y, size, 0)
    cv2.imwrite(str(out_img), img_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out_mask), mask_crop)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=r"D:\mask\shujuji")
    parser.add_argument("--out", default=r"D:\mask\baseline_runs\patch_dataset")
    parser.add_argument("--patch", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--positive-per-image", type=int, default=10)
    parser.add_argument("--negative-per-image", type=int, default=3)
    parser.add_argument("--crack-per-image", type=int)
    parser.add_argument("--spall-per-image", type=int)
    parser.add_argument("--background-per-image", type=int)
    parser.add_argument("--max-background-damage-ratio", type=float)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    src = Path(args.src)
    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)

    pairs = find_pairs(src)
    rng.shuffle(pairs)
    val_count = max(1, round(len(pairs) * args.val_ratio))
    splits = {"val": pairs[:val_count], "train": pairs[val_count:]}

    summary = {"src": str(src), "out": str(out), "pairs": len(pairs), "splits": {}}
    for split, items in splits.items():
        img_dir = out / split / "images"
        mask_dir = out / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        patch_count = 0
        class_pixels = {0: 0, 1: 0, 2: 0}

        for image_path, json_path in items:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            h, w = image.shape[:2]
            mask = labelme_to_mask(json_path, h, w)
            if args.crack_per_image is None and args.spall_per_image is None and args.background_per_image is None:
                centers = []
                centers.extend(sample_class_centers(mask, 1, args.positive_per_image, rng))
                centers.extend(sample_class_centers(mask, 2, args.positive_per_image, rng))
                for _ in range(args.negative_per_image):
                    centers.append((rng.randrange(w), rng.randrange(h)))
            else:
                centers = []
                centers.extend(sample_class_centers(mask, 1, args.crack_per_image or 0, rng))
                centers.extend(sample_class_centers(mask, 2, args.spall_per_image or 0, rng))
                centers.extend(
                    sample_background_centers(
                        mask,
                        args.background_per_image or 0,
                        rng,
                        args.patch,
                        args.max_background_damage_ratio,
                    )
                )

            for i, (cx, cy) in enumerate(centers):
                jitter = args.patch // 3
                x = cx - args.patch // 2 + rng.randint(-jitter, jitter)
                y = cy - args.patch // 2 + rng.randint(-jitter, jitter)
                stem = f"{image_path.stem}_{i:02d}"
                out_img = img_dir / f"{stem}.jpg"
                out_mask = mask_dir / f"{stem}.png"
                write_patch(image, mask, out_img, out_mask, x, y, args.patch)
                patch_mask = cv2.imread(str(out_mask), cv2.IMREAD_GRAYSCALE)
                vals, counts = np.unique(patch_mask, return_counts=True)
                for val, count in zip(vals, counts):
                    class_pixels[int(val)] += int(count)
                patch_count += 1

        total = sum(class_pixels.values()) or 1
        summary["splits"][split] = {
            "images": len(items),
            "patches": patch_count,
            "class_pixels": class_pixels,
            "class_percent": {str(k): round(v * 100 / total, 4) for k, v in class_pixels.items()},
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
