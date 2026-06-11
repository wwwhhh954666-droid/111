import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(src: Path):
    return sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def crop_with_padding(arr, cx, cy, size, pad_value=0):
    h, w = arr.shape[:2]
    x = int(round(cx - size / 2))
    y = int(round(cy - size / 2))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + size), min(h, y + size)
    crop = arr[y0:y1, x0:x1]
    top, left = y0 - y, x0 - x
    bottom, right = y + size - y1, x + size - x1
    if top or bottom or left or right:
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)
    return crop, x, y


def component_rows(image_path, image, mask, max_per_image, patch_size):
    binary = (mask > 0).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    rows = []
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if area < 150:
            continue
        long_side = max(w, h)
        short_side = max(1, min(w, h))
        aspect = long_side / short_side
        fill = area / max(1.0, float(w * h))
        # Smaller, shorter, and blob-like detections are useful hard-negative review candidates.
        suspicion = (1.0 / max(aspect, 0.1)) * 100.0 + fill * 80.0 - min(long_side, 180) / 10.0
        cx, cy = centroids[idx]
        rows.append({
            "source": image_path.name,
            "cx": float(cx),
            "cy": float(cy),
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(area),
            "aspect": round(float(aspect), 3),
            "fill": round(float(fill), 3),
            "suspicion": round(float(suspicion), 3),
            "_score": suspicion,
        })
    rows.sort(key=lambda r: r["_score"], reverse=True)
    return rows[:max_per_image]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--patch", type=int, default=512)
    parser.add_argument("--max-per-image", type=int, default=8)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.images)
    mask_dir = Path(args.masks)
    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    img_out = out / "images"
    overlay_out = out / "overlays"
    img_out.mkdir(parents=True, exist_ok=True)
    overlay_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in list_images(image_dir):
        mask_path = mask_dir / f"{image_path.stem}_pred.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        for rank, row in enumerate(component_rows(image_path, image, mask, args.max_per_image, args.patch)):
            crop, crop_x, crop_y = crop_with_padding(image, row["cx"], row["cy"], args.patch, (0, 0, 0))
            mask_crop, _, _ = crop_with_padding(mask, row["cx"], row["cy"], args.patch, 0)
            overlay = crop.copy()
            overlay[mask_crop > 0] = (overlay[mask_crop > 0] * 0.55 + np.array([40, 40, 255]) * 0.45).astype(np.uint8)
            stem = f"{image_path.stem}_rank{rank:02d}_x{crop_x}_y{crop_y}"
            cv2.imwrite(str(img_out / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
            cv2.imwrite(str(overlay_out / f"{stem}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 94])
            row.update({"file": f"{stem}.jpg", "crop_x": crop_x, "crop_y": crop_y})
            row.pop("_score", None)
            rows.append(row)

    if rows:
        with (out / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
            keys = ["file", "source", "crop_x", "crop_y", "cx", "cy", "x", "y", "w", "h", "area", "aspect", "fill", "suspicion"]
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(f"exported={len(rows)}")
    print(f"out={out}")


if __name__ == "__main__":
    main()
