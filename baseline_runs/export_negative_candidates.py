import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_MAP = {"creak": 1, "crack": 1, "spall": 2}


def find_pairs(root: Path):
    pairs = []
    for jp in sorted(root.rglob("*.json")):
        for ext in IMAGE_EXTS:
            for suffix in (ext, ext.upper()):
                image_path = jp.with_suffix(suffix)
                if image_path.exists():
                    pairs.append((image_path, jp))
                    break
            else:
                continue
            break
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


def line_count(gray):
    edges = cv2.Canny(gray, 60, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45, minLineLength=80, maxLineGap=8)
    if lines is None:
        return 0
    return int(len(lines))


def crop_stats(image_crop, mask_crop, compute_lines=True):
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    black_ratio = float(np.mean(gray < 8))
    dark_ratio = float(np.mean(gray < 25))
    damage_ratio = float(np.count_nonzero(mask_crop) / mask_crop.size)
    std = float(gray.std())
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lines = line_count(gray) if compute_lines else 0
    # Prefer textured hard negatives, but keep disease ratio as the dominant filter.
    score = std + min(lap / 40.0, 40.0) - damage_ratio * 10000.0 - black_ratio * 200.0 - lines * 5.0
    clean_score = -lines * 100.0 - lap / 60.0 - dark_ratio * 80.0 - damage_ratio * 10000.0
    return damage_ratio, black_ratio, dark_ratio, std, lap, lines, score, clean_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=r"D:\mask\shujuji")
    parser.add_argument("--out", default=r"D:\mask\baseline_runs\negative_candidates_for_review")
    parser.add_argument("--patch", type=int, default=512)
    parser.add_argument("--max-per-image", type=int, default=5)
    parser.add_argument("--samples-per-image", type=int, default=180)
    parser.add_argument("--max-damage-ratio", type=float, default=0.005)
    parser.add_argument("--max-black-ratio", type=float, default=0.002)
    parser.add_argument("--max-dark-ratio", type=float, default=0.35)
    parser.add_argument("--require-zero-mask", action="store_true")
    parser.add_argument("--max-long-lines", type=int)
    parser.add_argument("--max-lap-var", type=float)
    parser.add_argument("--prefer-clean", action="store_true")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    src = Path(args.src)
    out = Path(args.out)
    img_out = out / "images"
    mask_out = out / "masks_reference"
    if args.clean and out.exists():
        shutil.rmtree(out)
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    rows = []
    pairs = find_pairs(src)
    for image_path, json_path in pairs:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        if h < args.patch or w < args.patch:
            continue
        mask = labelme_to_mask(json_path, h, w)
        candidates = []

        # Mix random candidates with a coarse grid so broad clean areas are not missed.
        positions = []
        for _ in range(args.samples_per_image):
            x = rng.randrange(0, w - args.patch + 1)
            y = rng.randrange(0, h - args.patch + 1)
            positions.append((x, y))
        stride = args.patch
        for y in range(0, h - args.patch + 1, stride):
            for x in range(0, w - args.patch + 1, stride):
                positions.append((x, y))

        seen = set()
        for x, y in positions:
            key = (x // 64, y // 64)
            if key in seen:
                continue
            seen.add(key)
            crop = image[y : y + args.patch, x : x + args.patch]
            mask_crop = mask[y : y + args.patch, x : x + args.patch]
            quick_damage_ratio = float(np.count_nonzero(mask_crop) / mask_crop.size)
            if args.require_zero_mask and quick_damage_ratio > 0:
                continue
            if quick_damage_ratio > args.max_damage_ratio:
                continue
            damage_ratio, black_ratio, dark_ratio, std, lap, lines, score, clean_score = crop_stats(
                crop, mask_crop, compute_lines=args.max_long_lines is not None
            )
            if args.require_zero_mask and damage_ratio > 0:
                continue
            if damage_ratio > args.max_damage_ratio:
                continue
            if black_ratio > args.max_black_ratio or dark_ratio > args.max_dark_ratio:
                continue
            if args.max_long_lines is not None and lines > args.max_long_lines:
                continue
            if args.max_lap_var is not None and lap > args.max_lap_var:
                continue
            candidates.append((clean_score if args.prefer_clean else score, x, y, damage_ratio, black_ratio, dark_ratio, std, lap, lines))

        candidates.sort(reverse=True)
        for rank, (score, x, y, damage_ratio, black_ratio, dark_ratio, std, lap, lines) in enumerate(candidates[: args.max_per_image]):
            stem = f"{image_path.stem}_neg_{rank:02d}_x{x}_y{y}"
            crop = image[y : y + args.patch, x : x + args.patch]
            mask_crop = mask[y : y + args.patch, x : x + args.patch]
            out_img = img_out / f"{stem}.jpg"
            out_mask = mask_out / f"{stem}.png"
            cv2.imwrite(str(out_img), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
            cv2.imwrite(str(out_mask), mask_crop)
            rows.append(
                {
                    "file": out_img.name,
                    "source_image": str(image_path),
                    "source_json": str(json_path),
                    "x": x,
                    "y": y,
                    "patch": args.patch,
                    "damage_ratio": round(damage_ratio, 6),
                    "black_ratio": round(black_ratio, 6),
                    "dark_ratio": round(dark_ratio, 6),
                    "gray_std": round(std, 3),
                    "lap_var": round(lap, 3),
                    "long_lines": lines,
                    "score": round(score, 3),
                }
            )

    with (out / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "source_image",
                "source_json",
                "x",
                "y",
                "patch",
                "damage_ratio",
                "black_ratio",
                "dark_ratio",
                "gray_std",
                "lap_var",
                "long_lines",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"exported={len(rows)}")
    print(f"images={img_out}")
    print(f"metadata={out / 'metadata.csv'}")


if __name__ == "__main__":
    main()
