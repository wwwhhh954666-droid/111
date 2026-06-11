import argparse
import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def imread_color(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def copy_dataset(base: Path, out: Path):
    for split in ("train", "val"):
        for sub in ("images", "masks"):
            dst = out / split / sub
            dst.mkdir(parents=True, exist_ok=True)
            for src in sorted((base / split / sub).glob("*")):
                if src.is_file():
                    shutil.copy2(src, dst / src.name)


def make_polyline(h, w, rng, branches=False):
    margin = max(24, min(h, w) // 12)
    x = rng.randint(margin, w - margin)
    y = rng.randint(margin, h - margin)
    angle = rng.uniform(0, math.tau)
    length = rng.uniform(0.35, 0.9) * min(h, w)
    steps = rng.randint(5, 11)
    pts = []
    for i in range(steps):
        t = (i / max(1, steps - 1) - 0.5) * length
        bend = math.sin(i * rng.uniform(0.7, 1.4)) * rng.uniform(-22, 22)
        nx = x + math.cos(angle) * t + math.cos(angle + math.pi / 2) * bend
        ny = y + math.sin(angle) * t + math.sin(angle + math.pi / 2) * bend
        nx += rng.uniform(-18, 18)
        ny += rng.uniform(-18, 18)
        pts.append([int(np.clip(nx, 4, w - 5)), int(np.clip(ny, 4, h - 5))])
    lines = [np.asarray(pts, dtype=np.int32)]
    if branches:
        for _ in range(rng.randint(1, 3)):
            base = pts[rng.randrange(1, len(pts) - 1)]
            b_angle = angle + rng.choice([-1, 1]) * rng.uniform(0.55, 1.25)
            b_len = length * rng.uniform(0.12, 0.28)
            b_pts = [base]
            for j in range(1, rng.randint(3, 6)):
                t = j / 4.0 * b_len
                bx = base[0] + math.cos(b_angle) * t + rng.uniform(-10, 10)
                by = base[1] + math.sin(b_angle) * t + rng.uniform(-10, 10)
                b_pts.append([int(np.clip(bx, 4, w - 5)), int(np.clip(by, 4, h - 5))])
            lines.append(np.asarray(b_pts, dtype=np.int32))
    return lines


def add_cracks(image, mask, rng):
    h, w = mask.shape
    count = rng.randint(1, 4)
    for _ in range(count):
        lines = make_polyline(h, w, rng, branches=rng.random() < 0.45)
        width = rng.randint(2, 6)
        edge_width = width + rng.randint(2, 5)
        edge = np.zeros_like(mask)
        core = np.zeros_like(mask)
        for pts in lines:
            cv2.polylines(edge, [pts], False, 255, edge_width, cv2.LINE_AA)
            cv2.polylines(core, [pts], False, 255, width, cv2.LINE_AA)

        dark = rng.uniform(0.25, 0.55)
        halo = rng.uniform(0.82, 0.95)
        edge_bool = edge > 0
        core_bool = core > 0
        image[edge_bool] = np.clip(image[edge_bool].astype(np.float32) * halo, 0, 255).astype(np.uint8)
        image[core_bool] = np.clip(image[core_bool].astype(np.float32) * dark - rng.uniform(4, 18), 0, 255).astype(np.uint8)
        mask[core_bool] = 1
    return image, mask


def irregular_polygon(h, w, rng):
    cx = rng.randint(w // 5, w * 4 // 5)
    cy = rng.randint(h // 5, h * 4 // 5)
    radius = rng.uniform(35, 115)
    n = rng.randint(9, 18)
    pts = []
    for i in range(n):
        a = math.tau * i / n + rng.uniform(-0.18, 0.18)
        r = radius * rng.uniform(0.45, 1.25)
        x = int(np.clip(cx + math.cos(a) * r, 2, w - 3))
        y = int(np.clip(cy + math.sin(a) * r, 2, h - 3))
        pts.append([x, y])
    return np.asarray(pts, dtype=np.int32)


def add_spalls(image, mask, rng):
    h, w = mask.shape
    count = rng.randint(1, 3)
    for _ in range(count):
        poly = irregular_polygon(h, w, rng)
        region = np.zeros_like(mask)
        cv2.fillPoly(region, [poly], 255)
        k = rng.choice([3, 5, 7])
        region = cv2.morphologyEx(region, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        border = cv2.dilate(region, np.ones((7, 7), np.uint8)) - cv2.erode(region, np.ones((5, 5), np.uint8))

        noise = cv2.GaussianBlur(np.random.normal(0, rng.uniform(7, 18), image.shape).astype(np.float32), (0, 0), 2)
        factor = rng.uniform(0.68, 1.18)
        bias = rng.uniform(-38, 28)
        reg = region > 0
        image[reg] = np.clip(image[reg].astype(np.float32) * factor + bias + noise[reg], 0, 255).astype(np.uint8)
        bd = border > 0
        image[bd] = np.clip(image[bd].astype(np.float32) * rng.uniform(0.55, 0.8), 0, 255).astype(np.uint8)
        mask[reg] = 2
    return image, mask


def hist(root: Path, split: str):
    counts = {0: 0, 1: 0, 2: 0}
    for p in (root / split / "masks").glob("*.png"):
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        vals, nums = np.unique(mask, return_counts=True)
        for v, n in zip(vals.tolist(), nums.tolist()):
            counts[int(v)] = counts.get(int(v), 0) + int(n)
    total = sum(counts.values()) or 1
    return {"pixels": counts, "percent": {str(k): round(v * 100 / total, 4) for k, v in counts.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--clean-images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--mode-weights", default="0.45,0.35,0.20")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    base = Path(args.base)
    clean_dir = Path(args.clean_images)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")

    copy_dataset(base, out)
    images = sorted(p for p in clean_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    rng.shuffle(images)
    if not images:
        raise SystemExit(f"No clean images found: {clean_dir}")
    mode_weights = [float(x) for x in args.mode_weights.split(",")]
    if len(mode_weights) != 3:
        raise ValueError("--mode-weights must be crack,spall,mixed")

    img_out = out / "train" / "images"
    mask_out = out / "train" / "masks"
    rows = []
    for i in range(args.count):
        src = images[i % len(images)]
        image = imread_color(src)
        if image is None:
            continue
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mode = rng.choices(["crack", "spall", "mixed"], weights=mode_weights, k=1)[0]
        if mode in ("crack", "mixed"):
            image, mask = add_cracks(image, mask, rng)
        if mode in ("spall", "mixed"):
            image, mask = add_spalls(image, mask, rng)

        stem = f"synthetic_{mode}_{i:04d}_{src.stem}"
        cv2.imwrite(str(img_out / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(mask_out / f"{stem}.png"), mask)
        rows.append({"source": src.name, "mode": mode, "image": f"{stem}.jpg", "mask": f"{stem}.png"})

    summary = {
        "base": str(base),
        "clean_images": str(clean_dir),
        "out": str(out),
        "synthetic_count": len(rows),
        "mode_weights": {"crack": mode_weights[0], "spall": mode_weights[1], "mixed": mode_weights[2]},
        "train_images": len(list((out / "train" / "images").glob("*.jpg"))),
        "val_images": len(list((out / "val" / "images").glob("*.jpg"))),
        "train_hist": hist(out, "train"),
        "val_hist": hist(out, "val"),
    }
    (out / "synthetic_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
