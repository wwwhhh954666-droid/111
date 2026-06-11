import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from build_synthetic_damage_dataset import IMG_EXTS, add_cracks, add_spalls, imread_color


COLORS = np.array([[0, 0, 0], [255, 40, 40], [40, 180, 255]], dtype=np.uint8)
LABELS = {1: "crack", 2: "spall"}


def list_negative_images(root: Path):
    images = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    return [p for p in images if not p.with_suffix(".json").exists()]


def contour_to_shape(contour, label, epsilon_ratio):
    area = float(cv2.contourArea(contour))
    if area < 20:
        return None
    peri = float(cv2.arcLength(contour, True))
    eps = max(1.0, peri * epsilon_ratio)
    approx = cv2.approxPolyDP(contour, eps, True)
    points = approx.reshape(-1, 2).astype(float).tolist()
    if len(points) < 3:
        return None
    return {
        "label": label,
        "points": points,
        "group_id": None,
        "description": "synthetic damage on text/engraving negative background",
        "shape_type": "polygon",
        "flags": {},
        "mask": None,
        "_area": area,
    }


def mask_to_labelme_shapes(mask, epsilon_ratio=0.0025):
    shapes = []
    for cls, label in LABELS.items():
        binary = (mask == cls).astype(np.uint8) * 255
        if cls == 2:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            shape = contour_to_shape(contour, label, epsilon_ratio)
            if shape:
                shapes.append(shape)
    shapes.sort(key=lambda s: s.pop("_area"), reverse=True)
    return shapes


def save_overlay(image_bgr, mask, out_path):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    color = COLORS[mask]
    damaged = mask > 0
    overlay[damaged] = (overlay[damaged].astype(np.float32) * 0.58 + color[damaged].astype(np.float32) * 0.42).astype(np.uint8)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=r"E:\1\xiuzheng1")
    parser.add_argument("--out", default=r"E:\1\text")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("images", "masks", "overlays"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    negatives = list_negative_images(src)
    if len(negatives) < args.count:
        raise SystemExit(f"Only {len(negatives)} negative images found, need {args.count}: {src}")
    selected = rng.sample(negatives, args.count)

    manifest = []
    for i, image_path in enumerate(selected, 1):
        image = imread_color(image_path)
        if image is None:
            continue
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        image_aug = image.copy()
        image_aug, mask = add_cracks(image_aug, mask, rng)
        image_aug, mask = add_spalls(image_aug, mask, rng)

        stem = f"text_synth_{i:02d}_{image_path.stem}"
        out_image = out / "images" / f"{stem}.jpg"
        out_mask = out / "masks" / f"{stem}_mask.png"
        out_overlay = out / "overlays" / f"{stem}_overlay.jpg"
        out_json = out / "images" / f"{stem}.json"

        cv2.imwrite(str(out_image), image_aug, [cv2.IMWRITE_JPEG_QUALITY, 94])
        cv2.imwrite(str(out_mask), mask)
        save_overlay(image_aug, mask, out_overlay)

        h, w = mask.shape
        data = {
            "version": "5.5.0",
            "flags": {},
            "shapes": mask_to_labelme_shapes(mask),
            "imagePath": out_image.name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
        }
        out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        total = float(mask.size)
        manifest.append(
            {
                "source": str(image_path),
                "image": str(out_image),
                "mask": str(out_mask),
                "json": str(out_json),
                "overlay": str(out_overlay),
                "crack_percent": round(float((mask == 1).sum()) * 100.0 / total, 4),
                "spall_percent": round(float((mask == 2).sum()) * 100.0 / total, 4),
                "shapes": len(data["shapes"]),
            }
        )
        print(
            f"[{i}/{args.count}] {image_path.name} -> {out_image.name} "
            f"crack={manifest[-1]['crack_percent']}% spall={manifest[-1]['spall_percent']}% shapes={manifest[-1]['shapes']}"
        )

    (out / "synthetic_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out={out}")


if __name__ == "__main__":
    main()
