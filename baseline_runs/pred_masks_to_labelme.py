import argparse
import base64
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_LABELS = {1: "creak", 2: "spall"}


def imread_color(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imread_gray(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def list_images(src: Path):
    return sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def encode_image(path: Path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def contour_to_shape(contour, label, epsilon_ratio):
    area = float(cv2.contourArea(contour))
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
        "description": "",
        "shape_type": "polygon",
        "flags": {},
        "mask": None,
        "_area": area,
    }


def mask_to_shapes(
    mask,
    min_area,
    epsilon_ratio,
    crack_min_area,
    crack_min_long_side,
    crack_min_arc_length,
    crack_min_aspect_ratio,
    crack_max_fill_ratio,
    spall_min_area,
    spall_min_long_side,
):
    shapes = []
    for cls, label in CLASS_LABELS.items():
        binary = (mask == cls).astype(np.uint8) * 255
        if cls == 2:
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            arc_length = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            long_side = max(w, h)
            short_side = max(1, min(w, h))
            aspect_ratio = long_side / short_side
            fill_ratio = area / max(1.0, float(w * h))
            if area < min_area:
                continue
            if cls == 1 and (
                area < crack_min_area
                or long_side < crack_min_long_side
                or arc_length < crack_min_arc_length
                or aspect_ratio < crack_min_aspect_ratio
                or fill_ratio > crack_max_fill_ratio
            ):
                continue
            if cls == 2 and (area < spall_min_area or long_side < spall_min_long_side):
                continue
            shape = contour_to_shape(contour, label, epsilon_ratio)
            if shape:
                shapes.append(shape)
    shapes.sort(key=lambda s: s.pop("_area"), reverse=True)
    return shapes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-area", type=float, default=180.0)
    parser.add_argument("--crack-min-area", type=float, default=180.0)
    parser.add_argument("--crack-min-long-side", type=float, default=0.0)
    parser.add_argument("--crack-min-arc-length", type=float, default=0.0)
    parser.add_argument("--crack-min-aspect-ratio", type=float, default=0.0)
    parser.add_argument("--crack-max-fill-ratio", type=float, default=1.0)
    parser.add_argument("--spall-min-area", type=float, default=180.0)
    parser.add_argument("--spall-min-long-side", type=float, default=0.0)
    parser.add_argument("--classes", default="creak,spall")
    parser.add_argument("--epsilon-ratio", type=float, default=0.003)
    parser.add_argument("--embed-image", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.images)
    mask_dir = Path(args.masks)
    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    converted = 0
    for image_path in list_images(image_dir):
        mask_path = mask_dir / f"{image_path.stem}_pred.png"
        if not mask_path.exists():
            mask_path = mask_dir / f"{image_path.stem}.png"
        mask = imread_gray(mask_path)
        image = imread_color(image_path)
        if image is None or mask is None:
            continue
        h, w = image.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        out_image = out / image_path.name
        shutil.copy2(image_path, out_image)
        keep_labels = {x.strip() for x in args.classes.split(",") if x.strip()}
        shapes = mask_to_shapes(
            mask,
            args.min_area,
            args.epsilon_ratio,
            args.crack_min_area,
            args.crack_min_long_side,
            args.crack_min_arc_length,
            args.crack_min_aspect_ratio,
            args.crack_max_fill_ratio,
            args.spall_min_area,
            args.spall_min_long_side,
        )
        shapes = [shape for shape in shapes if shape["label"] in keep_labels]

        data = {
            "version": "5.5.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_path.name,
            "imageData": encode_image(out_image) if args.embed_image else None,
            "imageHeight": h,
            "imageWidth": w,
        }
        (out / f"{image_path.stem}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        converted += 1
        print(f"{image_path.name}: shapes={len(data['shapes'])}")

    print(f"converted={converted}")
    print(f"out={out}")


if __name__ == "__main__":
    main()
