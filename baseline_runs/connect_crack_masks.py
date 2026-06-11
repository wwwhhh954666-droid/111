import argparse
from pathlib import Path

import cv2
import numpy as np


def line_kernel(length, angle_deg, thickness=1):
    length = max(3, int(length))
    size = length if length % 2 == 1 else length + 1
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    angle = np.deg2rad(angle_deg)
    dx = int(round(np.cos(angle) * center))
    dy = int(round(np.sin(angle) * center))
    cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, thickness)
    return kernel


def filter_components(mask, min_area, min_long_side, min_aspect_ratio, max_fill_ratio):
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        long_side = max(w, h)
        short_side = max(1, min(w, h))
        aspect = long_side / short_side
        fill = area / max(1.0, float(w * h))
        if area >= min_area and long_side >= min_long_side and aspect >= min_aspect_ratio and fill <= max_fill_ratio:
            out[labels == idx] = 1
    return out


def connect_mask(mask, close_length, close_thickness, angles, dilate_size, min_area, min_long_side, min_aspect_ratio, max_fill_ratio):
    binary = (mask > 0).astype(np.uint8)
    connected = binary.copy()
    for angle in angles:
        kernel = line_kernel(close_length, angle, close_thickness)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        connected = np.maximum(connected, closed)
    if dilate_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        connected = cv2.dilate(connected, k, iterations=1)
    connected = filter_components(connected, min_area, min_long_side, min_aspect_ratio, max_fill_ratio)
    return (connected * 1).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--close-length", type=int, default=41)
    parser.add_argument("--close-thickness", type=int, default=1)
    parser.add_argument("--angles", default="0,30,60,90,120,150")
    parser.add_argument("--dilate-size", type=int, default=3)
    parser.add_argument("--min-area", type=int, default=900)
    parser.add_argument("--min-long-side", type=int, default=90)
    parser.add_argument("--min-aspect-ratio", type=float, default=1.35)
    parser.add_argument("--max-fill-ratio", type=float, default=0.72)
    args = parser.parse_args()

    mask_dir = Path(args.masks)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    angles = [float(x.strip()) for x in args.angles.split(",") if x.strip()]
    count = 0
    for mask_path in sorted(mask_dir.glob("*_pred.png")):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = (mask == 1).astype(np.uint8)
        connected = connect_mask(
            mask,
            args.close_length,
            args.close_thickness,
            angles,
            args.dilate_size,
            args.min_area,
            args.min_long_side,
            args.min_aspect_ratio,
            args.max_fill_ratio,
        )
        cv2.imwrite(str(out / mask_path.name), connected)
        count += 1
    print(f"converted={count}")
    print(f"out={out}")


if __name__ == "__main__":
    main()
