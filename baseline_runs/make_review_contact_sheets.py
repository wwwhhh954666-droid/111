import argparse
import math
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def imread_color(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def draw_label(img, text):
    out = img.copy()
    pad = 22
    cv2.rectangle(out, (0, 0), (out.shape[1], pad), (255, 255, 255), -1)
    cv2.putText(out, text[:42], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def make_sheet(paths, out_path, thumb=180, cols=5):
    cells = []
    for path in paths:
        img = imread_color(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb / max(h, w)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((thumb + 22, thumb, 3), 245, dtype=np.uint8)
        x = (thumb - nw) // 2
        y = 22 + (thumb - nh) // 2
        canvas[y : y + nh, x : x + nw] = small
        cells.append(draw_label(canvas, path.name))
    if not cells:
        return
    rows = []
    total_rows = math.ceil(len(cells) / cols)
    blank = np.full_like(cells[0], 245)
    for r in range(total_rows):
        row = cells[r * cols : (r + 1) * cols]
        while len(row) < cols:
            row.append(blank)
        rows.append(np.concatenate(row, axis=1))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-sheet", type=int, default=50)
    parser.add_argument("--thumb", type=int, default=180)
    parser.add_argument("--cols", type=int, default=5)
    args = parser.parse_args()

    image_dir = Path(args.images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    for start in range(0, len(paths), args.per_sheet):
        batch = paths[start : start + args.per_sheet]
        index = start // args.per_sheet + 1
        make_sheet(batch, out / f"review_sheet_{index:03d}.jpg", args.thumb, args.cols)
    print(f"images={len(paths)}")
    print(f"sheets={math.ceil(len(paths) / args.per_sheet)}")
    print(f"out={out}")


if __name__ == "__main__":
    main()
