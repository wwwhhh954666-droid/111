import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from detect_text_regions import TextRegionConfig, detect_text_regions, load_text_mask
from predict_unlabeled_images import COLORS, list_images, write_contact_sheet


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


def build_text_like_mask(image_bgr, args):
    config = TextRegionConfig(
        edge_window=args.edge_window,
        high_percentile=args.high_percentile,
        min_density=args.text_min_density,
        close_kernel=args.text_close_kernel,
        dilate_kernel=args.text_dilate_kernel,
        min_region_area=args.text_min_region_area,
        max_region_area_ratio=args.text_max_region_area_ratio,
        pad=args.text_pad,
    )
    return detect_text_regions(image_bgr, config).mask


def get_text_like_mask(image_path, image_bgr, args):
    if args.text_masks:
        text_like = load_text_mask(Path(args.text_masks), image_path.stem, args.text_mask_suffix)
        if text_like is not None:
            if text_like.shape[:2] != image_bgr.shape[:2]:
                text_like = cv2.resize(text_like.astype(np.uint8), (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            return text_like
        print(f"missing precomputed text mask, detecting on the fly: {image_path.name}")
    return build_text_like_mask(image_bgr, args)


def component_shape_score(component):
    ys, xs = np.where(component)
    if len(xs) == 0:
        return {"area": 0, "aspect": 1.0, "fill": 1.0, "long_side": 0}
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w, h = x1 - x0 + 1, y1 - y0 + 1
    area = int(len(xs))
    long_side = max(w, h)
    aspect = long_side / max(1, min(w, h))
    fill = area / max(1.0, float(w * h))
    return {"area": area, "aspect": aspect, "fill": fill, "long_side": long_side}


def suppress_cracks(mask, text_like, args):
    crack = (mask == 1).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(crack, 8)
    out = mask.copy()
    removed = 0
    kept = 0
    for i in range(1, num):
        comp = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < args.min_component_area:
            continue
        shape = component_shape_score(comp)
        text_overlap = float(np.count_nonzero(comp & text_like)) / max(1.0, float(area))

        # Remove likely text-edge false positives. Keep long cracks that escape text-like areas.
        likely_text = (
            text_overlap >= args.text_overlap
            and shape["long_side"] <= args.max_text_component_long_side
            and (shape["aspect"] <= args.keep_aspect or shape["fill"] >= args.max_crack_fill)
        )
        very_texty_short = text_overlap >= args.strong_text_overlap and shape["long_side"] <= args.strong_max_long_side

        if likely_text or very_texty_short:
            out[comp] = 0
            removed += 1
        else:
            kept += 1
    return out, removed, kept


def save_outputs(image_path, image_bgr, mask, text_like, out_dir):
    for name in ("masks", "overlays", "panels", "text_like"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    cv2.imwrite(str(out_dir / "masks" / f"{stem}_pred.png"), mask)
    cv2.imwrite(str(out_dir / "text_like" / f"{stem}_text_like.png"), text_like.astype(np.uint8) * 255)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    color = COLORS[mask]
    damaged = mask > 0
    overlay[damaged] = (overlay[damaged].astype(np.float32) * 0.58 + color[damaged].astype(np.float32) * 0.42).astype(np.uint8)
    cv2.imwrite(str(out_dir / "overlays" / f"{stem}_overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    text_rgb = rgb.copy()
    text_rgb[text_like] = (text_rgb[text_like].astype(np.float32) * 0.55 + np.array([255, 220, 20]) * 0.45).astype(np.uint8)
    small_w = 760
    scale = small_w / rgb.shape[1]
    small_h = max(1, int(rgb.shape[0] * scale))
    parts = []
    for title, arr in (("image", rgb), ("text-like", text_rgb), ("suppressed pred", overlay)):
        small = cv2.resize(arr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(small, (0, 0), (small_w, 30), (255, 255, 255), -1)
        cv2.putText(small, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        parts.append(small)
    panel = np.concatenate(parts, axis=1)
    cv2.imwrite(str(out_dir / "panels" / f"{stem}_panel.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--text-overlap", type=float, default=0.62)
    parser.add_argument("--strong-text-overlap", type=float, default=0.82)
    parser.add_argument("--keep-aspect", type=float, default=5.5)
    parser.add_argument("--max-crack-fill", type=float, default=0.23)
    parser.add_argument("--max-text-component-long-side", type=int, default=760)
    parser.add_argument("--strong-max-long-side", type=int, default=1100)
    parser.add_argument("--min-component-area", type=int, default=20)
    parser.add_argument("--edge-window", type=int, default=41)
    parser.add_argument("--high-percentile", type=float, default=96)
    parser.add_argument("--text-masks", default="", help="Optional masks directory produced by detect_text_regions.py.")
    parser.add_argument("--text-mask-suffix", default="_text_mask.png")
    parser.add_argument("--text-min-density", type=float, default=0.08)
    parser.add_argument("--text-close-kernel", type=int, default=7)
    parser.add_argument("--text-dilate-kernel", type=int, default=3)
    parser.add_argument("--text-min-region-area", type=int, default=5000)
    parser.add_argument("--text-max-region-area-ratio", type=float, default=0.40)
    parser.add_argument("--text-pad", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.images)
    mask_root = Path(args.masks)
    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    images = [image_dir] if image_dir.is_file() else list_images(image_dir)
    for idx, image_path in enumerate(images, 1):
        image = imread_color(image_path)
        mask_path = mask_root if mask_root.is_file() else mask_root / f"{image_path.stem}_pred.png"
        if image is None or not mask_path.exists():
            continue
        mask = imread_gray(mask_path)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        text_like = get_text_like_mask(image_path, image, args)
        before_crack = int((mask == 1).sum())
        after, removed, kept = suppress_cracks(mask, text_like, args)
        after_crack = int((after == 1).sum())
        save_outputs(image_path, image, after, text_like, out)
        total = float(after.size)
        row = {
            "image": image_path.name,
            "before_crack_percent": round(before_crack * 100.0 / total, 4),
            "after_crack_percent": round(after_crack * 100.0 / total, 4),
            "removed_crack_pixels": before_crack - after_crack,
            "spall_percent": round(float((after == 2).sum()) * 100.0 / total, 4),
            "text_like_percent": round(float(text_like.sum()) * 100.0 / total, 4),
            "removed_components": removed,
            "kept_components": kept,
        }
        rows.append(row)
        print(f"[{idx}/{len(images)}] {image_path.name} crack {row['before_crack_percent']} -> {row['after_crack_percent']} removed_components={removed}")

    if rows:
        with (out / "suppression_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    write_contact_sheet(sorted((out / "panels").glob("*_panel.jpg")), out / "contact_sheet.jpg")
    print(f"out={out}")


if __name__ == "__main__":
    main()
