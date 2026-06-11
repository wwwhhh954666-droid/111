import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class TextRegionConfig:
    edge_window: int = 41
    high_percentile: float = 96.0
    min_density: float = 0.08
    close_kernel: int = 7
    dilate_kernel: int = 3
    min_region_area: int = 5000
    max_region_area_ratio: float = 0.40
    pad: int = 0


@dataclass
class TextDetectionResult:
    mask: np.ndarray
    density: np.ndarray
    raw_mask: np.ndarray
    regions: list


def odd_kernel(value, minimum=1):
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def imread_color(path: Path):
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imread_gray(path: Path):
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def imwrite(path: Path, image, params=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image, params or [])
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def list_images(src: Path, recursive=False):
    if src.is_file():
        return [src] if src.suffix.lower() in IMAGE_EXTS else []
    iterator = src.rglob("*") if recursive else src.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def build_text_like_mask(
    image_bgr,
    edge_window=41,
    high_percentile=82,
    min_density=0.035,
    close_kernel=13,
    dilate_kernel=9,
):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (0, 0), 1.2)

    med = float(np.median(blur))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med + 25))
    edges = cv2.Canny(blur, lo, hi)

    sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobel_x, sobel_y)
    mag_thr = np.percentile(mag, 78)
    strong = (mag >= mag_thr).astype(np.uint8) * 255
    edge = cv2.bitwise_or(edges, strong)

    edge_window = odd_kernel(edge_window, 3)
    density = cv2.blur((edge > 0).astype(np.float32), (edge_window, edge_window))
    thr = max(float(min_density), float(np.percentile(density, high_percentile)))
    dense = density >= thr

    close_kernel = odd_kernel(close_kernel, 1)
    dilate_kernel = odd_kernel(dilate_kernel, 1)
    text_like = cv2.morphologyEx(dense.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((close_kernel, close_kernel), np.uint8))
    text_like = cv2.dilate(text_like, np.ones((dilate_kernel, dilate_kernel), np.uint8), iterations=1)
    return text_like.astype(bool), density


def _clip_box(x, y, w, h, image_w, image_h, pad):
    x0 = max(0, int(x) - pad)
    y0 = max(0, int(y) - pad)
    x1 = min(image_w, int(x + w) + pad)
    y1 = min(image_h, int(y + h) + pad)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def extract_regions(raw_mask, density, config: TextRegionConfig):
    h, w = raw_mask.shape[:2]
    num, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask.astype(np.uint8), 8)
    max_area = max(config.min_region_area, int(h * w * float(config.max_region_area_ratio)))
    mask = np.zeros_like(raw_mask, dtype=bool)
    regions = []

    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < config.min_region_area or area > max_area:
            continue

        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        x, y, bw, bh = _clip_box(x, y, bw, bh, w, h, int(config.pad))

        comp = labels[y : y + bh, x : x + bw] == idx
        if not np.any(comp):
            continue
        region_density = float(density[y : y + bh, x : x + bw][comp].mean())
        fill = float(area) / max(1.0, float(bw * bh))
        long_side = max(bw, bh)
        short_side = max(1, min(bw, bh))
        aspect = float(long_side) / float(short_side)
        score = float(region_density * min(1.0, fill * 4.0))

        mask[y : y + bh, x : x + bw] |= comp
        regions.append(
            {
                "id": len(regions) + 1,
                "x": x,
                "y": y,
                "w": bw,
                "h": bh,
                "area": area,
                "fill": round(fill, 5),
                "aspect": round(aspect, 5),
                "mean_edge_density": round(region_density, 5),
                "score": round(score, 5),
            }
        )

    regions.sort(key=lambda r: r["area"], reverse=True)
    for new_id, region in enumerate(regions, 1):
        region["id"] = new_id
    return mask, regions


def detect_text_regions(image_bgr, config=None):
    config = config or TextRegionConfig()
    raw_mask, density = build_text_like_mask(
        image_bgr,
        edge_window=config.edge_window,
        high_percentile=config.high_percentile,
        min_density=config.min_density,
        close_kernel=config.close_kernel,
        dilate_kernel=config.dilate_kernel,
    )
    mask, regions = extract_regions(raw_mask, density, config)
    return TextDetectionResult(mask=mask, density=density, raw_mask=raw_mask, regions=regions)


def load_text_mask(mask_dir: Path, image_stem: str, suffix="_text_mask.png"):
    name = f"{image_stem}{suffix}"
    candidates = [mask_dir / name, mask_dir / "masks" / name]
    mask = None
    for path in candidates:
        mask = imread_gray(path)
        if mask is not None:
            break
    if mask is None:
        return None
    return mask > 0


def overlay_mask(image_bgr, mask, color=(255, 220, 20), alpha=0.45):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    overlay[mask] = (overlay[mask].astype(np.float32) * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    return overlay


def draw_regions(rgb, regions, scale_x=1.0, scale_y=1.0):
    out = rgb.copy()
    for region in regions:
        x0 = int(region["x"] * scale_x)
        y0 = int(region["y"] * scale_y)
        x1 = int((region["x"] + region["w"]) * scale_x)
        y1 = int((region["y"] + region["h"]) * scale_y)
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 220, 20), 2)
        cv2.putText(out, str(region["id"]), (x0 + 3, max(18, y0 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 20), 2, cv2.LINE_AA)
    return out


def save_panel(image_bgr, overlay_rgb, density, out_path):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    small_w = 760
    scale = small_w / rgb.shape[1]
    small_h = max(1, int(rgb.shape[0] * scale))
    density_norm = np.clip(density / max(1e-6, float(density.max())) * 255.0, 0, 255).astype(np.uint8)
    density_rgb = cv2.cvtColor(density_norm, cv2.COLOR_GRAY2RGB)

    parts = []
    for title, arr in (("image", rgb), ("text regions", overlay_rgb), ("edge density", density_rgb)):
        small = cv2.resize(arr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(small, (0, 0), (small_w, 30), (255, 255, 255), -1)
        cv2.putText(small, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        parts.append(small)
    panel = np.concatenate(parts, axis=1)
    imwrite(out_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def write_contact_sheet(panel_paths, out_path, thumb_w=420, cols=3):
    thumbs = []
    for path in panel_paths:
        img = imread_color(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        thumb_h = max(1, int(h * thumb_w / w))
        thumbs.append(cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    rows = []
    for i in range(0, len(thumbs), cols):
        row = thumbs[i : i + cols]
        max_h = max(t.shape[0] for t in row)
        padded = []
        for thumb in row:
            if thumb.shape[0] < max_h:
                thumb = cv2.copyMakeBorder(thumb, 0, max_h - thumb.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            padded.append(thumb)
        while len(padded) < cols:
            padded.append(np.full((max_h, thumb_w, 3), 255, dtype=np.uint8))
        rows.append(np.concatenate(padded, axis=1))
    sheet = np.concatenate(rows, axis=0)
    imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def save_outputs(image_path, image_bgr, result: TextDetectionResult, config: TextRegionConfig, out_dir: Path):
    for name in ("masks", "raw_masks", "density", "overlays", "panels", "regions"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    mask_u8 = result.mask.astype(np.uint8) * 255
    raw_u8 = result.raw_mask.astype(np.uint8) * 255
    density_u8 = np.clip(result.density / max(1e-6, float(result.density.max())) * 255.0, 0, 255).astype(np.uint8)

    imwrite(out_dir / "masks" / f"{stem}_text_mask.png", mask_u8)
    imwrite(out_dir / "raw_masks" / f"{stem}_raw_text_mask.png", raw_u8)
    imwrite(out_dir / "density" / f"{stem}_density.png", density_u8)

    overlay_rgb = overlay_mask(image_bgr, result.mask)
    overlay_rgb = draw_regions(overlay_rgb, result.regions)
    imwrite(out_dir / "overlays" / f"{stem}_text_overlay.jpg", cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    save_panel(image_bgr, overlay_rgb, result.density, out_dir / "panels" / f"{stem}_text_panel.jpg")

    payload = {
        "image": str(image_path),
        "width": int(image_bgr.shape[1]),
        "height": int(image_bgr.shape[0]),
        "config": asdict(config),
        "regions": result.regions,
    }
    (out_dir / "regions" / f"{stem}_text_regions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_row(image_path, image_bgr, result: TextDetectionResult):
    total = float(result.mask.size)
    text_pixels = int(np.count_nonzero(result.mask))
    largest = max((int(r["area"]) for r in result.regions), default=0)
    mean_score = float(np.mean([r["score"] for r in result.regions])) if result.regions else 0.0
    return {
        "image": image_path.name,
        "width": image_bgr.shape[1],
        "height": image_bgr.shape[0],
        "text_region_count": len(result.regions),
        "text_pixels": text_pixels,
        "text_percent": round(text_pixels * 100.0 / total, 4),
        "largest_region_area": largest,
        "mean_region_score": round(mean_score, 5),
    }


def main():
    parser = argparse.ArgumentParser(description="Detect text or engraved-text-like regions before crack suppression.")
    parser.add_argument("--src", required=True, help="Image file or image directory.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--edge-window", type=int, default=41)
    parser.add_argument("--high-percentile", type=float, default=96.0)
    parser.add_argument("--min-density", type=float, default=0.08)
    parser.add_argument("--close-kernel", type=int, default=7)
    parser.add_argument("--dilate-kernel", type=int, default=3)
    parser.add_argument("--min-region-area", type=int, default=5000)
    parser.add_argument("--max-region-area-ratio", type=float, default=0.40)
    parser.add_argument("--pad", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    config = TextRegionConfig(
        edge_window=args.edge_window,
        high_percentile=args.high_percentile,
        min_density=args.min_density,
        close_kernel=args.close_kernel,
        dilate_kernel=args.dilate_kernel,
        min_region_area=args.min_region_area,
        max_region_area_ratio=args.max_region_area_ratio,
        pad=args.pad,
    )

    images = list_images(src, args.recursive)
    if not images:
        raise SystemExit(f"No images found in {src}")

    rows = []
    for idx, image_path in enumerate(images, 1):
        image = imread_color(image_path)
        if image is None:
            print(f"[{idx}/{len(images)}] skip unreadable: {image_path}")
            continue
        result = detect_text_regions(image, config)
        save_outputs(image_path, image, result, config, out)
        row = summarize_row(image_path, image, result)
        rows.append(row)
        print(f"[{idx}/{len(images)}] {image_path.name} text={row['text_percent']}% regions={row['text_region_count']}")

    if rows:
        with (out / "text_detection_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_contact_sheet(sorted((out / "panels").glob("*_text_panel.jpg")), out / "contact_sheet.jpg")
    print(f"out={out}")


if __name__ == "__main__":
    main()
