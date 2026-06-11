import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from predict_unlabeled_images import MEAN, STD, make_starts, write_contact_sheet
from train_crack_assist import make_model


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def imread_color(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def list_images(src: Path):
    return sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def preprocess(rgb):
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    return (x - MEAN) / STD


@torch.no_grad()
def flush_tiles(model, items, prob_sum, count, device):
    batch_tensor = torch.stack([item[2] for item in items], 0).to(device)
    logits = model(batch_tensor)["out"]
    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy().astype(np.float32)
    for (x, y, _), p in zip(items, probs):
        h, w = p.shape
        prob_sum[y : y + h, x : x + w] += p
        count[y : y + h, x : x + w] += 1.0


@torch.no_grad()
def predict_image(model, image_bgr, device, tile, stride, batch, threshold):
    h, w = image_bgr.shape[:2]
    prob_sum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    items = []
    for y in make_starts(h, tile, stride):
        for x in make_starts(w, tile, stride):
            patch = image_bgr[y : y + tile, x : x + tile]
            rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            items.append((x, y, preprocess(rgb)))
            if len(items) >= batch:
                flush_tiles(model, items, prob_sum, count, device)
                items = []
    if items:
        flush_tiles(model, items, prob_sum, count, device)
    prob = prob_sum / np.maximum(count, 1e-6)
    pred = (prob >= threshold).astype(np.uint8)
    return pred, prob


def remove_small_components(mask, min_area, min_long_side, min_aspect_ratio, max_fill_ratio):
    binary = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        long_side = max(w, h)
        short_side = max(1, min(w, h))
        aspect = long_side / short_side
        fill = area / max(1.0, float(w * h))
        if area >= min_area and long_side >= min_long_side and aspect >= min_aspect_ratio and fill <= max_fill_ratio:
            out[labels == i] = 1
    return out.astype(np.uint8)


def save_outputs(image_path, image_bgr, pred, prob, out_dir):
    for name in ("masks", "overlays", "panels", "confidence"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    cv2.imwrite(str(out_dir / "masks" / f"{stem}_pred.png"), pred)
    cv2.imwrite(str(out_dir / "confidence" / f"{stem}_conf.png"), np.clip(prob * 255, 0, 255).astype(np.uint8))

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    overlay[pred == 1] = (overlay[pred == 1] * 0.55 + np.array([255, 40, 40]) * 0.45).astype(np.uint8)
    cv2.imwrite(str(out_dir / "overlays" / f"{stem}_overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    small_w = 900
    scale = small_w / rgb.shape[1]
    small_h = max(1, int(rgb.shape[0] * scale))
    rgb_s = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
    overlay_s = cv2.resize(overlay, (small_w, small_h), interpolation=cv2.INTER_AREA)
    mask_rgb = np.zeros_like(rgb)
    mask_rgb[pred == 1] = (255, 40, 40)
    mask_s = cv2.resize(mask_rgb, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    panel = np.concatenate([rgb_s, overlay_s, mask_s], axis=1)
    cv2.imwrite(str(out_dir / "panels" / f"{stem}_panel.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--min-area", type=int, default=900)
    parser.add_argument("--min-long-side", type=int, default=100)
    parser.add_argument("--min-aspect-ratio", type=float, default=1.8)
    parser.add_argument("--max-fill-ratio", type=float, default=0.55)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = make_model().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = []
    images = list_images(src)
    for i, image_path in enumerate(images, 1):
        image = imread_color(image_path)
        if image is None:
            continue
        pred, prob = predict_image(model, image, device, args.tile, args.stride, args.batch, args.threshold)
        pred = remove_small_components(pred, args.min_area, args.min_long_side, args.min_aspect_ratio, args.max_fill_ratio)
        save_outputs(image_path, image, pred, prob, out)
        pct = round(float(pred.sum()) * 100.0 / float(pred.size), 4)
        rows.append({"image": image_path.name, "crack_pixels": int(pred.sum()), "crack_percent": pct, "mean_probability": round(float(prob.mean()), 4)})
        print(f"[{i}/{len(images)}] {image_path.name} crack={pct}%")

    with (out / "prediction_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_contact_sheet(sorted((out / "panels").glob("*_panel.jpg")), out / "contact_sheet.jpg")
    print(f"out={out}")


if __name__ == "__main__":
    main()
