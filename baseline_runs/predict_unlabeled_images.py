import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from train_mask_baselines import make_model


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
COLORS = np.array([[0, 0, 0], [255, 40, 40], [40, 180, 255]], dtype=np.uint8)


def list_images(src: Path):
    return sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def make_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, max(1, length - tile + 1), stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def preprocess(rgb):
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    return (x - MEAN) / STD


@torch.no_grad()
def predict_image(model, image_bgr, device, tile=512, stride=384, batch=8):
    h, w = image_bgr.shape[:2]
    prob_sum = np.zeros((3, h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    xs = make_starts(w, tile, stride)
    ys = make_starts(h, tile, stride)
    items = []

    for y in ys:
        for x in xs:
            patch = image_bgr[y : y + tile, x : x + tile]
            rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            items.append((x, y, preprocess(rgb)))
            if len(items) >= batch:
                flush_tiles(model, items, prob_sum, count, device)
                items = []
    if items:
        flush_tiles(model, items, prob_sum, count, device)

    prob = prob_sum / np.maximum(count[None, :, :], 1e-6)
    pred = prob.argmax(0).astype(np.uint8)
    conf = prob.max(0).astype(np.float32)
    return pred, conf, prob


@torch.no_grad()
def flush_tiles(model, items, prob_sum, count, device):
    batch_tensor = torch.stack([item[2] for item in items], 0).to(device)
    logits = model(batch_tensor)
    logits = logits["out"] if isinstance(logits, dict) else logits
    probs = F.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
    for (x, y, _), p in zip(items, probs):
        tile_h, tile_w = p.shape[1:]
        prob_sum[:, y : y + tile_h, x : x + tile_w] += p
        count[y : y + tile_h, x : x + tile_w] += 1.0


def save_outputs(image_path, image_bgr, pred, conf, out_dir, alpha=0.42):
    mask_dir = out_dir / "masks"
    overlay_dir = out_dir / "overlays"
    panel_dir = out_dir / "panels"
    conf_dir = out_dir / "confidence"
    for d in (mask_dir, overlay_dir, panel_dir, conf_dir):
        d.mkdir(parents=True, exist_ok=True)

    color = COLORS[pred]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = np.where(
        pred[..., None] > 0,
        (image_rgb * (1.0 - alpha) + color * alpha).astype(np.uint8),
        image_rgb,
    )

    stem = image_path.stem
    cv2.imwrite(str(mask_dir / f"{stem}_pred.png"), pred)
    cv2.imwrite(str(conf_dir / f"{stem}_conf.png"), np.clip(conf * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(str(overlay_dir / f"{stem}_overlay.jpg"), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    small_w = 900
    scale = small_w / image_rgb.shape[1]
    small_h = max(1, int(image_rgb.shape[0] * scale))
    rgb_s = cv2.resize(image_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
    overlay_s = cv2.resize(overlay_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
    mask_color_s = cv2.resize(COLORS[pred], (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    panel = np.concatenate([rgb_s, overlay_s, mask_color_s], axis=1)
    cv2.imwrite(str(panel_dir / f"{stem}_panel.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def write_contact_sheet(panel_paths, out_path, thumb_w=420, cols=3):
    thumbs = []
    for p in panel_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
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
        for t in row:
            if t.shape[0] < max_h:
                t = cv2.copyMakeBorder(t, 0, max_h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            padded.append(t)
        while len(padded) < cols:
            padded.append(np.full((max_h, thumb_w, 3), 255, dtype=np.uint8))
        rows.append(np.concatenate(padded, axis=1))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = list_images(src)
    if not images:
        raise SystemExit(f"No images found in {src}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt.get("name") or ckpt.get("model") or "dual_lraspp_aux_mnv3"
    model = make_model(model_name).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = []
    for i, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        pred, conf, _ = predict_image(model, image, device, args.tile, args.stride, args.batch)
        save_outputs(image_path, image, pred, conf, out)
        total = float(pred.size)
        rows.append({
            "image": image_path.name,
            "width": image.shape[1],
            "height": image.shape[0],
            "crack_pixels": int((pred == 1).sum()),
            "spall_pixels": int((pred == 2).sum()),
            "crack_percent": round(float((pred == 1).sum()) * 100.0 / total, 4),
            "spall_percent": round(float((pred == 2).sum()) * 100.0 / total, 4),
            "mean_confidence": round(float(conf.mean()), 4),
        })
        print(f"[{i}/{len(images)}] {image_path.name} crack={rows[-1]['crack_percent']}% spall={rows[-1]['spall_percent']}%")

    with (out / "prediction_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    panel_paths = sorted((out / "panels").glob("*_panel.jpg"))
    write_contact_sheet(panel_paths, out / "contact_sheet.jpg")
    print(f"out={out}")


if __name__ == "__main__":
    main()
