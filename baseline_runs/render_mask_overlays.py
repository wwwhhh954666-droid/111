import argparse
from pathlib import Path

import cv2
import numpy as np

from predict_unlabeled_images import write_contact_sheet


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(src: Path):
    return sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    image_dir = Path(args.images)
    mask_dir = Path(args.masks)
    out = Path(args.out)
    overlay_dir = out / "overlays"
    panel_dir = out / "panels"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for image_path in list_images(image_dir):
        mask_path = mask_dir / f"{image_path.stem}_pred.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay = rgb.copy()
        overlay[mask > 0] = (overlay[mask > 0] * 0.55 + np.array([255, 40, 40]) * 0.45).astype(np.uint8)
        cv2.imwrite(str(overlay_dir / f"{image_path.stem}_overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

        small_w = 900
        scale = small_w / rgb.shape[1]
        small_h = max(1, int(rgb.shape[0] * scale))
        rgb_s = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
        overlay_s = cv2.resize(overlay, (small_w, small_h), interpolation=cv2.INTER_AREA)
        mask_rgb = np.zeros_like(rgb)
        mask_rgb[mask > 0] = (255, 40, 40)
        mask_s = cv2.resize(mask_rgb, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        panel = np.concatenate([rgb_s, overlay_s, mask_s], axis=1)
        cv2.imwrite(str(panel_dir / f"{image_path.stem}_panel.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        count += 1

    write_contact_sheet(sorted(panel_dir.glob("*_panel.jpg")), out / "contact_sheet.jpg")
    print(f"rendered={count}")
    print(f"out={out}")


if __name__ == "__main__":
    main()
