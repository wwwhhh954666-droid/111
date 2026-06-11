import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from build_synthetic_damage_dataset import IMG_EXTS, imread_color
from generate_text_negative_synthetic_examples import mask_to_labelme_shapes, save_overlay


def find_image_for_json(json_path: Path):
    for ext in IMG_EXTS:
        p = json_path.with_suffix(ext)
        if p.exists():
            return p
        p = json_path.with_suffix(ext.upper())
        if p.exists():
            return p
    return None


def load_spall_templates(roots, prefer_names=None, max_templates=220):
    prefer_names = set(prefer_names or [])
    rows = []
    for root in roots:
        for jp in Path(root).rglob("*.json"):
            img_path = find_image_for_json(jp)
            if img_path is None:
                continue
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                continue
            image = None
            for shape in data.get("shapes", []):
                if str(shape.get("label", "")).strip().lower() != "spall":
                    continue
                pts = np.asarray(shape.get("points", []), dtype=np.float32)
                if len(pts) < 3:
                    continue
                x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
                if w < 30 or h < 30:
                    continue
                if image is None:
                    image = imread_color(img_path)
                    if image is None:
                        break
                pad = 24
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
                local_pts = pts - np.array([x0, y0], dtype=np.float32)
                crop = image[y0:y1, x0:x1].copy()
                mask = np.zeros(crop.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [local_pts.astype(np.int32)], 255)
                area = int((mask > 0).sum())
                if area < 1000:
                    continue
                score = 1 if img_path.name in prefer_names else 0
                rows.append({"image": crop, "mask": mask, "source": str(img_path), "area": area, "score": score})
    rows.sort(key=lambda r: (r["score"], r["area"]), reverse=True)
    return rows[:max_templates]


def list_negative_images(root: Path):
    images = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    return [p for p in images if not p.with_suffix(".json").exists()]


def resize_rotate_template(template, rng, target_long):
    img = template["image"]
    mask = template["mask"]
    h, w = mask.shape[:2]
    scale = target_long / max(h, w)
    scale *= rng.uniform(0.72, 1.25)
    nw, nh = max(16, int(w * scale)), max(16, int(h * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    angle = rng.uniform(-35, 35)
    center = (nw / 2, nh / 2)
    mat = cv2.getRotationMatrix2D(center, angle, rng.uniform(0.85, 1.12))
    cos, sin = abs(mat[0, 0]), abs(mat[0, 1])
    bw = int(nh * sin + nw * cos) + 12
    bh = int(nh * cos + nw * sin) + 12
    mat[0, 2] += bw / 2 - center[0]
    mat[1, 2] += bh / 2 - center[1]
    img = cv2.warpAffine(img, mat, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, mat, (bw, bh), flags=cv2.INTER_NEAREST, borderValue=0)
    return img, mask


def color_match_lab(src, dst_roi, mask):
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(dst_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    fg = mask > 0
    near = cv2.dilate(mask, np.ones((31, 31), np.uint8)) > 0
    bg = near & (~fg)
    if fg.sum() < 50 or bg.sum() < 50:
        return src
    out = src_lab.copy()
    for c in range(3):
        s_mean, s_std = src_lab[..., c][fg].mean(), src_lab[..., c][fg].std() + 1e-6
        d_mean, d_std = dst_lab[..., c][bg].mean(), dst_lab[..., c][bg].std() + 1e-6
        # Keep the donor identity, but reduce obvious pasted color gaps.
        target_std = 0.55 * s_std + 0.45 * d_std
        target_mean = 0.45 * s_mean + 0.55 * d_mean
        out[..., c] = (out[..., c] - s_mean) / s_std * target_std + target_mean
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def paste_spall(background, donor, donor_mask, rng):
    h, w = background.shape[:2]
    dh, dw = donor_mask.shape[:2]
    if dh >= h or dw >= w:
        scale = min((h - 20) / max(1, dh), (w - 20) / max(1, dw), 1.0)
        donor = cv2.resize(donor, (max(16, int(dw * scale)), max(16, int(dh * scale))), interpolation=cv2.INTER_LINEAR)
        donor_mask = cv2.resize(donor_mask, (donor.shape[1], donor.shape[0]), interpolation=cv2.INTER_NEAREST)
        dh, dw = donor_mask.shape[:2]
    x = rng.randint(0, max(0, w - dw))
    y = rng.randint(0, max(0, h - dh))
    roi = background[y : y + dh, x : x + dw].copy()
    donor = color_match_lab(donor, roi, donor_mask)
    feather = cv2.GaussianBlur((donor_mask.astype(np.float32) / 255.0), (0, 0), rng.uniform(2.0, 5.0))
    feather = np.clip(feather[..., None], 0.0, 1.0)
    blended = (roi.astype(np.float32) * (1.0 - feather) + donor.astype(np.float32) * feather).astype(np.uint8)

    # Add a subtle darker broken rim; it helps retain the "missing layer" cue.
    rim = cv2.dilate(donor_mask, np.ones((5, 5), np.uint8)) - cv2.erode(donor_mask, np.ones((3, 3), np.uint8))
    rim_bool = rim > 0
    blended[rim_bool] = np.clip(blended[rim_bool].astype(np.float32) * rng.uniform(0.82, 0.93), 0, 255).astype(np.uint8)

    out = background.copy()
    out[y : y + dh, x : x + dw] = blended
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y : y + dh, x : x + dw][donor_mask > 0] = 2
    return out, mask


def make_panel(before, after, mask, overlay_path, panel_path):
    save_overlay(after, mask, overlay_path)
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    target_w = 520
    parts = []
    for title, img in (("source negative", before), ("copy-paste spall", after), ("mask overlay", overlay)):
        h, w = img.shape[:2]
        th = max(1, int(h * target_w / w))
        small = cv2.resize(img, (target_w, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(small, (0, 0), (target_w, 30), (255, 255, 255), -1)
        cv2.putText(small, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        parts.append(small)
    max_h = max(p.shape[0] for p in parts)
    padded = [cv2.copyMakeBorder(p, 0, max_h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in parts]
    cv2.imwrite(str(panel_path), cv2.hconcat(padded), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backgrounds", default=r"E:\1\xiuzheng1")
    parser.add_argument("--template-roots", nargs="+", default=[r"D:\mask\shujuji\shujuji4"])
    parser.add_argument("--out", default=r"E:\1\text\spall_copy_paste_v1")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    templates = load_spall_templates(args.template_roots, prefer_names={"IMG_6733.JPG"})
    if not templates:
        raise SystemExit("No spall templates found.")
    negatives = list_negative_images(Path(args.backgrounds))
    if not negatives:
        raise SystemExit("No negative background images found.")
    out = Path(args.out)
    for sub in ("images", "masks", "overlays", "panels"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    manifest = []
    for i in range(args.count):
        bg_path = rng.choice(negatives)
        bg = imread_color(bg_path)
        if bg is None:
            continue
        tmpl = rng.choice(templates[: min(60, len(templates))])
        long_side = rng.randint(180, 620)
        donor, donor_mask = resize_rotate_template(tmpl, rng, long_side)
        aug, mask = paste_spall(bg, donor, donor_mask, rng)
        stem = f"spall_cp_{i+1:02d}_{bg_path.stem}"
        image_path = out / "images" / f"{stem}.jpg"
        mask_path = out / "masks" / f"{stem}_mask.png"
        json_path = out / "images" / f"{stem}.json"
        overlay_path = out / "overlays" / f"{stem}_overlay.jpg"
        panel_path = out / "panels" / f"{stem}_panel.jpg"
        cv2.imwrite(str(image_path), aug, [cv2.IMWRITE_JPEG_QUALITY, 94])
        cv2.imwrite(str(mask_path), mask)
        make_panel(bg, aug, mask, overlay_path, panel_path)
        data = {
            "version": "5.5.0",
            "flags": {},
            "shapes": mask_to_labelme_shapes(mask, epsilon_ratio=0.0025),
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": mask.shape[0],
            "imageWidth": mask.shape[1],
        }
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append(
            {
                "background": str(bg_path),
                "template": tmpl["source"],
                "image": str(image_path),
                "mask": str(mask_path),
                "overlay": str(overlay_path),
                "panel": str(panel_path),
                "spall_percent": round(float((mask == 2).sum()) * 100.0 / float(mask.size), 4),
                "shapes": len(data["shapes"]),
            }
        )
        print(f"[{i+1}/{args.count}] {bg_path.name} template={Path(tmpl['source']).name} spall={manifest[-1]['spall_percent']}%")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"templates={len(templates)} out={out}")


if __name__ == "__main__":
    main()
