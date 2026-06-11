import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

from build_synthetic_damage_dataset import IMG_EXTS, imread_color
from generate_text_negative_synthetic_examples import mask_to_labelme_shapes, save_overlay


LABEL_MAP = {"crack": 1, "creak": 1, "spall": 2}


def list_negative_images(root: Path):
    images = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    return [p for p in images if not p.with_suffix(".json").exists()]


def load_templates(roots):
    templates = {1: [], 2: []}
    for root in roots:
        root = Path(root)
        for jp in root.rglob("*.json"):
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                continue
            for shape in data.get("shapes", []):
                cls = LABEL_MAP.get(str(shape.get("label", "")).strip().lower())
                pts = np.asarray(shape.get("points", []), dtype=np.float32)
                if cls is None or len(pts) < 3:
                    continue
                x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
                if w < 8 or h < 8:
                    continue
                local = pts - np.array([x, y], dtype=np.float32) + 8.0
                mask = np.zeros((h + 16, w + 16), dtype=np.uint8)
                cv2.fillPoly(mask, [local.astype(np.int32)], 255)
                area = int((mask > 0).sum())
                aspect = max(w, h) / max(1, min(w, h))
                if area < 80:
                    continue
                templates[cls].append({"mask": mask, "aspect": aspect, "area": area})
    return templates


def transform_template(template, rng, target_long_range):
    src = template["mask"]
    h, w = src.shape[:2]
    long_side = rng.randint(*target_long_range)
    scale = long_side / max(h, w)
    if rng.random() < 0.45:
        scale *= rng.uniform(0.75, 1.25)
    new_w = max(8, int(w * scale))
    new_h = max(8, int(h * scale))
    m = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    angle = rng.uniform(0, 180)
    center = (new_w / 2, new_h / 2)
    rot = cv2.getRotationMatrix2D(center, angle, rng.uniform(0.85, 1.15))
    cos = abs(rot[0, 0])
    sin = abs(rot[0, 1])
    bound_w = int(new_h * sin + new_w * cos) + 8
    bound_h = int(new_h * cos + new_w * sin) + 8
    rot[0, 2] += bound_w / 2 - center[0]
    rot[1, 2] += bound_h / 2 - center[1]
    return cv2.warpAffine(m, rot, (bound_w, bound_h), flags=cv2.INTER_NEAREST, borderValue=0)


def place_mask(dst_shape, obj_mask, rng):
    h, w = dst_shape
    oh, ow = obj_mask.shape[:2]
    if oh >= h or ow >= w:
        scale = min((h - 16) / max(1, oh), (w - 16) / max(1, ow), 1.0)
        obj_mask = cv2.resize(obj_mask, (max(8, int(ow * scale)), max(8, int(oh * scale))), interpolation=cv2.INTER_NEAREST)
        oh, ow = obj_mask.shape[:2]
    x = rng.randint(0, max(0, w - ow))
    y = rng.randint(0, max(0, h - oh))
    canvas = np.zeros((h, w), dtype=np.uint8)
    canvas[y : y + oh, x : x + ow] = obj_mask
    return canvas


def darken_crack(image, crack_mask, rng):
    edge = cv2.dilate(crack_mask, np.ones((5, 5), np.uint8))
    halo = (edge > 0) & (crack_mask == 0)
    core = crack_mask > 0
    image[halo] = np.clip(image[halo].astype(np.float32) * rng.uniform(0.80, 0.94), 0, 255).astype(np.uint8)
    image[core] = np.clip(
        image[core].astype(np.float32) * rng.uniform(0.25, 0.50) - rng.uniform(2, 16),
        0,
        255,
    ).astype(np.uint8)


def add_flaky_spall_texture(image, spall_mask, rng):
    region = spall_mask > 0
    if not np.any(region):
        return
    h, w = spall_mask.shape
    base_factor = rng.uniform(0.72, 1.10)
    base_bias = rng.uniform(-28, 18)
    noise = cv2.GaussianBlur(np.random.normal(0, rng.uniform(6, 15), image.shape).astype(np.float32), (0, 0), 2)
    image[region] = np.clip(image[region].astype(np.float32) * base_factor + base_bias + noise[region], 0, 255).astype(np.uint8)

    contours, _ = cv2.findContours((spall_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 400:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        flake_count = max(3, min(22, int(area / 18000)))
        for _ in range(flake_count):
            cx = rng.randint(x, x + max(1, bw - 1))
            cy = rng.randint(y, y + max(1, bh - 1))
            if spall_mask[cy, cx] == 0:
                continue
            ax = rng.randint(10, max(12, min(70, bw // 3 + 10)))
            ay = rng.randint(4, max(6, min(28, bh // 5 + 6)))
            angle = rng.uniform(0, 180)
            flake = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(flake, (cx, cy), (ax, ay), angle, 0, 360, 255, -1, cv2.LINE_AA)
            flake = cv2.bitwise_and(flake, spall_mask)
            fb = flake > 0
            if not np.any(fb):
                continue
            shade = rng.uniform(0.78, 1.18)
            bias = rng.uniform(-20, 18)
            image[fb] = np.clip(image[fb].astype(np.float32) * shade + bias, 0, 255).astype(np.uint8)
            rim = (cv2.dilate(flake, np.ones((3, 3), np.uint8)) > 0) & (flake == 0) & region
            image[rim] = np.clip(image[rim].astype(np.float32) * rng.uniform(0.60, 0.85), 0, 255).astype(np.uint8)

    boundary = cv2.dilate(spall_mask, np.ones((7, 7), np.uint8)) - cv2.erode(spall_mask, np.ones((5, 5), np.uint8))
    bd = boundary > 0
    image[bd] = np.clip(image[bd].astype(np.float32) * rng.uniform(0.55, 0.78), 0, 255).astype(np.uint8)


def choose_template(templates, cls, rng, min_aspect=None):
    pool = templates[cls]
    if min_aspect:
        filtered = [t for t in pool if t["aspect"] >= min_aspect]
        if filtered:
            pool = filtered
    return rng.choice(pool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=r"E:\1\xiuzheng1")
    parser.add_argument("--out", default=r"E:\1\text_mimic")
    parser.add_argument("--template-roots", nargs="+", default=[r"D:\mask\shujuji", r"E:\1\xiuzheng1"])
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    src = Path(args.src)
    out = Path(args.out)
    for sub in ("images", "masks", "overlays"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    templates = load_templates(args.template_roots)
    if not templates[1] or not templates[2]:
        raise SystemExit(f"Need crack and spall templates, got crack={len(templates[1])} spall={len(templates[2])}")
    negatives = list_negative_images(src)
    selected = rng.sample(negatives, min(args.count, len(negatives)))
    manifest = []
    for i, image_path in enumerate(selected, 1):
        image = imread_color(image_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        image_aug = image.copy()

        crack_count = rng.randint(1, 3)
        for _ in range(crack_count):
            tmpl = choose_template(templates, 1, rng, min_aspect=2.0)
            obj = transform_template(tmpl, rng, (120, 420))
            crack = place_mask((h, w), obj, rng)
            crack = cv2.morphologyEx(crack, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            darken_crack(image_aug, crack, rng)
            mask[crack > 0] = 1

        spall_count = rng.randint(1, 2)
        for _ in range(spall_count):
            tmpl = choose_template(templates, 2, rng)
            obj = transform_template(tmpl, rng, (110, 360))
            spall = place_mask((h, w), obj, rng)
            spall = cv2.morphologyEx(spall, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            add_flaky_spall_texture(image_aug, spall, rng)
            mask[spall > 0] = 2

        stem = f"text_mimic_{i:02d}_{image_path.stem}"
        out_image = out / "images" / f"{stem}.jpg"
        out_mask = out / "masks" / f"{stem}_mask.png"
        out_overlay = out / "overlays" / f"{stem}_overlay.jpg"
        out_json = out / "images" / f"{stem}.json"
        cv2.imwrite(str(out_image), image_aug, [cv2.IMWRITE_JPEG_QUALITY, 94])
        cv2.imwrite(str(out_mask), mask)
        save_overlay(image_aug, mask, out_overlay)
        data = {
            "version": "5.5.0",
            "flags": {},
            "shapes": mask_to_labelme_shapes(mask, epsilon_ratio=0.0025),
            "imagePath": out_image.name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
        }
        out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        total = float(mask.size)
        row = {
            "source": str(image_path),
            "image": str(out_image),
            "mask": str(out_mask),
            "json": str(out_json),
            "overlay": str(out_overlay),
            "crack_percent": round(float((mask == 1).sum()) * 100.0 / total, 4),
            "spall_percent": round(float((mask == 2).sum()) * 100.0 / total, 4),
            "shapes": len(data["shapes"]),
        }
        manifest.append(row)
        print(f"[{i}/{len(selected)}] {image_path.name} crack={row['crack_percent']}% spall={row['spall_percent']}% shapes={row['shapes']}")

    (out / "synthetic_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"templates: crack={len(templates[1])} spall={len(templates[2])}")
    print(f"out={out}")


if __name__ == "__main__":
    main()
