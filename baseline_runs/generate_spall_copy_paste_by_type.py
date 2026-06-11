import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from build_synthetic_damage_dataset import imread_color
from generate_spall_copy_paste_examples import (
    list_negative_images,
    load_spall_templates,
    make_panel,
    paste_spall,
    resize_rotate_template,
)
from generate_text_negative_synthetic_examples import mask_to_labelme_shapes


def template_features(template):
    mask = (template["mask"] > 0).astype(np.uint8)
    area = float(mask.sum())
    contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"area": area, "fill": 0.0, "aspect": 1.0, "edge_density": 0.0}
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    fill = area / max(1.0, float(w * h))
    aspect = max(w, h) / max(1, min(w, h))
    peri = cv2.arcLength(contour, True)
    edge_density = peri / max(1.0, np.sqrt(area))
    return {"area": area, "fill": fill, "aspect": aspect, "edge_density": edge_density}


def split_templates(templates):
    enriched = []
    for t in templates:
        f = template_features(t)
        row = dict(t)
        row.update(f)
        enriched.append(row)
    # Layered/uncertain boundary: elongated or ragged, usually lower fill.
    layered = [t for t in enriched if t["aspect"] >= 2.0 or t["fill"] < 0.42 or t["edge_density"] > 7.0]
    # Chunk missing: compact, larger filled region.
    chunk = [t for t in enriched if t["aspect"] < 2.2 and t["fill"] >= 0.32]
    if len(layered) < 5:
        layered = sorted(enriched, key=lambda t: (t["aspect"], -t["fill"]), reverse=True)[: max(5, len(enriched) // 3)]
    if len(chunk) < 5:
        chunk = sorted(enriched, key=lambda t: (t["fill"], t["area"]), reverse=True)[: max(5, len(enriched) // 3)]
    return layered, chunk


def write_one(out, mode, idx, bg_path, bg, tmpl, rng):
    target = rng.randint(220, 720) if mode == "layered" else rng.randint(180, 560)
    donor, donor_mask = resize_rotate_template(tmpl, rng, target)
    aug, mask = paste_spall(bg, donor, donor_mask, rng)
    stem = f"{mode}_spall_cp_{idx:02d}_{bg_path.stem}"
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
    return {
        "mode": mode,
        "background": str(bg_path),
        "template": tmpl["source"],
        "image": str(image_path),
        "panel": str(panel_path),
        "spall_percent": round(float((mask == 2).sum()) * 100.0 / float(mask.size), 4),
        "template_fill": round(float(tmpl.get("fill", 0)), 4),
        "template_aspect": round(float(tmpl.get("aspect", 0)), 4),
        "template_edge_density": round(float(tmpl.get("edge_density", 0)), 4),
    }


def contact_sheet(panel_paths, out_path):
    thumbs = []
    for p in panel_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        tw = 900
        th = max(1, int(h * tw / w))
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (tw, 26), (255, 255, 255), -1)
        cv2.putText(img, p.name.replace("_panel.jpg", "")[:80], (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        thumbs.append(img)
    rows = []
    for i in range(0, len(thumbs), 2):
        row = thumbs[i : i + 2]
        mh = max(x.shape[0] for x in row)
        padded = []
        for x in row:
            if x.shape[0] < mh:
                x = cv2.copyMakeBorder(x, 0, mh - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            padded.append(x)
        while len(padded) < 2:
            padded.append(np.full((mh, 900, 3), 255, dtype=np.uint8))
        rows.append(cv2.hconcat(padded))
    if rows:
        cv2.imwrite(str(out_path), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backgrounds", default=r"E:\1\xiuzheng1")
    parser.add_argument("--template-roots", nargs="+", default=[r"D:\mask\shujuji\shujuji4"])
    parser.add_argument("--out", default=r"E:\1\text\spall_copy_paste_by_type")
    parser.add_argument("--per-type", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260615)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    for sub in ("images", "masks", "overlays", "panels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    templates = load_spall_templates(args.template_roots, prefer_names={"IMG_6733.JPG"}, max_templates=400)
    layered, chunk = split_templates(templates)
    negatives = list_negative_images(Path(args.backgrounds))
    rows = []
    panel_paths = []
    for mode, pool in (("layered", layered), ("chunk", chunk)):
        for i in range(1, args.per_type + 1):
            bg_path = rng.choice(negatives)
            bg = imread_color(bg_path)
            if bg is None:
                continue
            tmpl = rng.choice(pool[: min(120, len(pool))])
            row = write_one(out, mode, i, bg_path, bg, tmpl, rng)
            rows.append(row)
            panel_paths.append(Path(row["panel"]))
            print(f"{mode} {i}: {Path(row['template']).name} spall={row['spall_percent']}% aspect={row['template_aspect']} fill={row['template_fill']}")
    (out / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_sheet(panel_paths, out / "contact_sheet.jpg")
    print(f"templates total={len(templates)} layered={len(layered)} chunk={len(chunk)} out={out}")


if __name__ == "__main__":
    main()
