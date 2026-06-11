import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from build_synthetic_damage_dataset import IMG_EXTS, imread_color
from generate_annotation_mimic_synthetic_examples import (
    add_flaky_spall_texture,
    choose_template,
    darken_crack,
    list_negative_images,
    load_templates,
    place_mask,
    transform_template,
)
from generate_text_negative_synthetic_examples import mask_to_labelme_shapes, save_overlay


def apply_crack(image, mask, templates, rng, count_range=(2, 4), long_range=(180, 620)):
    h, w = mask.shape
    for _ in range(rng.randint(*count_range)):
        tmpl = choose_template(templates, 1, rng, min_aspect=2.0)
        obj = transform_template(tmpl, rng, long_range)
        crack = place_mask((h, w), obj, rng)
        crack = cv2.morphologyEx(crack, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        darken_crack(image, crack, rng)
        mask[crack > 0] = 1


def apply_spall(image, mask, templates, rng, count_range=(1, 2), long_range=(160, 460)):
    h, w = mask.shape
    for _ in range(rng.randint(*count_range)):
        tmpl = choose_template(templates, 2, rng)
        obj = transform_template(tmpl, rng, long_range)
        spall = place_mask((h, w), obj, rng)
        spall = cv2.morphologyEx(spall, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        add_flaky_spall_texture(image, spall, rng)
        mask[spall > 0] = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=r"E:\1\xiuzheng1")
    parser.add_argument("--out", default=r"E:\1\text\graded_synthetic")
    parser.add_argument("--template-roots", nargs="+", default=[r"D:\mask\shujuji", r"E:\1\xiuzheng1"])
    parser.add_argument("--per-type", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    for sub in ("images", "masks", "overlays"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    templates = load_templates(args.template_roots)
    negatives = list_negative_images(Path(args.src))
    selected = rng.sample(negatives, min(len(negatives), args.per_type * 3))
    modes = ["thin_crack", "flaky_spall", "mixed"]
    manifest = []

    idx = 0
    for mode in modes:
        for _ in range(args.per_type):
            image_path = selected[idx % len(selected)]
            idx += 1
            image = imread_color(image_path)
            if image is None:
                continue
            aug = image.copy()
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            if mode == "thin_crack":
                apply_crack(aug, mask, templates, rng, count_range=(2, 5), long_range=(180, 700))
            elif mode == "flaky_spall":
                apply_spall(aug, mask, templates, rng, count_range=(1, 3), long_range=(180, 520))
            else:
                apply_crack(aug, mask, templates, rng, count_range=(1, 3), long_range=(140, 520))
                apply_spall(aug, mask, templates, rng, count_range=(1, 2), long_range=(150, 460))

            stem = f"{mode}_{len(manifest)+1:02d}_{image_path.stem}"
            out_image = out / "images" / f"{stem}.jpg"
            out_mask = out / "masks" / f"{stem}_mask.png"
            out_overlay = out / "overlays" / f"{stem}_overlay.jpg"
            out_json = out / "images" / f"{stem}.json"
            cv2.imwrite(str(out_image), aug, [cv2.IMWRITE_JPEG_QUALITY, 94])
            cv2.imwrite(str(out_mask), mask)
            save_overlay(aug, mask, out_overlay)
            data = {
                "version": "5.5.0",
                "flags": {},
                "shapes": mask_to_labelme_shapes(mask, epsilon_ratio=0.0025),
                "imagePath": out_image.name,
                "imageData": None,
                "imageHeight": mask.shape[0],
                "imageWidth": mask.shape[1],
            }
            out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            total = float(mask.size)
            manifest.append(
                {
                    "mode": mode,
                    "source": str(image_path),
                    "image": str(out_image),
                    "mask": str(out_mask),
                    "overlay": str(out_overlay),
                    "crack_percent": round(float((mask == 1).sum()) * 100.0 / total, 4),
                    "spall_percent": round(float((mask == 2).sum()) * 100.0 / total, 4),
                    "shapes": len(data["shapes"]),
                }
            )
            print(f"{stem}: crack={manifest[-1]['crack_percent']}% spall={manifest[-1]['spall_percent']}%")
    (out / "synthetic_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out={out}")


if __name__ == "__main__":
    main()
