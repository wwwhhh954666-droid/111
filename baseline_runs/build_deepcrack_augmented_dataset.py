import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def copy_split(src_root: Path, dst_root: Path, split: str):
    src_images = src_root / split / "images"
    src_masks = src_root / split / "masks"
    dst_images = dst_root / split / "images"
    dst_masks = dst_root / split / "masks"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_masks.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_path in sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMG_EXTS):
        mask_path = src_masks / f"{img_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {img_path}: {mask_path}")
        shutil.copy2(img_path, dst_images / img_path.name)
        shutil.copy2(mask_path, dst_masks / mask_path.name)
        copied += 1
    return copied


def add_deepcrack(raw_root: Path, dst_root: Path, size: int, include_test: bool):
    dst_images = dst_root / "train" / "images"
    dst_masks = dst_root / "train" / "masks"
    added = 0
    sources = ["train"]
    if include_test:
        sources.append("test")

    for source in sources:
        img_dir = raw_root / f"{source}_img"
        lab_dir = raw_root / f"{source}_lab"
        for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            lab_path = lab_dir / f"{img_path.stem}.png"
            if not lab_path.exists():
                raise FileNotFoundError(f"Missing DeepCrack label for {img_path}: {lab_path}")

            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            lab = cv2.imread(str(lab_path), cv2.IMREAD_GRAYSCALE)
            if img is None or lab is None:
                raise ValueError(f"Unreadable DeepCrack pair: {img_path}, {lab_path}")

            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            lab = cv2.resize(lab, (size, size), interpolation=cv2.INTER_NEAREST)
            mask = (lab > 127).astype(np.uint8)

            out_stem = f"deepcrack_{source}_{img_path.stem}"
            cv2.imwrite(str(dst_images / f"{out_stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(dst_masks / f"{out_stem}.png"), mask)
            added += 1
    return added


def mask_hist(root: Path):
    hist = {0: 0, 1: 0, 2: 0}
    for mask_path in sorted((root / "train" / "masks").glob("*.png")):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        values, counts = np.unique(mask, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            hist[int(value)] = hist.get(int(value), 0) + int(count)
    return hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=r"D:\mask\baseline_runs\strategy_patch_dataset_cs_newdata_hn_reviewed")
    parser.add_argument("--deepcrack", default=r"D:\mask\baseline_runs\external_deepcrack_raw")
    parser.add_argument("--out", default=r"D:\mask\baseline_runs\strategy_patch_dataset_cs_newdata_hn_deepcrack")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()

    base = Path(args.base)
    raw = Path(args.deepcrack)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Output already exists, choose a new path or remove it first: {out}")

    train_base = copy_split(base, out, "train")
    val_base = copy_split(base, out, "val")
    deepcrack_added = add_deepcrack(raw, out, args.size, args.include_test)

    summary = {
        "base": str(base),
        "deepcrack": str(raw),
        "out": str(out),
        "train_base_images": train_base,
        "val_base_images": val_base,
        "deepcrack_train_added": deepcrack_added,
        "include_test": args.include_test,
        "size": args.size,
        "train_images_total": len(list((out / "train" / "images").glob("*.jpg"))),
        "val_images_total": len(list((out / "val" / "images").glob("*.jpg"))),
        "train_mask_hist": mask_hist(out),
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
