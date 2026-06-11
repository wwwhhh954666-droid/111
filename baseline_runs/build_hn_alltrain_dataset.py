import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def copy_tree(src: Path, dst: Path, clean: bool):
    if clean and dst.exists():
        shutil.rmtree(dst)
    for split in ("train", "val"):
        for sub in ("images", "masks"):
            out_dir = dst / split / sub
            out_dir.mkdir(parents=True, exist_ok=True)
            for p in sorted((src / split / sub).glob("*")):
                if p.is_file():
                    shutil.copy2(p, out_dir / p.name)


def count_zero_masks(root: Path, split: str):
    zero = 0
    nonzero = 0
    for p in sorted((root / split / "masks").glob("*.png")):
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if np.count_nonzero(mask) == 0:
            zero += 1
        else:
            nonzero += 1
    return {"zero": zero, "nonzero": nonzero, "total": zero + nonzero}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hn-images", nargs="+", required=True)
    parser.add_argument("--max-add", type=int, default=0, help="0 means no limit")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    base = Path(args.base)
    out = Path(args.out)
    copy_tree(base, out, args.clean)

    existing_names = set()
    for split in ("train", "val"):
        existing_names.update(p.name for p in (out / split / "images").glob("*") if p.is_file())

    added = []
    skipped_existing = 0
    skipped_bad = 0
    seen_source_names = set()
    for hn_root in [Path(p) for p in args.hn_images]:
        for src in sorted(hn_root.glob("*.jpg")):
            if args.max_add and len(added) >= args.max_add:
                break
            if src.name in seen_source_names:
                skipped_existing += 1
                continue
            seen_source_names.add(src.name)
            out_name = f"hn_all_{src.name}"
            if out_name in existing_names or src.name in existing_names or f"hn_{src.name}" in existing_names:
                skipped_existing += 1
                continue
            image = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if image is None:
                skipped_bad += 1
                continue
            h, w = image.shape[:2]
            shutil.copy2(src, out / "train" / "images" / out_name)
            cv2.imwrite(str(out / "train" / "masks" / f"{Path(out_name).stem}.png"), np.zeros((h, w), dtype=np.uint8))
            existing_names.add(out_name)
            added.append(str(src))

    summary = {
        "base": str(base),
        "out": str(out),
        "hn_images": args.hn_images,
        "added_train_negatives": len(added),
        "skipped_existing": skipped_existing,
        "skipped_bad": skipped_bad,
        "train_images": len(list((out / "train" / "images").glob("*.jpg"))),
        "val_images": len(list((out / "val" / "images").glob("*.jpg"))),
        "train_zero_masks": count_zero_masks(out, "train"),
        "val_zero_masks": count_zero_masks(out, "val"),
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
