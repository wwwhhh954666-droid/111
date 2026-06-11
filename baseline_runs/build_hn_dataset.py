import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def copy_tree(src: Path, dst: Path, clean: bool):
    if clean and dst.exists():
        shutil.rmtree(dst)
    for split in ("train", "val"):
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "masks").mkdir(parents=True, exist_ok=True)
        for sub in ("images", "masks"):
            for p in sorted((src / split / sub).glob("*")):
                if p.is_file():
                    shutil.copy2(p, dst / split / sub / p.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=r"D:\mask\baseline_runs\strategy_patch_dataset_cs")
    parser.add_argument("--hn-images", default=r"D:\mask\baseline_runs\clean_negative_candidates_for_review\images")
    parser.add_argument("--out", default=r"D:\mask\baseline_runs\strategy_patch_dataset_cs_wl_hn_reviewed")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-train", type=int, default=420)
    parser.add_argument("--max-val", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base = Path(args.base)
    hn_dir = Path(args.hn_images)
    out = Path(args.out)
    copy_tree(base, out, args.clean)

    hn_images = sorted(hn_dir.glob("*.jpg"))
    rng.shuffle(hn_images)
    val_count = min(args.max_val, round(len(hn_images) * args.val_ratio))
    train_count = min(args.max_train, len(hn_images) - val_count)
    splits = {
        "val": hn_images[:val_count],
        "train": hn_images[val_count : val_count + train_count],
    }

    for split, images in splits.items():
        img_out = out / split / "images"
        mask_out = out / split / "masks"
        for src in images:
            image = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if image is None:
                continue
            h, w = image.shape[:2]
            name = f"hn_{src.name}"
            shutil.copy2(src, img_out / name)
            cv2.imwrite(str(mask_out / f"{Path(name).stem}.png"), np.zeros((h, w), dtype=np.uint8))

    for split in ("train", "val"):
        image_count = len(list((out / split / "images").glob("*.jpg")))
        mask_count = len(list((out / split / "masks").glob("*.png")))
        print(f"{split}: images={image_count} masks={mask_count}")
    print(f"out={out}")


if __name__ == "__main__":
    main()
