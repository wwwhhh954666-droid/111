import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def copy_dataset(base: Path, out: Path):
    for split in ("train", "val"):
        for sub in ("images", "masks"):
            dst = out / split / sub
            dst.mkdir(parents=True, exist_ok=True)
            for src in sorted((base / split / sub).glob("*")):
                if src.is_file():
                    shutil.copy2(src, dst / src.name)


def mask_ratios(mask_path: Path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Unreadable mask: {mask_path}")
    total = float(mask.size)
    return {
        "crack": float((mask == 1).sum()) / total,
        "spall": float((mask == 2).sum()) / total,
        "damage": float((mask > 0).sum()) / total,
    }


def collect_candidates(s4_root: Path):
    rows = []
    for split in ("train", "val"):
        mask_dir = s4_root / split / "masks"
        image_dir = s4_root / split / "images"
        if not mask_dir.exists():
            continue
        for mask_path in sorted(mask_dir.glob("*.png")):
            image_path = image_dir / f"{mask_path.stem}.jpg"
            if not image_path.exists():
                continue
            ratios = mask_ratios(mask_path)
            rows.append({"image": image_path, "mask": mask_path, "split": split, **ratios})
    return rows


def sample_balanced(rows, limit, rng):
    spall = [r for r in rows if 0.02 <= r["spall"] <= 0.75 and r["damage"] >= 0.05]
    crack = [r for r in rows if r["crack"] >= 0.02 and r["spall"] <= 0.25]
    mixed = [r for r in rows if r["crack"] >= 0.01 and r["spall"] >= 0.01]
    damage = [r for r in rows if 0.02 <= r["damage"] <= 0.75]

    selected = []
    seen = set()

    def take(pool, n):
        pool = list(pool)
        rng.shuffle(pool)
        added = 0
        for item in pool:
            key = item["mask"].as_posix()
            if key in seen:
                continue
            selected.append(item)
            seen.add(key)
            added += 1
            if len(selected) >= limit or added >= n:
                break

    take(spall, round(limit * 0.45))
    take(crack, round(limit * 0.35))
    take(mixed, round(limit * 0.10))
    take(damage, limit)
    return selected[:limit]


def add_selected(out: Path, selected):
    img_out = out / "train" / "images"
    mask_out = out / "train" / "masks"
    for row in selected:
        stem = f"s4mix_{row['split']}_{row['image'].stem}"
        shutil.copy2(row["image"], img_out / f"{stem}.jpg")
        shutil.copy2(row["mask"], mask_out / f"{stem}.png")


def hist(root: Path, split: str):
    counts = {0: 0, 1: 0, 2: 0}
    for p in (root / split / "masks").glob("*.png"):
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        vals, nums = np.unique(mask, return_counts=True)
        for v, n in zip(vals.tolist(), nums.tolist()):
            counts[int(v)] = counts.get(int(v), 0) + int(n)
    total = sum(counts.values()) or 1
    return {
        "pixels": counts,
        "percent": {str(k): round(v * 100.0 / total, 4) for k, v in counts.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=r"D:\mask\baseline_runs\strategy_patch_dataset_cs_newdata_hn_reviewed")
    parser.add_argument("--s4", default=r"D:\mask\baseline_runs\shujuji4_patch_tmp")
    parser.add_argument("--out", default=r"D:\mask\baseline_runs\strategy_patch_dataset_cs_newdata_hn_reviewed_s4mix300")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    base = Path(args.base)
    s4 = Path(args.s4)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")

    rng = random.Random(args.seed)
    copy_dataset(base, out)
    rows = collect_candidates(s4)
    selected = sample_balanced(rows, args.limit, rng)
    add_selected(out, selected)

    summary = {
        "base": str(base),
        "s4": str(s4),
        "out": str(out),
        "limit": args.limit,
        "selected": len(selected),
        "selected_avg": {
            "crack": round(sum(r["crack"] for r in selected) / max(1, len(selected)), 4),
            "spall": round(sum(r["spall"] for r in selected) / max(1, len(selected)), 4),
            "damage": round(sum(r["damage"] for r in selected) / max(1, len(selected)), 4),
        },
        "train_images": len(list((out / "train" / "images").glob("*.jpg"))),
        "val_images": len(list((out / "val" / "images").glob("*.jpg"))),
        "train_hist": hist(out, "train"),
        "val_hist": hist(out, "val"),
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
