import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from train_mask_baselines import make_model


def collect_images(roots):
    paths = []
    seen = set()
    for root in [Path(p) for p in roots]:
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            for path in sorted(root.glob(pattern)):
                key = path.name.lower()
                if key in seen:
                    continue
                seen.add(key)
                paths.append(path)
    return paths


def load_batch(paths, device):
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        images.append(tensor)
    batch = torch.stack(images).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (batch - mean) / std


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "mean": round(float(arr.mean()), 6),
        "median": round(float(np.median(arr)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "max": round(float(arr.max()), 6),
        "gt_0_5_pct_count": int((arr > 0.5).sum()),
        "gt_1_pct_count": int((arr > 1.0).sum()),
    }


def evaluate_checkpoint(checkpoint, image_paths, args, device):
    model = make_model(args.model).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    rows = []
    with torch.no_grad():
        for start in range(0, len(image_paths), args.batch):
            batch_paths = image_paths[start : start + args.batch]
            batch = load_batch(batch_paths, device)
            logits = model(batch)
            pred_logits = logits["out"] if isinstance(logits, dict) else logits
            pred = pred_logits.argmax(1).detach().cpu().numpy()
            for path, mask in zip(batch_paths, pred):
                total = float(mask.size)
                crack = float((mask == 1).sum()) * 100.0 / total
                spall = float((mask == 2).sum()) * 100.0 / total
                rows.append({
                    "image": str(path),
                    "crack_percent": round(crack, 6),
                    "spall_percent": round(spall, 6),
                    "damage_percent": round(crack + spall, 6),
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--negative-dirs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="dual_lraspp_aux_fcf_mnv3")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    image_paths = collect_images(args.negative_dirs)
    if not image_paths:
        raise SystemExit("no negative images found")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary = {
        "negative_image_count": len(image_paths),
        "negative_dirs": args.negative_dirs,
        "checkpoints": {},
    }
    for checkpoint in args.checkpoints:
        ckpt_path = Path(checkpoint)
        name = ckpt_path.parent.parent.name if ckpt_path.name in {"best.pt", "last.pt"} else ckpt_path.stem
        rows = evaluate_checkpoint(ckpt_path, image_paths, args, device)
        with (out_dir / f"{name}_per_image.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "crack_percent", "spall_percent", "damage_percent"])
            writer.writeheader()
            writer.writerows(rows)
        summary["checkpoints"][name] = {
            "checkpoint": str(ckpt_path),
            "crack": summarize([row["crack_percent"] for row in rows]),
            "spall": summarize([row["spall_percent"] for row in rows]),
            "damage": summarize([row["damage_percent"] for row in rows]),
        }

    (out_dir / "negative_fp_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
