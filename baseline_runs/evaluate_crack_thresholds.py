import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_crack_assist import CrackOnlyPatchDataset, evaluate, make_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--class-weights", default="0.25,2.0")
    parser.add_argument(
        "--thresholds",
        default="0.72,0.78,0.82,0.86,0.88,0.90,0.92,0.94,0.96",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CrackOnlyPatchDataset(args.data, "val", augment=False)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = make_model().to(device)
    model.load_state_dict(ckpt["model"])
    weights = torch.tensor(
        [float(x) for x in args.class_weights.split(",")],
        dtype=torch.float32,
        device=device,
    )

    rows = []
    for threshold in [float(x) for x in args.thresholds.split(",")]:
        metrics = evaluate(model, loader, device, threshold, weights)
        rows.append(metrics)
        print(
            f"{threshold:.2f},"
            f"precision={metrics['precision']:.6f},"
            f"recall={metrics['recall']:.6f},"
            f"f1={metrics['f1']:.6f},"
            f"iou={metrics['iou']:.6f},"
            f"fp={metrics['fp']}"
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
