import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_mask_baselines import PatchDataset, class_weights_from_arg, evaluate, make_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--loss-mode", default="focal_tversky")
    parser.add_argument("--class-weights", default="0.35,2.5,1.3")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt.get("name") or ckpt.get("model") or "dual_lraspp_aux_mnv3"
    model = make_model(model_name).to(device)
    model.load_state_dict(ckpt["model"])

    val_ds = PatchDataset(args.data, "val", augment=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    ce_weight = class_weights_from_arg(args.class_weights, device)
    metrics = evaluate(model, val_loader, device, loss_mode=args.loss_mode, ce_weight=ce_weight)
    metrics["model"] = model_name
    metrics["checkpoint"] = str(Path(args.checkpoint))
    metrics["data"] = str(Path(args.data))
    metrics["val_items"] = len(val_ds)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
