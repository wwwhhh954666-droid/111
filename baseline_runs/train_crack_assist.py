import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset


MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CrackOnlyPatchDataset(Dataset):
    def __init__(self, root, split, augment=False):
        self.root = Path(root) / split
        self.images = sorted((self.root / "images").glob("*.jpg"))
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.root / "masks" / f"{img_path.stem}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask == 1).astype(np.int64)

        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[::-1])
                mask = np.ascontiguousarray(mask[::-1])
            if random.random() < 0.45:
                gain = random.uniform(0.75, 1.25)
                bias = random.uniform(-22, 22)
                img = np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
            if random.random() < 0.25:
                sigma = random.uniform(0.0, 8.0)
                noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        img = (img - MEAN) / STD
        return img, torch.from_numpy(mask)


def make_model():
    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=2
    )
    return model


def focal_loss(logits, target, alpha=None, gamma=2.0):
    ce = F.cross_entropy(logits, target, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    return (((1.0 - pt) ** gamma) * ce).mean()


def tversky_loss(logits, target, alpha=0.8, beta=0.2):
    probs = torch.softmax(logits, dim=1)[:, 1]
    target = target.float()
    dims = (0, 1, 2)
    tp = torch.sum(probs * target, dims)
    fp = torch.sum(probs * (1.0 - target), dims)
    fn = torch.sum((1.0 - probs) * target, dims)
    score = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    return 1.0 - score


def loss_fn(logits, target, weight):
    return focal_loss(logits, target, weight) + tversky_loss(logits, target)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.72, weight=None):
    model.eval()
    tp = fp = fn = tn = 0
    total_loss = 0.0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        out = model(img)["out"]
        total_loss += float(loss_fn(out, mask, weight).item())
        prob = torch.softmax(out, dim=1)[:, 1]
        pred = prob >= threshold
        target = mask == 1
        tp += int((pred & target).sum().item())
        fp += int((pred & ~target).sum().item())
        fn += int((~pred & target).sum().item())
        tn += int((~pred & ~target).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    iou = tp / max(1, tp + fp + fn)
    return {
        "val_loss": total_loss / max(1, len(loader)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": threshold,
    }


def save_preview(model, dataset, out_dir, device, threshold=0.72, max_items=8):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for i in range(min(max_items, len(dataset))):
        img, mask = dataset[i]
        with torch.no_grad():
            out = model(img.unsqueeze(0).to(device))["out"]
            prob = torch.softmax(out, dim=1)[0, 1].cpu().numpy()
        pred = (prob >= threshold).astype(np.uint8)
        denorm = img * STD + MEAN
        rgb = np.clip(denorm.numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        gt = rgb.copy()
        gt[mask.numpy() == 1] = (gt[mask.numpy() == 1] * 0.55 + np.array([255, 40, 40]) * 0.45).astype(np.uint8)
        pv = rgb.copy()
        pv[pred == 1] = (pv[pred == 1] * 0.55 + np.array([255, 40, 40]) * 0.45).astype(np.uint8)
        panel = np.concatenate([rgb, gt, pv], axis=1)
        cv2.imwrite(str(out_dir / f"preview_{i:02d}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=r"D:\mask\baseline_runs\strategy_patch_dataset_cs_newdata_hn_reviewed")
    parser.add_argument("--out", default=r"D:\mask\baseline_runs\experiments_crack_assist")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--class-weights", default="0.25,4.0")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--eval-initial", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CrackOnlyPatchDataset(args.data, "train", augment=True)
    val_ds = CrackOnlyPatchDataset(args.data, "val", augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    model = make_model().to(device)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    class_weights = torch.tensor([float(x) for x in args.class_weights.split(",")], dtype=torch.float32, device=device)
    best = {"score": -1.0}
    history = []

    if args.eval_initial:
        metrics = evaluate(model, val_loader, device, args.threshold, class_weights)
        metrics["epoch"] = 0
        metrics["train_loss"] = 0.0
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))
        score = metrics["precision"] * 0.7 + metrics["f1"] * 0.3
        best = dict(metrics)
        best["score"] = score
        torch.save({"model": model.state_dict(), "metrics": best, "name": "crack_assist_lraspp_mnv3"}, out_dir / "best.pt")
        save_preview(model, val_ds, out_dir / "previews", device, args.threshold)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(img)["out"]
                loss = loss_fn(out, mask, class_weights)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item())
        metrics = evaluate(model, val_loader, device, args.threshold, class_weights)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running / max(1, len(train_loader))
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))

        # For assisted annotation, false positives are expensive; rank precision first, then F1.
        score = metrics["precision"] * 0.7 + metrics["f1"] * 0.3
        if score > best["score"]:
            best = dict(metrics)
            best["score"] = score
            torch.save({"model": model.state_dict(), "metrics": best, "name": "crack_assist_lraspp_mnv3"}, out_dir / "best.pt")
            save_preview(model, val_ds, out_dir / "previews", device, args.threshold)

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    (out_dir / "best_metrics.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print("BEST")
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
