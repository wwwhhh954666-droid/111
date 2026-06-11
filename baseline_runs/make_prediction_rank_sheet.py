import argparse
import csv
import shutil
from pathlib import Path

import cv2


def draw_label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (255, 255, 255), -1)
    cv2.putText(out, text[:95], (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def make_sheet(paths, labels, out_path, thumb_w=560, cols=2):
    thumbs = []
    for p, label in zip(paths, labels):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        thumb_h = max(1, int(h * thumb_w / w))
        img = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        thumbs.append(draw_label(img, label))
    if not thumbs:
        return
    rows = []
    for i in range(0, len(thumbs), cols):
        row = thumbs[i : i + cols]
        max_h = max(x.shape[0] for x in row)
        padded = []
        for x in row:
            if x.shape[0] < max_h:
                x = cv2.copyMakeBorder(x, 0, max_h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            padded.append(x)
        while len(padded) < cols:
            padded.append(255 * cv2.UMat(max_h, thumb_w, cv2.CV_8UC3).get())
        rows.append(cv2.hconcat(padded))
    cv2.imwrite(str(out_path), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--overlays", required=True)
    parser.add_argument("--source", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--metric", default="spall_percent", choices=["crack_percent", "spall_percent", "damage_percent"])
    parser.add_argument("--only-no-json", action="store_true")
    args = parser.parse_args()

    summary = Path(args.summary)
    overlay_dir = Path(args.overlays)
    source = Path(args.source) if args.source else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "items").mkdir(exist_ok=True)

    rows = []
    with summary.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = Path(row["image"]).stem
            if args.only_no_json and source and (source / f"{stem}.json").exists():
                continue
            crack = float(row["crack_percent"])
            spall = float(row["spall_percent"])
            row["damage_percent"] = crack + spall
            row["_stem"] = stem
            row["_metric"] = float(row[args.metric]) if args.metric != "damage_percent" else row["damage_percent"]
            rows.append(row)
    rows.sort(key=lambda r: r["_metric"], reverse=True)
    rows = rows[: args.top]

    paths = []
    labels = []
    with (out / "ranked_items.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["rank", "image", "crack_percent", "spall_percent", "damage_percent", "overlay"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            overlay = overlay_dir / f"{row['_stem']}_overlay.jpg"
            if not overlay.exists():
                continue
            copied = out / "items" / f"{i:02d}_{overlay.name}"
            shutil.copy2(overlay, copied)
            paths.append(copied)
            labels.append(
                f"{i:02d} {row['image']} crack={float(row['crack_percent']):.3f}% "
                f"spall={float(row['spall_percent']):.3f}% damage={float(row['damage_percent']):.3f}%"
            )
            writer.writerow(
                {
                    "rank": i,
                    "image": row["image"],
                    "crack_percent": row["crack_percent"],
                    "spall_percent": row["spall_percent"],
                    "damage_percent": f"{row['damage_percent']:.4f}",
                    "overlay": str(copied),
                }
            )
    make_sheet(paths, labels, out / "rank_sheet.jpg")
    print(f"out={out}")


if __name__ == "__main__":
    main()
