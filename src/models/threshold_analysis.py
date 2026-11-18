"""Phase 8, Step 3: image-level recall vs confidence threshold T (VAL set only).

Frozen rule (set before looking at numbers):
  - T grid 0.05 .. 0.95, step 0.05, same for both models
  - strict recall = image has >= 1 detection with conf >= T, same class as a GT box,
    and IoU >= 0.5 with that GT box
  - choose the highest T with strict recall >= 0.95 (>= 137 of 144 images)
  - fallback: T = 0.05 if no T reaches the target
Decision-rule recall (any detection with conf >= T) is computed as a sanity check.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.utils.config import load_config

cfg = load_config()
CLASSES = cfg["data"]["classes"]
YOLO = Path(cfg["data"]["yolo_dir"])
DET = pd.read_csv("reports/exports/val_detections.csv")
GT_CSV = pd.read_csv("reports/exports/val_ground_truth.csv")
GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
TARGET, IOU_MIN = 0.95, 0.5
N = len(GT_CSV)
MIN_IMAGES = int(np.ceil(TARGET * N))          # 137 for 144 images


def load_gt(stem):
    w, h = Image.open(YOLO / "images" / "val" / f"{stem}.jpg").size
    boxes = []
    for line in (YOLO / "labels" / "val" / f"{stem}.txt").read_text().splitlines():
        if not line.strip():
            continue
        c, xc, yc, bw, bh = line.split()
        xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
        boxes.append((CLASSES[int(c)], xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2))
    return boxes


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


GT = {s: load_gt(s) for s in GT_CSV["image"]}
print(f"Images: {N} | GT boxes: {sum(len(v) for v in GT.values())} | need >= {MIN_IMAGES} images for recall >= {TARGET}")

results, curves, missed_lists = {}, {}, {}
for tag, d in DET.groupby("model"):
    # for every image: highest conf of ANY detection, and of a detection that matches a GT box
    any_conf = d.groupby("image")["conf"].max().reindex(GT_CSV["image"]).fillna(0.0)
    match_conf = {s: 0.0 for s in GT_CSV["image"]}
    for r in d.itertuples():
        for g in GT[r.image]:
            if g[0] == r.cls and iou((r.x1, r.y1, r.x2, r.y2), g[1:]) >= IOU_MIN:
                match_conf[r.image] = max(match_conf[r.image], r.conf)
                break
    match_conf = pd.Series(match_conf)
    rows = []
    for t in GRID:
        rows.append({"T": t,
                     "strict_images": int((match_conf >= t).sum()),
                     "loose_images": int((any_conf >= t).sum())})
    c = pd.DataFrame(rows)
    c["strict_recall"] = (c["strict_images"] / N).round(3)
    c["loose_recall"] = (c["loose_images"] / N).round(3)
    curves[tag] = c

    ok = c[c["strict_images"] >= MIN_IMAGES]
    chosen = float(ok["T"].max()) if len(ok) else float(GRID[0])
    results[tag] = (chosen, len(ok) > 0)
    miss = [s for s in GT_CSV["image"] if match_conf[s] < chosen]
    missed_lists[tag] = GT_CSV[GT_CSV["image"].isin(miss)].assign(model=tag, T=chosen)

for tag, c in curves.items():
    chosen, reached = results[tag]
    print(f"\n===== {tag} =====")
    print(c.to_string(index=False))
    print(f"Selected T (strict recall >= {TARGET}): {chosen}"
          + ("" if reached else "  [TARGET NOT REACHED: fallback T]"))
    m = missed_lists[tag]
    print(f"Images missed at T={chosen} (strict): {len(m)}")
    print(m["gt_classes"].value_counts().to_string())

Path("reports/exports").mkdir(parents=True, exist_ok=True)
pd.concat(missed_lists.values()).to_csv("reports/exports/val_missed_at_selected_T.csv", index=False)
pd.concat([c.assign(model=t) for t, c in curves.items()]).to_csv("reports/exports/val_recall_vs_T.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
for tag, c in curves.items():
    ax.plot(c["T"], c["strict_recall"], marker="o", label=f"{tag} strict")
    ax.plot(c["T"], c["loose_recall"], linestyle="--", label=f"{tag} any detection")
ax.axhline(TARGET, color="red", linewidth=1, label="target 0.95")
ax.set_xlabel("Confidence threshold T")
ax.set_ylabel("Image-level recall (val, 144 images)")
ax.set_title("Image-level recall vs T (seed 42, val set)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
Path("reports/figures").mkdir(parents=True, exist_ok=True)
fig.savefig("reports/figures/image_recall_vs_T.png", dpi=150)
print("\nSaved: reports/figures/image_recall_vs_T.png and CSVs in reports/exports/")