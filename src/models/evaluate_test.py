"""Phase 8, Step 6: ONE-TIME evaluation of the frozen model on the locked TEST set.

Frozen before this run (see docs/phase8_protocol.md, tag phase8-model-frozen):
  model  = models/final_v8n_best.pt (SHA256 checked below)
  T      = 0.15
  strict match = same class and IoU >= 0.5 with a ground-truth box
Nothing here may be used to change the model or T.
"""
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image
from ultralytics import YOLO

from src.utils.config import load_config

cfg = load_config()
CLASSES = cfg["data"]["classes"]
IMGSZ = cfg["preprocessing"]["imgsz"]
YOLO_DIR = Path(cfg["data"]["yolo_dir"])
WEIGHTS = Path("models/final_v8n_best.pt")
FROZEN_SHA256 = "94A2E409EDC19B4DE670423614F66C70DE41CAC1696D1FBFD0C89A55DF18FEAF"
T, IOU_MIN = 0.15, 0.5
SENTINEL = Path("reports/exports/TEST_RUN_DONE.txt")
OUT = Path("reports/exports")

if SENTINEL.exists():
    sys.exit("The test set was already evaluated once. Refusing to run again.")
sha = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest().upper()
if sha != FROZEN_SHA256:
    sys.exit(f"Weights do not match the frozen SHA256: {sha}")
print("Weights SHA256 matches the frozen decision.")


def load_gt(stem):
    w, h = Image.open(YOLO_DIR / "images" / "test" / f"{stem}.jpg").size
    boxes = []
    for line in (YOLO_DIR / "labels" / "test" / f"{stem}.txt").read_text().splitlines():
        if line.strip():
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


images = sorted((YOLO_DIR / "images" / "test").glob("*.jpg"))
model = YOLO(str(WEIGHTS))
print(f"Test images: {len(images)} | frozen T = {T} | IoU >= {IOU_MIN}")

# ---------- 1) box-level metrics (standard Ultralytics evaluation on the test split) ----------
m = model.val(data=cfg["data"]["dataset_yaml"], split="test", imgsz=IMGSZ,
              device="cpu", verbose=False, plots=False)
print("\n=== Box-level metrics, TEST (Ultralytics val) ===")
print(f"all: P={m.box.mp:.3f} R={m.box.mr:.3f} mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f}")
for i, c in enumerate(m.box.ap_class_index):
    print(f"{CLASSES[int(c)]:16s}: P={m.box.p[i]:.3f} R={m.box.r[i]:.3f} "
          f"mAP50={m.box.ap50[i]:.3f} mAP50-95={m.box.ap[i]:.3f}")

# ---------- 2) image-level and box-level analysis at the frozen T ----------
img_rows, tp, fp, gt_n = [], {c: 0 for c in CLASSES}, {c: 0 for c in CLASSES}, {c: 0 for c in CLASSES}
for p in images:
    gt = load_gt(p.stem)
    r = model.predict(str(p), imgsz=IMGSZ, conf=T, device="cpu", verbose=False)[0]
    dets = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        dets.append((model.names[int(b.cls)], float(b.conf), (x1, y1, x2, y2)))
    dets.sort(key=lambda d: -d[1])

    used, strict_hit = set(), False
    for cls, conf, box in dets:                      # greedy one-to-one matching
        best, best_j = 0.0, None
        for j, g in enumerate(gt):
            if g[0] == cls and j not in used:
                v = iou(box, g[1:])
                if v > best:
                    best, best_j = v, j
        if best >= IOU_MIN:
            used.add(best_j)
            tp[cls] += 1
            strict_hit = True
        else:
            fp[cls] += 1
    for g in gt:
        gt_n[g[0]] += 1
    img_rows.append({"image": p.stem, "gt_classes": "+".join(sorted({g[0] for g in gt})),
                     "n_gt": len(gt), "n_det": len(dets),
                     "strict_hit": strict_hit, "loose_hit": len(dets) > 0})

df = pd.DataFrame(img_rows)
N = len(df)
print(f"\n=== Image-level at T = {T} (all {N} test images are defective) ===")
print(f"Strict recall (same class, IoU>={IOU_MIN}): {df.strict_hit.sum()}/{N} = {df.strict_hit.mean():.3f}")
print(f"Decision-rule recall (any detection)      : {df.loose_hit.sum()}/{N} = {df.loose_hit.mean():.3f}")
print("Frozen target was strict recall >= 0.95 (>= 171 of 180 images).")
print("\nBy image stratum (strict hits / images):")
print(df.groupby("gt_classes")["strict_hit"].agg(["sum", "count"]).to_string())

print(f"\n=== Box-level at T = {T} ===")
tot_tp = tot_fp = tot_gt = 0
for c in CLASSES:
    fn = gt_n[c] - tp[c]
    prec = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
    rec = tp[c] / gt_n[c] if gt_n[c] else 0.0
    print(f"{c:16s}: GT={gt_n[c]:3d} TP={tp[c]:3d} FP={fp[c]:3d} FN={fn:3d} precision={prec:.3f} recall={rec:.3f}")
    tot_tp, tot_fp, tot_gt = tot_tp + tp[c], tot_fp + fp[c], tot_gt + gt_n[c]
print(f"{'all':16s}: GT={tot_gt:3d} TP={tot_tp:3d} FP={tot_fp:3d} FN={tot_gt - tot_tp:3d}")

df.to_csv(OUT / "test_image_level.csv", index=False)
df[~df.strict_hit].to_csv(OUT / "test_missed_images.csv", index=False)
SENTINEL.write_text(f"Test evaluated once on {datetime.now().isoformat()} with SHA256 {sha}, T={T}\n")
print("\nMissed images (strict):", sorted(df[~df.strict_hit]["image"]))
print("Saved CSVs and sentinel in", OUT)