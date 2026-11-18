"""Phase 8, Step 2: save raw detections on the VAL set for both seed-42 models.

Detections are saved with a very low confidence floor so that any threshold T
above the floor can be analysed later without re-running the models.
The TEST set is not touched here.
"""
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from src.utils.config import load_config

cfg = load_config()
IMGSZ = cfg["preprocessing"]["imgsz"]
CLASSES = cfg["data"]["classes"]
VAL_IMG = Path(cfg["data"]["yolo_dir"]) / "images" / "val"
VAL_LBL = Path(cfg["data"]["yolo_dir"]) / "labels" / "val"
MODELS = {"v8n": "models/final_v8n_best.pt", "v8s": "models/final_v8s_best.pt"}
CONF_FLOOR = 0.001
OUT_DET = Path("reports/exports/val_detections.csv")
OUT_GT = Path("reports/exports/val_ground_truth.csv")

images = sorted(VAL_IMG.glob("*.jpg"))

# ground truth per image (classes and number of boxes, straight from the label files)
gt = []
for p in images:
    lines = [l.split() for l in (VAL_LBL / f"{p.stem}.txt").read_text().splitlines() if l.strip()]
    ids = sorted({int(l[0]) for l in lines})
    gt.append({"image": p.stem, "n_gt_boxes": len(lines),
               "gt_classes": "+".join(CLASSES[i] for i in ids)})
OUT_GT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(gt).to_csv(OUT_GT, index=False)

rows = []
for tag, weights in MODELS.items():
    model = YOLO(weights)
    for p in images:
        r = model.predict(str(p), imgsz=IMGSZ, conf=CONF_FLOOR, device="cpu", verbose=False)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            rows.append({"model": tag, "image": p.stem, "cls": model.names[int(b.cls)],
                         "conf": round(float(b.conf), 4),
                         "x1": round(x1, 1), "y1": round(y1, 1),
                         "x2": round(x2, 1), "y2": round(y2, 1)})
    print(tag, "done")

det = pd.DataFrame(rows)
det.to_csv(OUT_DET, index=False)

print("\nVal images:", len(images), "| GT boxes in label files:", sum(g["n_gt_boxes"] for g in gt))
for tag in MODELS:
    d = det[det["model"] == tag]
    print(f"{tag}: detections={len(d)}, images with >=1 detection at floor {CONF_FLOOR}: {d['image'].nunique()}")
print("Saved:", OUT_DET, "and", OUT_GT)