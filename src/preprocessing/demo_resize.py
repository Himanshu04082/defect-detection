"""Show original 200x200 images next to their 640x640 resized version."""
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import load_config

cfg = load_config()
SIZE = cfg["preprocessing"]["imgsz"]
IMG_DIR = Path(cfg["data"]["yolo_dir"]) / "images" / "train"
OUT = Path("reports/figures/preprocess_demo.png")

# One example per class. Change a name if the script says "missing image".
STEMS = ["inclusion_1", "scratches_1", "rolled-in_scale_1"]

rows = []
for stem in STEMS:
    img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
    assert img is not None, f"missing image in train: {stem}"
    big = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    # original (unscaled) on a white canvas, same height as the resized image
    canvas = np.full((SIZE, 220, 3), 255, np.uint8)
    canvas[10:10 + img.shape[0], 10:10 + img.shape[1]] = img
    cv2.putText(canvas, f"{img.shape[1]}x{img.shape[0]}", (10, 235),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    rows.append(np.hstack([canvas, big]))
    print(stem, "original", img.shape, "->", big.shape)

OUT.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT), np.vstack(rows))
print("Saved:", OUT)