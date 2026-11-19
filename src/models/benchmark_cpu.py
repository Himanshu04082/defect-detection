"""Phase 8, Step 4: CPU speed benchmark (batch 1, imgsz 640, warm-up, median latency).

Same protocol for both models: images are read from disk BEFORE timing; the timed part is
model.predict() on a numpy array (preprocess + inference + postprocess).
"""
import platform
import statistics as st
import time
from pathlib import Path

import cv2
import pandas as pd
import torch
from ultralytics import YOLO

from src.utils.config import load_config

cfg = load_config()
IMGSZ = cfg["preprocessing"]["imgsz"]
VAL = sorted((Path(cfg["data"]["yolo_dir"]) / "images" / "val").glob("*.jpg"))
MODELS = {"v8n": "models/final_v8n_best.pt", "v8s": "models/final_v8s_best.pt"}
T = 0.15          # frozen threshold from Step 3
WARMUP, ROUNDS = 10, 2

images = [cv2.imread(str(p)) for p in VAL]
print("Python", platform.python_version(), "| torch", torch.__version__,
      "| torch threads", torch.get_num_threads(), "| CPU", platform.processor())
print("Images:", len(images), "| imgsz", IMGSZ, "| conf", T, "| device cpu | batch 1\n")

loaded = {tag: YOLO(w) for tag, w in MODELS.items()}
rows = []
for rnd in range(1, ROUNDS + 1):
    for tag, model in loaded.items():
        for img in images[:WARMUP]:
            model.predict(img, imgsz=IMGSZ, conf=T, device="cpu", verbose=False)
        for img in images:
            t0 = time.perf_counter()
            r = model.predict(img, imgsz=IMGSZ, conf=T, device="cpu", verbose=False)[0]
            total = (time.perf_counter() - t0) * 1000
            rows.append({"model": tag, "round": rnd, "total_ms": total,
                         "inference_ms": r.speed["inference"]})
        print(f"round {rnd} {tag} done")

df = pd.DataFrame(rows)
summary = df.groupby(["model", "round"]).agg(
    median_total_ms=("total_ms", "median"),
    p95_total_ms=("total_ms", lambda s: s.quantile(0.95)),
    median_inference_ms=("inference_ms", "median"),
).round(1)
summary["fps_from_median"] = (1000 / summary["median_total_ms"]).round(1)
print("\n", summary.to_string())

Path("reports/exports").mkdir(parents=True, exist_ok=True)
summary.to_csv("reports/exports/cpu_speed.csv")
print("\nSaved: reports/exports/cpu_speed.csv")