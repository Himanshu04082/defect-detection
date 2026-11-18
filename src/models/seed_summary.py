"""Summarise YOLOv8n vs YOLOv8s over 3 seeds (validation set, 50 epochs).

Numbers are copied from the Colab validation outputs:
  seed 42  -> final validation table printed at the end of training
  seed 7, 123 -> summary table printed by the seeds cell
Columns: P, R, mAP50, mAP50-95, recall for inclusion / scratches / rolled-in_scale
"""
from pathlib import Path

import pandas as pd

COLS = ["model", "seed", "P", "R", "mAP50", "mAP50_95",
        "rec_inclusion", "rec_scratches", "rec_rolled_in_scale"]
ROWS = [
    ["v8n", 42, 0.777, 0.771, 0.818, 0.448, 0.806, 0.928, 0.578],
    ["v8n", 7,  0.813, 0.720, 0.818, 0.434, 0.768, 0.892, 0.500],
    ["v8n", 123, 0.785, 0.734, 0.811, 0.439, 0.791, 0.892, 0.520],
    ["v8s", 42, 0.761, 0.761, 0.804, 0.449, 0.853, 0.940, 0.490],
    ["v8s", 7,  0.805, 0.714, 0.802, 0.442, 0.781, 0.880, 0.480],
    ["v8s", 123, 0.744, 0.762, 0.800, 0.427, 0.854, 0.892, 0.539],
]
OUT = Path("reports/training/seed_summary.csv")

df = pd.DataFrame(ROWS, columns=COLS)
metrics = COLS[2:]
summary = df.groupby("model")[metrics].agg(["mean", "std", "min", "max"]).round(3)

pd.set_option("display.width", 200)
print(df.to_string(index=False))
print("\nMean / std (sample, n=3) / min / max per model:")
print(summary.T.to_string())

OUT.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUT)
print("\nSaved:", OUT)