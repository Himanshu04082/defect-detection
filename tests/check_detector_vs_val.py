"""Check: DefectDetector output on VAL images == saved val detections (v8n, conf >= T)."""
import statistics as st
from pathlib import Path

import pandas as pd

from src.inference.detector import DefectDetector

det = DefectDetector()
T = det.threshold
saved = pd.read_csv("reports/exports/val_detections.csv")
saved = saved[(saved["model"] == "v8n") & (saved["conf"] >= T)]
expected = {k: sorted(v) for k, v in saved.groupby("image")["conf"]}

images = sorted(Path("data/yolo/images/val").glob("*.jpg"))
mismatch, times, decisions, total = [], [], {"Good": 0, "Defective": 0}, 0
for i, p in enumerate(images):
    res = det.inspect_path(p)
    decisions[res.decision] += 1
    total += res.n_defects
    if i >= 5:                      # skip the first calls (warm-up)
        times.append(res.inference_ms)
    got = sorted(d.confidence for d in res.detections)
    exp = expected.get(p.stem, [])
    same = len(got) == len(exp) and all(abs(a - b) < 0.001 for a, b in zip(got, exp))
    if not same:
        mismatch.append((p.stem, len(got), len(exp)))

print(f"Threshold: {T} | images: {len(images)}")
print("Decisions:", decisions, "| total detections:", total)
print("Images where module output differs from saved detections:", len(mismatch))
for m in mismatch[:10]:
    print("  ", m, "(image, module count, saved count)")
times.sort()
print(f"Latency after warm-up: median {st.median(times):.1f} ms | "
      f"p95 {times[int(0.95 * len(times))]:.1f} ms | n={len(times)}")