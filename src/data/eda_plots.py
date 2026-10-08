"""EDA plots + class co-occurrence on the cleaned NEU-DET copy."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV = Path("reports/exports/box_table.csv")
FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
tr = df[df["split"] == "train"]
classes = sorted(df["cls"].unique())

# 1) Boxes per class
counts = tr["cls"].value_counts().reindex(classes)
ax = counts.plot(kind="bar", figsize=(8, 4), color="steelblue")
ax.set_title("Train: boxes per class")
ax.set_ylabel("Number of boxes")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIG / "eda_boxes_per_class.png", dpi=150)
plt.close()

# 2) Box area distribution per class
fig, ax = plt.subplots(figsize=(9, 4))
ax.boxplot([tr[tr["cls"] == c]["area_pct"] for c in classes], tick_labels=classes)
ax.set_title("Train: box area (% of image) per class")
ax.set_ylabel("Area %")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIG / "eda_box_area.png", dpi=150)
plt.close()

# 3) Co-occurrence: how many train images contain both class A and class B
present = tr.groupby(["image", "cls"]).size().unstack(fill_value=0) > 0
co = present.astype(int).T.dot(present.astype(int))
print("--- Co-occurrence (number of train images containing both classes) ---")
print(co)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(co.values, cmap="Blues")
ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
ax.set_yticks(range(len(classes)), classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, co.values[i, j], ha="center", va="center", fontsize=8)
ax.set_title("Train: class co-occurrence (images)")
plt.colorbar(im)
plt.tight_layout()
plt.savefig(FIG / "eda_cooccurrence.png", dpi=150)
plt.close()
print("Saved 3 figures in", FIG)