"""Count images under two selection rules for the 3 frozen classes."""
import pandas as pd

SELECTED = {"inclusion", "scratches", "rolled-in_scale"}
df = pd.read_csv("reports/exports/box_table.csv")

for split in ("train", "validation"):
    d = df[df["split"] == split]
    per_image = d.groupby("image")["cls"].apply(set)
    opt1 = per_image[per_image.apply(lambda s: len(s & SELECTED) > 0)]
    opt2 = per_image[per_image.apply(lambda s: s <= SELECTED)]
    print(f"\n== {split} (total images {len(per_image)}) ==")
    print("Option 1 (>=1 selected class):", len(opt1))
    print("Option 2 (only selected classes):", len(opt2))
    print("Boxes kept, Option 2, per class:")
    kept = d[d["image"].isin(opt2.index) & d["cls"].isin(SELECTED)]
    print(kept["cls"].value_counts().to_string())