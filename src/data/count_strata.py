"""Count class-set strata for Option 2 images (no split is done here).

Option 2 = images whose XML contains ONLY the 3 selected classes.
"""
import pandas as pd

SELECTED = {"inclusion", "scratches", "rolled-in_scale"}
df = pd.read_csv("reports/exports/box_table.csv")

for split in ("train", "validation"):
    d = df[df["split"] == split]
    per_image = d.groupby("image")["cls"].apply(frozenset)
    opt2 = per_image[per_image.apply(lambda s: s <= SELECTED)]
    strata = opt2.apply(lambda s: "+".join(sorted(s)))
    print(f"\n== {split}: Option 2 images = {len(opt2)} ==")
    print(strata.value_counts().to_string())