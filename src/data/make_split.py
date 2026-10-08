"""Step 4D: create train/val/test assignment for Option 2 images.

Mapping (document this in the report):
  original NEU-DET 'validation' (Option 2)  -> project 'test'  (locked)
  original NEU-DET 'train' (Option 2), 80%   -> project 'train'
  original NEU-DET 'train' (Option 2), 20%   -> project 'val'
Only a CSV is written here. No images are copied.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SELECTED = {"inclusion", "scratches", "rolled-in_scale"}
SEED = 42
VAL_FRACTION = 0.20
BOX_CSV = Path("reports/exports/box_table.csv")
OUT = Path("data/yolo/split_assignment.csv")


def option2_images(df, orig_split):
    d = df[df["split"] == orig_split]
    classes = d.groupby("image")["cls"].apply(frozenset)
    classes = classes[classes.apply(lambda s: s <= SELECTED)]
    return pd.DataFrame({
        "image": classes.index,
        "orig_split": orig_split,
        "stratum": classes.apply(lambda s: "+".join(sorted(s))).values,
    })


def main():
    df = pd.read_csv(BOX_CSV)
    train_pool = option2_images(df, "train")
    test = option2_images(df, "validation")
    test["project_split"] = "test"

    tr, va = train_test_split(
        train_pool, test_size=VAL_FRACTION,
        stratify=train_pool["stratum"], random_state=SEED,
    )
    tr = tr.assign(project_split="train")
    va = va.assign(project_split="val")

    out = pd.concat([tr, va, test]).sort_values("image").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("Saved:", OUT, "| rows:", len(out))
    print("\nImages per project split:")
    print(out["project_split"].value_counts().to_string())
    print("\nStrata per project split:")
    print(pd.crosstab(out["stratum"], out["project_split"]).to_string())

    boxes = df.merge(out, on="image")
    boxes = boxes[boxes["cls"].isin(SELECTED)]
    print("\nBoxes per class per project split:")
    print(pd.crosstab(boxes["cls"], boxes["project_split"]).to_string())

    print("\nOverlap checks (must all be 0):")
    s = {k: set(g["image"]) for k, g in out.groupby("project_split")}
    print("train&val :", len(s["train"] & s["val"]))
    print("train&test:", len(s["train"] & s["test"]))
    print("val&test  :", len(s["val"] & s["test"]))
    print("duplicate image names:", out["image"].duplicated().sum())


if __name__ == "__main__":
    main()