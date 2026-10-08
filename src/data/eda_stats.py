"""EDA statistics on the cleaned NEU-DET copy (boxes per class, box sizes)."""
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path("data/processed/neu-det-clean")
OUT = Path("reports/exports/box_table.csv")


def load_boxes():
    rows = []
    for split in ("train", "validation"):
        for xml_path in sorted((ROOT / split / "annotations").glob("*.xml")):
            root = ET.parse(xml_path).getroot()
            w = int(root.findtext("size/width"))
            h = int(root.findtext("size/height"))
            for obj in root.findall("object"):
                b = obj.find("bndbox")
                x1, y1 = int(b.findtext("xmin")), int(b.findtext("ymin"))
                x2, y2 = int(b.findtext("xmax")), int(b.findtext("ymax"))
                rows.append({
                    "split": split,
                    "image": xml_path.stem,
                    "cls": obj.findtext("name"),
                    "bw": x2 - x1,
                    "bh": y2 - y1,
                    "area_pct": 100 * (x2 - x1) * (y2 - y1) / (w * h),
                    "truncated": int(obj.findtext("truncated") or 0),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    df = load_boxes()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print("Total boxes:", len(df), "| images:", df["image"].nunique())

    print("\n--- Boxes per class (train / validation) ---")
    print(pd.crosstab(df["cls"], df["split"], margins=True))

    print("\n--- Images containing each class (train / validation) ---")
    imgs = df.drop_duplicates(["split", "image", "cls"])
    print(pd.crosstab(imgs["cls"], imgs["split"]))

    print("\n--- Box area as % of image (train only) ---")
    tr = df[df["split"] == "train"]
    print(tr.groupby("cls")["area_pct"].describe().round(1)[["count", "mean", "min", "50%", "max"]])

    print("\n--- Box width / height in pixels (train, median) ---")
    print(tr.groupby("cls")[["bw", "bh"]].median())

    print("\n--- Share of small boxes (area < 5% of image) per class, train ---")
    print((tr.assign(small=tr["area_pct"] < 5).groupby("cls")["small"].mean() * 100).round(1))