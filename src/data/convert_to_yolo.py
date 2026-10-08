"""Step 4E: convert Option 2 images to YOLO format + sanity checks.

Labels come from XML <name> (never from folder names).
Class ids: inclusion=0, scratches=1, rolled-in_scale=2.
Project 'test' = original NEU-DET 'validation' (locked).
"""
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

CLEAN = Path("data/processed/neu-det-clean")
YOLO = Path("data/yolo")
ASSIGN = YOLO / "split_assignment.csv"
CLASSES = ["inclusion", "scratches", "rolled-in_scale"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}


def find_image(orig_split, stem):
    matches = list((CLEAN / orig_split / "images").glob(f"*/{stem}.jpg"))
    assert len(matches) == 1, f"image lookup failed for {stem}: {matches}"
    return matches[0]


def parse_xml(orig_split, stem):
    root = ET.parse(CLEAN / orig_split / "annotations" / f"{stem}.xml").getroot()
    w = int(root.findtext("size/width"))
    h = int(root.findtext("size/height"))
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        assert name in CLASS_ID, f"unexpected class {name} in {stem}"
        b = obj.find("bndbox")
        x1, y1 = int(b.findtext("xmin")), int(b.findtext("ymin"))
        x2, y2 = int(b.findtext("xmax")), int(b.findtext("ymax"))
        boxes.append((CLASS_ID[name], x1, y1, x2, y2, w, h))
    return boxes


def main():
    df = pd.read_csv(ASSIGN)

    # rebuild output folders from scratch (reproducible); keep the assignment CSV
    for sub in ("images", "labels"):
        if (YOLO / sub).exists():
            shutil.rmtree(YOLO / sub)

    xml_box_count = 0
    for row in df.itertuples():
        split, stem = row.project_split, row.image
        img_src = find_image(row.orig_split, stem)
        (YOLO / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_src, YOLO / "images" / split / f"{stem}.jpg")

        lines = []
        for cid, x1, y1, x2, y2, w, h in parse_xml(row.orig_split, stem):
            xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            xml_box_count += 1
        (YOLO / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    # dataset.yaml (relative path; Ultralytics resolves it from 'path')
    yaml = ["path: data/yolo", "train: images/train", "val: images/val",
            "test: images/test", "names:"]
    yaml += [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
    (YOLO / "dataset.yaml").write_text("\n".join(yaml) + "\n")
    print("Wrote", YOLO / "dataset.yaml")

    # ---------------- sanity checks ----------------
    print("\n=== Sanity checks ===")
    txt_box_count, bad_coord, bad_id = 0, 0, 0
    stems = {}
    for split in ("train", "val", "test"):
        imgs = {p.stem for p in (YOLO / "images" / split).glob("*.jpg")}
        lbls = {p.stem for p in (YOLO / "labels" / split).glob("*.txt")}
        assert imgs == lbls, f"{split}: image/label mismatch"
        stems[split] = imgs
        for p in (YOLO / "labels" / split).glob("*.txt"):
            for line in p.read_text().strip().splitlines():
                cid, *vals = line.split()
                txt_box_count += 1
                bad_id += int(int(cid) not in (0, 1, 2))
                bad_coord += int(any(not (0.0 <= float(v) <= 1.0) for v in vals))
        print(f"{split:5s}: images={len(imgs)}, labels={len(lbls)} (one-to-one OK)")

    print("XML boxes:", xml_box_count, "| TXT boxes:", txt_box_count,
          "| equal:", xml_box_count == txt_box_count)
    print("Invalid class ids:", bad_id)
    print("Coordinates outside [0,1]:", bad_coord)
    print("Overlap train&val :", len(stems["train"] & stems["val"]))
    print("Overlap train&test:", len(stems["train"] & stems["test"]))
    print("Overlap val&test  :", len(stems["val"] & stems["test"]))
    print("Total images:", sum(len(v) for v in stems.values()))


if __name__ == "__main__":
    main()