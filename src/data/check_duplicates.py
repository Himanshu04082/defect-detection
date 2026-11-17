"""Find exact duplicate boxes in YOLO labels and in the source XML."""
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pandas as pd

YOLO = Path("data/yolo")
CLEAN = Path("data/processed/neu-det-clean")
assign = pd.read_csv(YOLO / "split_assignment.csv")
orig = dict(zip(assign["image"], assign["orig_split"]))

bad_txt, bad_xml = {}, {}
for p in sorted((YOLO / "labels").glob("*/*.txt")):
    lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    extra = sum(c - 1 for c in Counter(lines).values() if c > 1)
    if extra:
        bad_txt[p.stem] = extra

for stem in bad_txt:
    root = ET.parse(CLEAN / orig[stem] / "annotations" / f"{stem}.xml").getroot()
    boxes = []
    for o in root.findall("object"):
        b = o.find("bndbox")
        boxes.append((o.findtext("name"),) + tuple(int(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")))
    extra = sum(c - 1 for c in Counter(boxes).values() if c > 1)
    bad_xml[stem] = extra

print("Label files with duplicate boxes:", len(bad_txt))
print("Total duplicate boxes in YOLO labels:", sum(bad_txt.values()))
for stem, n in bad_txt.items():
    print(f"  {stem}: txt duplicates={n}, same duplicates already in source XML={bad_xml[stem]}")