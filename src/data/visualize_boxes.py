"""Draw XML boxes on selected images and save a grid for visual checking."""
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("data/raw/neu-det/NEU-DET")
OUT = Path("reports/figures/box_check.png")

COLORS = {
    "crazing": (255, 0, 0),
    "inclusion": (0, 200, 0),
    "patches": (0, 0, 255),
    "pitted_surface": (0, 200, 200),
    "rolled-in_scale": (200, 0, 200),
    "scratches": (0, 128, 255),
}

# (split, folder, stem, where the XML lives)
SAMPLES = [
    ("train", "crazing", "crazing_240", "validation"),  # the misplaced one
    ("train", "crazing", "crazing_104", "train"),
    ("train", "crazing", "crazing_105", "train"),
    ("train", "patches", "patches_155", "train"),
    ("train", "patches", "patches_168", "train"),
    ("train", "scratches", "scratches_1", "train"),
]
SCALE = 3  # enlarge 200x200 -> 600x600 so boxes are visible


def draw(split, folder, stem, xml_split):
    img = cv2.imread(str(ROOT / split / "images" / folder / f"{stem}.jpg"))
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    xml_path = ROOT / xml_split / "annotations" / f"{stem}.xml"
    for obj in ET.parse(xml_path).getroot().findall("object"):
        name = obj.findtext("name")
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (int(b.findtext(k)) * SCALE for k in ("xmin", "ymin", "xmax", "ymax"))
        color = COLORS[name]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, name, (x1 + 3, max(y1 + 16, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(img, stem, (5, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


tiles = [draw(*s) for s in SAMPLES]
rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
OUT.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT), np.vstack(rows))
print("Saved:", OUT)