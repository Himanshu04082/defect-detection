"""Check quality of NEU-DET XML annotations (Pascal VOC format).

Matching is done by file stem (XML name == image name), because the
<filename> tag inside some XML files is unreliable.
"""
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path("data/processed/neu-det-clean")


def check_split(split):
    ann_dir = ROOT / split / "annotations"
    img_dir = ROOT / split / "images"

    # stem -> (image path, folder name)
    images = {p.stem: p for p in img_dir.glob("*/*.jpg")}
    xmls = {p.stem: p for p in ann_dir.glob("*.xml")}

    problems = Counter()
    examples = {}
    boxes_per_class = Counter()
    folder_mismatch_images = 0
    filename_no_ext = 0

    def note(kind, detail):
        problems[kind] += 1
        examples.setdefault(kind, []).append(detail)

    for stem in sorted(set(images) - set(xmls)):
        note("image without XML", stem)
    for stem in sorted(set(xmls) - set(images)):
        note("XML without image", stem)

    for stem, xml_path in sorted(xmls.items()):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            note("XML parse error", stem)
            continue

        filename = root.findtext("filename") or ""
        if not filename.lower().endswith(".jpg"):
            filename_no_ext += 1

        width = int(root.findtext("size/width"))
        height = int(root.findtext("size/height"))
        objs = root.findall("object")
        if not objs:
            note("no objects", stem)

        folder = images[stem].parent.name if stem in images else None
        names_in_xml = set()
        for obj in objs:
            name = obj.findtext("name")
            names_in_xml.add(name)
            boxes_per_class[name] += 1
            b = obj.find("bndbox")
            xmin, ymin = int(b.findtext("xmin")), int(b.findtext("ymin"))
            xmax, ymax = int(b.findtext("xmax")), int(b.findtext("ymax"))
            if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
                note("bad box", f"{stem} {(xmin, ymin, xmax, ymax)}")

        if folder and names_in_xml != {folder}:
            folder_mismatch_images += 1

    print(f"\n===== {split} =====")
    print("Images:", len(images), "| XML files:", len(xmls))
    print("Boxes per class:", dict(boxes_per_class))
    print("<filename> without .jpg:", filename_no_ext)
    print("Images whose XML classes differ from folder name:", folder_mismatch_images)
    print("Problems by type:", dict(problems) if problems else "none")
    for kind, items in examples.items():
        print(f"  {kind}: {items[:10]}")


if __name__ == "__main__":
    for s in ("train", "validation"):
        check_split(s)