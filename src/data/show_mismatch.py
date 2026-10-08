"""List a few images whose XML classes differ from their folder name."""
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path("data/raw/neu-det/NEU-DET")

shown = 0
for p in sorted((ROOT / "train" / "images").glob("*/*.jpg")):
    xml_path = ROOT / "train" / "annotations" / f"{p.stem}.xml"
    if not xml_path.exists():
        continue
    objs = ET.parse(xml_path).getroot().findall("object")
    names = [o.findtext("name") for o in objs]
    if set(names) != {p.parent.name}:
        print(f"{p.parent.name}/{p.name}  ->  XML classes: {names}")
        shown += 1
    if shown == 8:
        break