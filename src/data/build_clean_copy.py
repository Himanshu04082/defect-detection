"""Create a cleaned copy of NEU-DET in data/processed (raw stays untouched).

Rule 3: crazing_240.xml sits in validation/annotations while its image is
in train/images. Copy it into train/annotations instead.
"""
import shutil
from pathlib import Path

RAW = Path("data/raw/neu-det/NEU-DET")
CLEAN = Path("data/processed/neu-det-clean")

# (stem, split where the XML wrongly lives, split where the image is)
MOVES = [("crazing_240", "validation", "train")]


def main():
    if CLEAN.exists():
        shutil.rmtree(CLEAN)  # rebuild from scratch -> reproducible
    shutil.copytree(RAW, CLEAN)

    for stem, wrong_split, right_split in MOVES:
        src = RAW / wrong_split / "annotations" / f"{stem}.xml"
        dst = CLEAN / right_split / "annotations" / f"{stem}.xml"
        shutil.copy2(src, dst)  # copy from RAW, so raw is never changed
        (CLEAN / wrong_split / "annotations" / f"{stem}.xml").unlink()
        print(f"Moved (in clean copy only): {stem}.xml  {wrong_split} -> {right_split}")

    for split in ("train", "validation"):
        n_img = len(list((CLEAN / split / "images").glob("*/*.jpg")))
        n_xml = len(list((CLEAN / split / "annotations").glob("*.xml")))
        print(f"{split}: images={n_img}, xml={n_xml}")


if __name__ == "__main__":
    main()