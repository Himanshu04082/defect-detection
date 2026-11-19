"""Tests for src/database/db.py. Uses a temporary database; the real one is not touched."""
import sqlite3
import tempfile
from pathlib import Path

import numpy as np

from src.database.db import InspectionDB
from src.inference.detector import DefectDetector

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    passed += bool(ok)
    failed += (not ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    except Exception as e:
        print("   unexpected:", type(e).__name__, e)
        return False
    return False


det = DefectDetector()
good_img = np.full((200, 200, 3), 128, np.uint8)
good_res = det.inspect(good_img)
sample = Path("data/yolo/images/val/scratches_9.jpg")
bad_bytes = sample.read_bytes()
bad_res = det.inspect_bytes(bad_bytes, sample.name)

with tempfile.TemporaryDirectory() as tmp:
    db = InspectionDB(db_path=Path(tmp) / "t.db", uploads_dir=Path(tmp) / "uploads")

    with db.session() as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    check("tables created", {"inspections", "detections"} <= tables, f"-> {sorted(tables)}")

    # Defective inspection from a real detector result
    i1 = db.save_inspection(bad_res, bad_bytes, sample.name, product_id="coil-12", batch_id="B1")
    r1 = db.get_inspection(i1)
    check("defective saved", r1["result"] == "Defective" and r1["n_detections"] == bad_res.n_defects,
          f"-> {r1['n_detections']} detections, top {r1['top_defect_type']} {r1['top_confidence']}")
    check("detections rows match", len(r1["detections"]) == bad_res.n_defects == db.count("detections"))
    check("image copy exists", Path(r1["image_path"]).exists(), f"-> {r1['image_path']}")
    check("product and batch stored", r1["product_id"] == "coil-12" and r1["batch_id"] == "B1")
    check("model info stored", r1["model_name"] == "yolov8n" and r1["model_version"] == "seed42-50ep")
    check("timestamp format", len(r1["timestamp"]) == 19 and r1["timestamp"][10] == " ", f"-> {r1['timestamp']}")

    # Good inspection
    i2 = db.save_inspection(good_res, bad_bytes, "blank.png")
    r2 = db.get_inspection(i2)
    check("good saved", r2["result"] == "Good" and r2["n_detections"] == 0 and r2["top_defect_type"] is None,
          f"-> product '{r2['product_id']}', batch {r2['batch_id']}")
    check("good has no detection rows", len(r2["detections"]) == 0)

    # Constraints must reject invalid data
    base = ("INSERT INTO inspections (timestamp,image_name,image_path,result,threshold,n_detections,"
            "model_name,model_version) VALUES ('2026-01-01 00:00:00','a.png','x',?,?,?,'m','v')")
    for label, args in [("bad result value", ("Maybe", 0.15, 1)),
                        ("threshold above 1", ("Defective", 1.5, 1)),
                        ("Good with detections", ("Good", 0.15, 2)),
                        ("Defective with 0 detections", ("Defective", 0.15, 0))]:
        with db.session() as con:
            ok = raises(sqlite3.IntegrityError, lambda a=args: con.execute(base, a))
        check(f"rejects: {label}", ok)
    det_sql = ("INSERT INTO detections (inspection_id,defect_type,confidence,x1,y1,x2,y2,area_px,area_pct) "
               "VALUES (?,?,?,0,0,1,1,1,1)")
    for label, args in [("unknown defect type", (i1, "crack", 0.5)),
                        ("confidence above 1", (i1, "scratches", 1.5)),
                        ("missing parent inspection", (99999, "scratches", 0.5))]:
        with db.session() as con:
            ok = raises(sqlite3.IntegrityError, lambda a=args: con.execute(det_sql, a))
        check(f"rejects: {label}", ok)
    check("invalid extension refused", raises(ValueError, lambda: db.save_inspection(good_res, b"x", "a.txt")))
    check("no orphan files after failures", len(list((Path(tmp) / "uploads").iterdir())) == 2)

    # Cascade delete
    img_path = Path(r1["image_path"])
    check("delete returns True", db.delete_inspection(i1) is True)
    check("cascade removed detections", db.count("detections") == 0)
    check("image copy removed", not img_path.exists())
    check("delete of missing id returns False", db.delete_inspection(i1) is False)
    check("other inspection untouched", db.count("inspections") == 1)

print(f"\nPassed: {passed} | Failed: {failed}")