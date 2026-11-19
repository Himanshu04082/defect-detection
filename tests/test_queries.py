"""Tests for src/database/queries.py using real detector output and a temporary database."""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.database import queries as q
from src.database.db import InspectionDB
from src.inference.detector import DefectDetector

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    passed += bool(ok)
    failed += (not ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


det = DefectDetector()
val = Path("data/yolo/images/val")
items = []                                   # (name, bytes, result, product, batch)
for cls in ("inclusion", "scratches", "rolled-in_scale"):
    for p in sorted(val.glob(f"{cls}_*.jpg"))[:4]:
        data = p.read_bytes()
        items.append((p.name, data, det.inspect_bytes(data, p.name)))
blank = cv2.imencode(".png", np.full((200, 200, 3), 128, np.uint8))[1].tobytes()
items.append(("blank.png", blank, det.inspect_bytes(blank, "blank.png")))
N = len(items)


def has_type(res, t):
    return any(d.defect_type == t for d in res.detections)


with tempfile.TemporaryDirectory() as tmp:
    db = InspectionDB(db_path=Path(tmp) / "t.db", uploads_dir=Path(tmp) / "up")

    # empty database first
    s0 = q.summary(db)
    check("empty db summary", s0["total"] == 0 and s0["defect_rate_pct"] is None, f"-> {s0}")
    check("empty db queries return empty", q.list_inspections(db).empty
          and q.defect_distribution(db).empty and q.most_common_defect(db) is None)

    ids = []
    for k, (name, data, res) in enumerate(items):
        product = "P1" if k < 6 else "P2"
        batch = "B1" if k < 8 else ("B2" if k < 12 else None)
        ids.append(db.save_inspection(res, data, name, product_id=product, batch_id=batch))
    results = [r for _, _, r in items]
    n_def = sum(r.decision == "Defective" for r in results)

    s = q.summary(db)
    check("summary counts", s["total"] == N and s["defective"] == n_def and s["good"] == N - n_def,
          f"-> {s}")
    check("defect rate", s["defect_rate_pct"] == round(100 * n_def / N, 1))
    check("list all", len(q.list_inspections(db)) == N)
    check("list limit/offset", len(q.list_inspections(db, limit=5)) == 5
          and len(q.list_inspections(db, limit=5, offset=10)) == N - 10)
    check("filter result", len(q.list_inspections(db, result="Defective")) == n_def)
    for t in ("inclusion", "scratches", "rolled-in_scale"):
        exp = sum(has_type(r, t) for r in results)
        check(f"filter defect_type={t}", len(q.list_inspections(db, defect_type=t)) == exp, f"-> {exp}")
    check("filter product", len(q.list_inspections(db, product_id="P1")) == 6)
    check("filter batch", len(q.list_inspections(db, batch_id="B1")) == 8)
    check("combined filters", len(q.list_inspections(db, product_id="P2", result="Good"))
          == sum(r.decision == "Good" for r in results[6:]))

    dist = q.defect_distribution(db)
    total_dets = sum(r.n_defects for r in results)
    check("distribution detections total", int(dist["n_detections"].sum()) == total_dets,
          f"-> {total_dets}")
    for _, row in dist.iterrows():
        check(f"distribution images for {row['defect_type']}",
              int(row["n_images"]) == sum(has_type(r, row["defect_type"]) for r in results))
    best = int(dist["n_images"].max())
    check("most common defect", q.most_common_defect(db) in set(dist[dist["n_images"] == best]["defect_type"]))

    tc = q.top_confidences(db)
    check("top confidences only for defective", len(tc) == n_def and tc["top_confidence"].between(0, 1).all())

    g = q.by_group(db, "product_id")
    check("by product totals", int(g["total"].sum()) == N and set(g["grp"]) == {"P1", "P2"})
    gb = q.by_group(db, "batch_id")
    check("by batch skips missing batch", int(gb["total"].sum()) == 12 and set(gb["grp"]) == {"B1", "B2"})
    check("distinct values", q.distinct_values(db, "product_id") == ["P1", "P2"])

    # dates: move three rows to another day (test-only change)
    with db.session() as con:
        con.execute("UPDATE inspections SET timestamp = '2026-01-05 10:00:00' WHERE inspection_id IN (?,?,?)",
                    ids[:3])
    check("date filter", len(q.list_inspections(db, date_from="2026-01-01", date_to="2026-01-31")) == 3)
    tr = q.trend_by_day(db)
    check("trend by day", len(tr) == 2 and int(tr["total"].sum()) == N, f"-> {tr['day'].tolist()}")
    check("trend rate column", tr["defect_rate_pct"].between(0, 100).all())

    # safety
    evil = "x'; DROP TABLE inspections; --"
    check("injection text is harmless", q.list_inspections(db, product_id=evil).empty and db.count() == N)
    for fn in (lambda: q.by_group(db, "result"), lambda: q.distinct_values(db, "image_path")):
        try:
            fn()
            check("rejects bad column", False)
        except ValueError:
            check("rejects bad column", True)

print(f"\nPassed: {passed} | Failed: {failed}")