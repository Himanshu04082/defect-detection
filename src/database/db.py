"""SQLite storage for inspections (Phase 10).

Two tables: `inspections` (one row per image) and `detections` (one row per detection).
Constraints in the schema keep the data valid (NFR-07). Local, single-user use only.
"""
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from src.utils.config import load_config

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _schema_sql(classes):
    types = ", ".join(f"'{c}'" for c in classes)
    return f"""
CREATE TABLE IF NOT EXISTS inspections (
    inspection_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,
    product_id        TEXT NOT NULL DEFAULT 'unknown',
    batch_id          TEXT,
    image_name        TEXT NOT NULL,
    image_path        TEXT NOT NULL,
    result            TEXT NOT NULL CHECK (result IN ('Good', 'Defective')),
    threshold         REAL NOT NULL CHECK (threshold > 0 AND threshold < 1),
    n_detections      INTEGER NOT NULL CHECK (n_detections >= 0),
    top_defect_type   TEXT CHECK (top_defect_type IS NULL OR top_defect_type IN ({types})),
    top_confidence    REAL CHECK (top_confidence IS NULL OR (top_confidence >= 0 AND top_confidence <= 1)),
    total_box_area_px REAL CHECK (total_box_area_px IS NULL OR total_box_area_px >= 0),
    inference_ms      REAL,
    model_name        TEXT NOT NULL,
    model_version     TEXT NOT NULL,
    CHECK ((result = 'Good' AND n_detections = 0) OR (result = 'Defective' AND n_detections > 0))
);

CREATE TABLE IF NOT EXISTS detections (
    detection_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL REFERENCES inspections(inspection_id) ON DELETE CASCADE,
    defect_type   TEXT NOT NULL CHECK (defect_type IN ({types})),
    confidence    REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
    area_px       REAL NOT NULL CHECK (area_px >= 0),
    area_pct      REAL NOT NULL CHECK (area_pct >= 0)
);

CREATE INDEX IF NOT EXISTS idx_insp_time   ON inspections(timestamp);
CREATE INDEX IF NOT EXISTS idx_insp_result ON inspections(result);
CREATE INDEX IF NOT EXISTS idx_insp_batch  ON inspections(batch_id);
CREATE INDEX IF NOT EXISTS idx_det_insp    ON detections(inspection_id);
CREATE INDEX IF NOT EXISTS idx_det_type    ON detections(defect_type);
"""


class InspectionDB:
    def __init__(self, db_path=None, uploads_dir=None, cfg=None):
        cfg = cfg or load_config()
        self.db_path = Path(db_path or cfg["database"]["path"])
        self.uploads_dir = Path(uploads_dir or cfg["database"]["uploads_dir"])
        self.classes = cfg["data"]["classes"]
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        with self.session() as con:
            con.executescript(_schema_sql(self.classes))

    def connect(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")     # needed for ON DELETE CASCADE
        return con

    @contextmanager
    def session(self):
        con = self.connect()
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def save_inspection(self, result, image_bytes, image_name, product_id="unknown", batch_id=None):
        """Save one InspectionResult (from DefectDetector) plus a copy of the image."""
        name = Path(image_name).name
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise ValueError(f"Unsupported image extension: '{ext}'")
        product = (product_id or "").strip()[:100] or "unknown"
        batch = (batch_id or "").strip()[:100] or None
        now = datetime.now()
        stamp = now.strftime("%Y-%m-%d %H:%M:%S")
        saved = self.uploads_dir / f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
        saved.write_bytes(image_bytes)

        dets = result.detections
        top = dets[0] if dets else None
        try:
            with self.session() as con:
                cur = con.execute(
                    """INSERT INTO inspections
                       (timestamp, product_id, batch_id, image_name, image_path, result, threshold,
                        n_detections, top_defect_type, top_confidence, total_box_area_px,
                        inference_ms, model_name, model_version)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (stamp, product, batch, name, saved.as_posix(), result.decision, result.threshold,
                     len(dets), top.defect_type if top else None, top.confidence if top else None,
                     sum(d.area_px for d in dets), result.inference_ms,
                     result.model_name, result.model_version))
                iid = cur.lastrowid
                con.executemany(
                    """INSERT INTO detections
                       (inspection_id, defect_type, confidence, x1, y1, x2, y2, area_px, area_pct)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    [(iid, d.defect_type, d.confidence, d.x1, d.y1, d.x2, d.y2, d.area_px, d.area_pct)
                     for d in dets])
        except Exception:
            saved.unlink(missing_ok=True)     # do not keep an orphan image if the insert failed
            raise
        return iid

    def get_inspection(self, inspection_id):
        with self.session() as con:
            row = con.execute("SELECT * FROM inspections WHERE inspection_id = ?",
                              (inspection_id,)).fetchone()
            if row is None:
                return None
            out = dict(row)
            out["detections"] = [dict(r) for r in con.execute(
                "SELECT * FROM detections WHERE inspection_id = ? ORDER BY confidence DESC",
                (inspection_id,))]
            return out

    def delete_inspection(self, inspection_id):
        with self.session() as con:
            row = con.execute("SELECT image_path FROM inspections WHERE inspection_id = ?",
                              (inspection_id,)).fetchone()
            if row is None:
                return False
            con.execute("DELETE FROM inspections WHERE inspection_id = ?", (inspection_id,))
        Path(row["image_path"]).unlink(missing_ok=True)
        return True

    def count(self, table="inspections"):
        if table not in ("inspections", "detections"):
            raise ValueError("unknown table")
        with self.session() as con:
            return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]