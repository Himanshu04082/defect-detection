"""Read-only query helpers for the dashboard (Phase 10, Step 3).

Every user-supplied value is bound as an SQL parameter. Results are pandas DataFrames.
"Defect rate" here means the share of inspections the model flagged Defective.
"""
import pandas as pd

GROUP_COLUMNS = ("product_id", "batch_id")


def _where(result=None, defect_type=None, product_id=None, batch_id=None,
           date_from=None, date_to=None):
    clauses, params = [], []
    if result:
        clauses.append("i.result = ?")
        params.append(result)
    if defect_type:
        clauses.append("EXISTS (SELECT 1 FROM detections d WHERE "
                       "d.inspection_id = i.inspection_id AND d.defect_type = ?)")
        params.append(defect_type)
    if product_id:
        clauses.append("i.product_id = ?")
        params.append(product_id)
    if batch_id:
        clauses.append("i.batch_id = ?")
        params.append(batch_id)
    if date_from:                                   # 'YYYY-MM-DD' (or a date object)
        clauses.append("i.timestamp >= ?")
        params.append(f"{date_from} 00:00:00")
    if date_to:
        clauses.append("i.timestamp <= ?")
        params.append(f"{date_to} 23:59:59")
    return ("WHERE " + " AND ".join(clauses)) if clauses else "", params


def _df(db, sql, params=()):
    with db.session() as con:
        return pd.read_sql_query(sql, con, params=list(params))


def _rate(df):
    df["defect_rate_pct"] = (100 * df["defective"] / df["total"]).round(1)
    return df


def summary(db, **filters):
    where, params = _where(**filters)
    row = _df(db, f"""SELECT COUNT(*) AS total,
                             COALESCE(SUM(i.result = 'Good'), 0) AS good,
                             COALESCE(SUM(i.result = 'Defective'), 0) AS defective,
                             AVG(i.inference_ms) AS avg_ms
                      FROM inspections i {where}""", params).iloc[0]
    total = int(row["total"])
    return {"total": total, "good": int(row["good"]), "defective": int(row["defective"]),
            "defect_rate_pct": round(100 * int(row["defective"]) / total, 1) if total else None,
            "avg_inference_ms": None if (total == 0 or pd.isna(row["avg_ms"])) else round(float(row["avg_ms"]), 1)}


def list_inspections(db, limit=500, offset=0, **filters):
    where, params = _where(**filters)
    return _df(db, f"""SELECT i.* FROM inspections i {where}
                       ORDER BY i.timestamp DESC, i.inspection_id DESC LIMIT ? OFFSET ?""",
               params + [int(limit), int(offset)])


def defect_distribution(db, **filters):
    where, params = _where(**filters)
    return _df(db, f"""SELECT d.defect_type, COUNT(*) AS n_detections,
                              COUNT(DISTINCT d.inspection_id) AS n_images
                       FROM detections d JOIN inspections i ON i.inspection_id = d.inspection_id
                       {where} GROUP BY d.defect_type
                       ORDER BY n_images DESC, n_detections DESC, d.defect_type""", params)


def most_common_defect(db, **filters):
    d = defect_distribution(db, **filters)
    return None if d.empty else d.iloc[0]["defect_type"]


def trend_by_day(db, **filters):
    where, params = _where(**filters)
    df = _df(db, f"""SELECT substr(i.timestamp, 1, 10) AS day, COUNT(*) AS total,
                            COALESCE(SUM(i.result = 'Defective'), 0) AS defective
                     FROM inspections i {where} GROUP BY day ORDER BY day""", params)
    return _rate(df)


def top_confidences(db, **filters):
    where, params = _where(**filters)
    df = _df(db, f"""SELECT i.inspection_id, i.top_defect_type, i.top_confidence
                     FROM inspections i {where}""", params)
    return df.dropna(subset=["top_confidence"])


def by_group(db, column, **filters):
    if column not in GROUP_COLUMNS:
        raise ValueError(f"column must be one of {GROUP_COLUMNS}")
    where, params = _where(**filters)
    where = f"{where} AND i.{column} IS NOT NULL" if where else f"WHERE i.{column} IS NOT NULL"
    df = _df(db, f"""SELECT i.{column} AS grp, COUNT(*) AS total,
                            COALESCE(SUM(i.result = 'Defective'), 0) AS defective
                     FROM inspections i {where} GROUP BY i.{column}
                     ORDER BY total DESC, grp""", params)
    return _rate(df)


def distinct_values(db, column):
    if column not in GROUP_COLUMNS:
        raise ValueError(f"column must be one of {GROUP_COLUMNS}")
    with db.session() as con:
        return [r[0] for r in con.execute(
            f"SELECT DISTINCT {column} FROM inspections WHERE {column} IS NOT NULL ORDER BY {column}")]