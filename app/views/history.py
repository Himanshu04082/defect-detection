"""Inspection History page: filter the database, open a record, delete a record."""
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from common import CAUTION, get_cfg, get_db
from src.database import queries as q

cfg = get_cfg()
db = get_db()
classes = cfg["data"]["classes"]
PALETTE = [(0, 200, 0), (255, 128, 0), (0, 128, 255), (200, 0, 200), (0, 200, 200)]  # BGR
COLORS = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}
MAX_ROWS = 500

st.title("Inspection History")
st.caption("Every saved inspection. Open a record to see its image and detections.")

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

if q.summary(db)["total"] == 0:
    st.info("No inspections saved yet. Use Single Inspection or Batch Inspection first.")
    st.stop()

# ---------------- filters ----------------
c1, c2, c3 = st.columns(3)
result = c1.selectbox("Result", ["All", "Good", "Defective"])
dtype = c2.selectbox("Defect type (any detection)", ["All"] + classes)
product = c3.selectbox("Product ID", ["All"] + q.distinct_values(db, "product_id"))
c4, c5, c6 = st.columns(3)
batch = c4.selectbox("Batch ID", ["All"] + q.distinct_values(db, "batch_id"))
d_from = c5.date_input("From date (optional)", value=None)
d_to = c6.date_input("To date (optional)", value=None)

filters = {}
if result != "All":
    filters["result"] = result
if dtype != "All":
    filters["defect_type"] = dtype
if product != "All":
    filters["product_id"] = product
if batch != "All":
    filters["batch_id"] = batch
if d_from:
    filters["date_from"] = d_from.isoformat()
if d_to:
    filters["date_to"] = d_to.isoformat()

matching = q.summary(db, **filters)["total"]
df = q.list_inspections(db, limit=MAX_ROWS, **filters)
note = f" (showing the latest {MAX_ROWS})" if matching > MAX_ROWS else ""
st.caption(f"{matching} inspection(s) match the filters{note}.")

if df.empty:
    st.warning("No inspection matches these filters.")
    st.stop()

cols = ["inspection_id", "timestamp", "product_id", "batch_id", "image_name", "result",
        "n_detections", "top_defect_type", "top_confidence", "total_box_area_px", "threshold"]
st.dataframe(df[cols], hide_index=True)
st.download_button("Download this table as CSV", df[cols].to_csv(index=False).encode("utf-8"),
                   file_name="inspection_history.csv", mime="text/csv")

# ---------------- one record ----------------
st.divider()
st.subheader("Open a record")
labels = {int(r.inspection_id): f"#{r.inspection_id} | {r.timestamp} | {r.image_name} | {r.result}"
          for r in df.itertuples()}
choice = st.selectbox("Inspection", list(labels), format_func=lambda i: labels[i])
rec = db.get_inspection(choice)

if rec["result"] == "Defective":
    st.error(f"DEFECTIVE: {rec['n_detections']} detection(s) at T = {rec['threshold']}")
else:
    st.success(f"GOOD: no detection above T = {rec['threshold']}")

img_path = Path(rec["image_path"])
if img_path.exists():
    img = cv2.imread(str(img_path))
    if img is None:
        st.warning("The saved image file could not be read.")
    else:
        annotated = img.copy()
        for d in rec["detections"]:
            color = COLORS.get(d["defect_type"], (0, 200, 0))
            p1, p2 = (int(d["x1"]), int(d["y1"])), (int(d["x2"]), int(d["y2"]))
            cv2.rectangle(annotated, p1, p2, color, 2)
            cv2.putText(annotated, f"{d['defect_type']} {d['confidence']:.2f}",
                        (p1[0] + 2, max(p1[1] + 12, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        left, right = st.columns(2)
        left.markdown("**Original**")
        left.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=400)
        right.markdown("**Detections (drawn from the saved coordinates)**")
        right.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width=400)
else:
    st.warning("The saved image file is missing (it may have been deleted from the uploads folder).")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Product ID", rec["product_id"])
m2.metric("Detections", rec["n_detections"])
m3.metric("Model", f"{rec['model_name']} ({rec['model_version']})")
m4.metric("Time (ms)", rec["inference_ms"] if rec["inference_ms"] is not None else "-")
if rec["batch_id"]:
    st.caption(f"Batch ID: {rec['batch_id']}")

if rec["detections"]:
    st.dataframe(pd.DataFrame([{
        "Defect type": d["defect_type"],
        "Confidence": d["confidence"],
        "Box (x1, y1, x2, y2)": f"{d['x1']:.0f}, {d['y1']:.0f}, {d['x2']:.0f}, {d['y2']:.0f}",
        "Box area (px)": d["area_px"],
        "Box area (% of image)": d["area_pct"],
    } for d in rec["detections"]]), hide_index=True)
    st.caption("Detections are model boxes, not confirmed defects; overlapping boxes can belong "
               "to one defect.")

# ---------------- delete ----------------
with st.expander("Delete this record"):
    st.write("This removes the record, its detections and the saved image copy. It cannot be undone.")
    sure = st.checkbox(f"Yes, delete inspection #{choice}", key=f"sure_{choice}")
    if st.button("Delete", disabled=not sure):
        db.delete_inspection(choice)
        st.session_state["flash"] = f"Inspection #{choice} deleted."
        st.rerun()

with st.expander("Important limitation"):
    st.write(CAUTION)