"""Home page: overview, counters from the database, model information."""
from pathlib import Path

import pandas as pd
import streamlit as st

from common import CAUTION, get_cfg, get_db
from src.database import queries as q

cfg = get_cfg()
db = get_db()
inf = cfg["inference"]

st.title("AI Defect Detection")
st.caption("Computer-vision inspection of steel surface images "
           "(YOLOv8n trained on a 3-class subset of NEU-DET).")

st.write(
    "Upload a product image and the system finds the defect type, where it is, and how "
    "sure the model is. Every inspection is saved in a database so quality trends can be "
    "studied later."
)

s = q.summary(db)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total inspections", s["total"])
c2.metric("Good", s["good"])
c3.metric("Flagged Defective", s["defective"])
c4.metric("Flagged Defective (%)", "n/a" if s["defect_rate_pct"] is None else f"{s['defect_rate_pct']}%")

st.warning(CAUTION)

st.subheader("Model information")
model_file = Path(inf["model_path"])
info = pd.DataFrame({
    "Item": ["Model", "Version", "Confidence limit (T)", "Input size", "Device",
             "Defect classes", "Model file found"],
    "Value": [inf["model_name"], inf["model_version"], inf["conf_threshold"],
              cfg["preprocessing"]["imgsz"], inf["device"],
              ", ".join(cfg["data"]["classes"]), "Yes" if model_file.exists() else "NO"],
})
st.table(info.astype(str).set_index("Item"))
st.caption("T = 0.15 was chosen on the validation set to limit missed defects "
           "(see Model Performance page, added later).")

st.subheader("Latest inspections")
latest = q.list_inspections(db, limit=5)
if latest.empty:
    st.info("No inspections saved yet. The Single Inspection page is added in the next step.")
else:
    st.dataframe(
        latest[["inspection_id", "timestamp", "product_id", "image_name", "result",
                "n_detections", "top_defect_type", "top_confidence"]],
        hide_index=True,
    )