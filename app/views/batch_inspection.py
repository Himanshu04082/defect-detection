"""Batch Inspection page: many images at once, results table, filters, CSV download."""
from datetime import datetime

import pandas as pd
import streamlit as st

from common import CAUTION, get_cfg, get_db, get_detector

MAX_FILES = 100

cfg = get_cfg()
db = get_db()
FROZEN_T = float(cfg["inference"]["conf_threshold"])

st.title("Batch Inspection")
st.caption("Upload several images at once. Each image is inspected and saved; "
           "a damaged file is skipped and listed under 'Failed', it does not stop the batch.")

files = st.file_uploader("Upload images (up to 100)", type=["jpg", "jpeg", "png", "bmp"],
                         accept_multiple_files=True)
col_a, col_b, col_c = st.columns(3)
product_id = col_a.text_input("Product ID (optional)", max_chars=100)
batch_name = col_b.text_input("Batch name (optional)", max_chars=60,
                              help="The date and time are added automatically.")
threshold = col_c.slider("Confidence limit (T)", 0.05, 0.95, FROZEN_T, 0.05)
if abs(threshold - FROZEN_T) > 1e-9:
    st.info(f"You changed T. The frozen value from the evaluation is {FROZEN_T}; "
            f"this batch will be saved with T = {threshold}.")
save = st.checkbox("Save these inspections to the database", value=True)

too_many = len(files) > MAX_FILES
if too_many:
    st.error(f"{len(files)} files selected. Please upload at most {MAX_FILES} images per batch.")

if st.button("Run batch", type="primary", disabled=(not files) or too_many):
    detector = get_detector()
    batch_id = f"{(batch_name or 'batch').strip()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rows, failed = [], []
    bar = st.progress(0.0, text="Starting...")
    for k, f in enumerate(files, 1):
        try:
            data = f.getvalue()
            res = detector.inspect_bytes(data, f.name, threshold)
            iid = (db.save_inspection(res, data, f.name, product_id=product_id, batch_id=batch_id)
                   if save else None)
            top = res.top_detection
            rows.append({
                "inspection_id": iid,
                "file": f.name,
                "result": res.decision,
                "detections": res.n_defects,
                "defect_types": ", ".join(sorted({d.defect_type for d in res.detections})),
                "top_defect": top.defect_type if top else "",
                "top_confidence": top.confidence if top else None,
                "total_box_area_px": round(sum(d.area_px for d in res.detections), 1),
                "time_ms": res.inference_ms,
                "threshold": res.threshold,
                "product_id": (product_id or "").strip() or "unknown",
                "batch_id": batch_id,
                "model": f"{res.model_name} ({res.model_version})",
            })
        except Exception as e:                       # one bad file must not stop the batch
            failed.append({"file": f.name, "reason": str(e)})
        bar.progress(k / len(files), text=f"{k} of {len(files)} images done")
    bar.empty()
    st.session_state["batch"] = {"rows": rows, "failed": failed, "batch_id": batch_id, "saved": save}

b = st.session_state.get("batch")
if b:
    st.divider()
    df = pd.DataFrame(b["rows"])
    n_ok, n_fail = len(b["rows"]), len(b["failed"])
    n_def = int((df["result"] == "Defective").sum()) if n_ok else 0

    st.subheader(f"Batch result: {b['batch_id']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Processed", n_ok)
    m2.metric("Good", n_ok - n_def)
    m3.metric("Flagged Defective", n_def)
    m4.metric("Failed", n_fail)
    if b["saved"]:
        st.success(f"{n_ok} inspection(s) saved to the database with batch ID '{b['batch_id']}'.")
    else:
        st.info("These inspections were not saved.")

    if b["failed"]:
        st.warning("Some files could not be inspected:")
        st.dataframe(pd.DataFrame(b["failed"]), hide_index=True)

    if n_ok:
        f1, f2 = st.columns(2)
        pick_result = f1.multiselect("Show results", ["Good", "Defective"], default=["Good", "Defective"])
        all_types = sorted({t for s in df["defect_types"] for t in s.split(", ") if t})
        pick_types = f2.multiselect("Only images with defect type", all_types)

        view = df[df["result"].isin(pick_result)]
        if pick_types:
            view = view[view["defect_types"].apply(lambda s: any(t in s.split(", ") for t in pick_types))]

        st.caption(f"Showing {len(view)} of {n_ok} inspected images.")
        st.dataframe(view, hide_index=True)
        st.download_button("Download CSV (rows shown above)",
                           view.to_csv(index=False).encode("utf-8"),
                           file_name=f"{b['batch_id']}.csv", mime="text/csv",
                           disabled=view.empty)
        st.caption("'detections' counts model boxes, not confirmed defects: one defect can get "
                   "several overlapping boxes.")
    with st.expander("Important limitation"):
        st.write(CAUTION)