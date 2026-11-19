"""Single Inspection page: upload one image, run the detector, show and save the result."""
import pandas as pd
import streamlit as st

from common import CAUTION, get_cfg, get_db, get_detector
from src.inference.detector import InspectionError

cfg = get_cfg()
db = get_db()
FROZEN_T = float(cfg["inference"]["conf_threshold"])

st.title("Single Inspection")
st.caption("Upload one image. The model finds defects; the product is flagged Defective "
           "if at least one detection is above the confidence limit.")

file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "bmp"])
col_a, col_b = st.columns(2)
product_id = col_a.text_input("Product ID (optional)", max_chars=100)
threshold = col_b.slider("Confidence limit (T)", 0.05, 0.95, FROZEN_T, 0.05)
if abs(threshold - FROZEN_T) > 1e-9:
    st.info(f"You changed T. The frozen value from the evaluation is {FROZEN_T}; "
            f"this inspection will be saved with T = {threshold}.")
save = st.checkbox("Save this inspection to the database", value=True)

if st.button("Inspect", type="primary", disabled=file is None):
    try:
        data = file.getvalue()
        with st.spinner("Analysing image..."):
            res = get_detector().inspect_bytes(data, file.name, threshold)
        iid = db.save_inspection(res, data, file.name, product_id=product_id) if save else None
        st.session_state["single"] = {"res": res, "data": data, "name": file.name, "iid": iid}
    except (InspectionError, ValueError) as e:
        st.session_state.pop("single", None)
        st.error(str(e))

state = st.session_state.get("single")
if state:
    res = state["res"]
    st.divider()
    st.subheader(f"Result for: {state['name']}")
    if res.decision == "Defective":
        st.error(f"DEFECTIVE: the model flagged {res.n_defects} detection(s) at T = {res.threshold}")
    else:
        st.success(f"GOOD: no detection above T = {res.threshold}")

    left, right = st.columns(2)
    left.markdown("**Original**")
    left.image(state["data"], width=400)
    right.markdown("**Detected**")
    right.image(res.annotated_rgb, width=400)

    top = res.top_detection
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Detections", res.n_defects)
    m2.metric("Top defect type", top.defect_type if top else "-")
    m3.metric("Top confidence", f"{top.confidence:.2f}" if top else "-")
    m4.metric("Time (ms)", res.inference_ms)

    if res.detections:
        st.dataframe(pd.DataFrame([{
            "Defect type": d.defect_type,
            "Confidence": d.confidence,
            "Box (x1, y1, x2, y2)": f"{d.x1:.0f}, {d.y1:.0f}, {d.x2:.0f}, {d.y2:.0f}",
            "Box area (px)": d.area_px,
            "Box area (% of image)": d.area_pct,
        } for d in res.detections]), hide_index=True)
        st.caption("These are model detections, not confirmed defects. One defect can get several "
                   "overlapping boxes, so the number of detections can be larger than the number of defects.")

    if state["iid"] is not None:
        st.success(f"Saved to the database as inspection #{state['iid']}.")
    else:
        st.info("This inspection was not saved.")
    with st.expander("Important limitation"):
        st.write(CAUTION)