"""Model Performance page. Numbers are copied from the logged outputs of Phases 6-8
(docs/phase8_protocol.md, reports/training/, the threshold and benchmark scripts)."""
from pathlib import Path

import pandas as pd
import streamlit as st

from common import ROOT

st.title("Model Performance")
st.caption("Frozen results of the YOLOv8n and YOLOv8s experiments (3 defect classes of NEU-DET). "
           "Selected model: YOLOv8n, confidence limit T = 0.15 (chosen on the validation set).")


def show_image(rel_path, caption, width=450):
    p = ROOT / rel_path
    if p.exists():
        st.image(str(p), caption=caption, width=width)
    else:
        st.caption(f"(image not found: {rel_path})")


# ---------------- 1. validation, per seed ----------------
st.header("1. Validation set (144 images), 50 epochs")
COLS = ["model", "seed", "precision", "recall", "mAP50", "mAP50-95",
        "recall inclusion", "recall scratches", "recall rolled-in_scale"]
ROWS = [
    ["YOLOv8n", 42, 0.777, 0.771, 0.818, 0.448, 0.806, 0.928, 0.578],
    ["YOLOv8n", 7, 0.813, 0.720, 0.818, 0.434, 0.768, 0.892, 0.500],
    ["YOLOv8n", 123, 0.785, 0.734, 0.811, 0.439, 0.791, 0.892, 0.520],
    ["YOLOv8s", 42, 0.761, 0.761, 0.804, 0.449, 0.853, 0.940, 0.490],
    ["YOLOv8s", 7, 0.805, 0.714, 0.802, 0.442, 0.781, 0.880, 0.480],
    ["YOLOv8s", 123, 0.744, 0.762, 0.800, 0.427, 0.854, 0.892, 0.539],
]
df = pd.DataFrame(ROWS, columns=COLS)
st.dataframe(df, hide_index=True)

metrics = COLS[2:]
summary = {m: {} for m in metrics}
for name, g in df.groupby("model"):
    for m in metrics:
        summary[m][name] = f"{g[m].mean():.3f} ± {g[m].std():.3f}"
st.subheader("Mean ± standard deviation over 3 seeds")
st.dataframe(pd.DataFrame(summary).T.rename_axis("metric"))
st.caption("Differences between the two models are small (mAP50 0.816 vs 0.802; mAP50-95 0.440 vs 0.439) "
           "and come from 3 seeds on one small validation set. This is a practical comparison, "
           "not a statistical test. F1 is not reported separately; precision and recall are "
           "the values printed by the training framework.")

# ---------------- 2. image-level recall vs T ----------------
st.header("2. Image-level recall vs confidence limit T (validation, seed 42)")
st.write("Rule fixed before the analysis: choose the highest T whose strict image-level recall "
         "(a detection of the right class with IoU >= 0.5) is at least 0.95, i.e. 137 of 144 images.")
st.dataframe(pd.DataFrame({
    "model": ["YOLOv8n", "YOLOv8s"],
    "selected T": [0.15, 0.15],
    "strict recall at T = 0.15": ["137/144 = 0.951", "138/144 = 0.958"],
    "any-detection recall": ["143/144 = 0.993", "142/144 = 0.986"],
    "missed images": [7, 6],
}), hide_index=True)
show_image("reports/figures/image_recall_vs_T.png", "Image-level recall vs T (validation set)", 650)

# ---------------- 3. test ----------------
st.header("3. Locked test set (180 images), evaluated once")
st.write("Model YOLOv8n, T = 0.15, both fixed before the test set was opened "
         "(git tag phase8-model-frozen).")
t1, t2 = st.columns(2)
t1.metric("Strict image-level recall", "171/180 = 0.950")
t2.metric("Any-detection recall", "180/180 = 1.000")
st.dataframe(pd.DataFrame({
    "class": ["all", "inclusion", "scratches", "rolled-in_scale"],
    "mAP50": [0.729, 0.816, 0.832, 0.541],
    "mAP50-95": [0.363, 0.470, 0.398, 0.221],
    "images missed (strict)": [9, 1, 0, 8],
}), hide_index=True)
st.write("Box-level at T = 0.15: 324 matched, 430 unmatched (false positive) boxes, 79 missed "
         "(precision 0.430, recall 0.804). Test precision/recall of the framework: 0.681 / 0.667.")
st.warning("The target (171 of 180) was met with zero margin. Validation numbers were higher "
           "than test numbers (mAP50 0.818 vs 0.729), so the validation figures are optimistic. "
           "8 of the 9 missed test images are rolled-in_scale.")

# ---------------- 4. speed ----------------
st.header("4. Speed (laptop CPU, batch 1, 640 px, 144 val images, 2 rounds)")
st.dataframe(pd.DataFrame({
    "model": ["YOLOv8n", "YOLOv8n", "YOLOv8s", "YOLOv8s"],
    "round": [1, 2, 1, 2],
    "median latency (ms)": [128.9, 132.9, 349.4, 298.4],
    "FPS (1000 / median)": [7.8, 7.5, 2.9, 3.4],
}), hide_index=True)
st.caption("Measured on one laptop CPU (12th Gen Intel Core i5). Timings on a Colab T4 GPU "
           "(about 3 ms and 5 ms per image) are on different hardware and are not comparable.")

# ---------------- 5. confusion matrices ----------------
st.header("5. Confusion matrices (validation, normalised)")
c1, c2 = st.columns(2)
with c1:
    show_image("reports/training/final_v8n/confusion_matrix_normalized.png", "YOLOv8n, seed 42", 420)
with c2:
    show_image("reports/training/final_v8s/confusion_matrix_normalized.png", "YOLOv8s, seed 42", 420)
with st.expander("Training curves (YOLOv8n, seed 42)"):
    show_image("reports/training/final_v8n/results.png", "Loss and metrics per epoch", 800)

# ---------------- 6. limitations ----------------
st.header("6. Limitations")
st.markdown(
    "- Every validation and test image contains a defect, so false alarms on defect-free "
    "products and Good-side accuracy could not be measured.\n"
    "- T = 0.15 was chosen to limit missed defects; it produced about 2.4 unmatched boxes per "
    "defective test image.\n"
    "- Image-level analysis uses seed-42 weights only; seed variability was assessed on "
    "validation box metrics.\n"
    "- Only 3 test images contain two defect types, so results mainly reflect single-defect images.\n"
    "- One duplicate box exists in the source annotations (inclusion_62); it was left unchanged."
)