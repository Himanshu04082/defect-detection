"""Quality Analytics page: charts computed from the saved inspections."""
import pandas as pd
import streamlit as st

from common import CAUTION, get_db
from src.database import queries as q

db = get_db()

st.title("Quality Analytics")
st.caption("Charts from the inspections saved in the database. 'Flagged Defective' means the "
           "model flagged at least one detection; it is not a confirmed quality measurement.")

if q.summary(db)["total"] == 0:
    st.info("No inspections saved yet. Use Single Inspection or Batch Inspection first.")
    st.stop()

# ---------------- filters ----------------
c1, c2 = st.columns(2)
product = c1.selectbox("Product ID", ["All"] + q.distinct_values(db, "product_id"))
batch = c2.selectbox("Batch ID", ["All"] + q.distinct_values(db, "batch_id"))
c3, c4 = st.columns(2)
d_from = c3.date_input("From date (optional)", value=None)
d_to = c4.date_input("To date (optional)", value=None)

filters = {}
if product != "All":
    filters["product_id"] = product
if batch != "All":
    filters["batch_id"] = batch
if d_from:
    filters["date_from"] = d_from.isoformat()
if d_to:
    filters["date_to"] = d_to.isoformat()

s = q.summary(db, **filters)
if s["total"] == 0:
    st.warning("No inspection matches these filters.")
    st.stop()

# ---------------- headline numbers ----------------
common_defect = q.most_common_defect(db, **filters)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total inspections", s["total"])
m2.metric("Good", s["good"])
m3.metric("Flagged Defective", s["defective"])
m4.metric("Flagged Defective (%)", f"{s['defect_rate_pct']}%")
m5.metric("Most common defect", common_defect or "-")
st.caption("'Most common defect' = the defect type found in the most images.")
if s["total"] < 30:
    st.info(f"Only {s['total']} inspection(s) match: percentages and trends are not stable "
            "with so few images.")

# ---------------- good vs defective ----------------
st.subheader("Good vs flagged Defective")
st.bar_chart(pd.DataFrame({"inspections": [s["good"], s["defective"]]}, index=["Good", "Defective"]))

# ---------------- defect distribution ----------------
st.subheader("Defect distribution")
dist = q.defect_distribution(db, **filters)
if dist.empty:
    st.info("No detections in the selected inspections.")
else:
    st.bar_chart(dist.set_index("defect_type")[["n_images", "n_detections"]], stack=False)
    st.dataframe(dist.rename(columns={"n_images": "Images with this type",
                                      "n_detections": "Detections (boxes)"}), hide_index=True)
    st.caption("An image can contain several defect types, and one defect can get several "
               "overlapping boxes, so these counts do not add up to the number of images.")

# ---------------- trend ----------------
st.subheader("Trend over time (per day)")
tr = q.trend_by_day(db, **filters)
st.bar_chart(tr.set_index("day")[["total", "defective"]], stack=False)
st.dataframe(tr.rename(columns={"defect_rate_pct": "flagged_pct"}), hide_index=True)
if len(tr) < 3:
    st.caption("Only a few days of data so far; a trend needs inspections over several days.")

# ---------------- confidence ----------------
st.subheader("Confidence of the top detection")
tc = q.top_confidences(db, **filters)
if tc.empty:
    st.info("No detections to show.")
else:
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    counts = pd.cut(tc["top_confidence"], bins=bins, labels=labels, include_lowest=True).value_counts(sort=False)
    st.bar_chart(counts.rename("images"))
    per_type = (tc.groupby("top_defect_type")["top_confidence"].agg(["count", "mean", "median"])
                .round(3).rename(columns={"count": "images", "mean": "mean_conf", "median": "median_conf"}))
    st.dataframe(per_type)
    st.caption("Confidence is the model's score, not the probability that a defect is real.")

# ---------------- product / batch ----------------
st.subheader("Product-wise and batch-wise")
for col, title in (("product_id", "By product"), ("batch_id", "By batch")):
    st.markdown(f"**{title}**")
    g = q.by_group(db, col, **filters)
    if g.empty:
        st.caption("No records with this field in the selection.")
        continue
    g = g.rename(columns={"grp": col, "defect_rate_pct": "flagged_pct"})
    st.bar_chart(g.set_index(col)["flagged_pct"])
    st.dataframe(g, hide_index=True)

with st.expander("Important limitation"):
    st.write(CAUTION)