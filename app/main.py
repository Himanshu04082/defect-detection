"""Entry point:  python -m streamlit run app/main.py   (run from the project root)."""
import streamlit as st

import common  # noqa: F401  (sets up the project path)

st.set_page_config(page_title="AI Defect Detection", layout="wide")

pages = [
    st.Page("views/home.py", title="Home", default=True),
    st.Page("views/single_inspection.py", title="Single Inspection"),
    st.Page("views/batch_inspection.py", title="Batch Inspection"),
    st.Page("views/history.py", title="Inspection History"),
]
st.navigation(pages).run()