"""Shared helpers for the Streamlit app: project path setup, database, caution text."""
import os
import sys
from pathlib import Path

import streamlit as st

# Streamlit puts only app/ on sys.path; the project root is needed for `import src...`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)          # config and model paths in config.yaml are relative to the root

from src.database.db import InspectionDB   # noqa: E402  (must come after the path setup)
from src.utils.config import load_config   # noqa: E402

CAUTION = (
    "Good-side accuracy has not been measured: every validation and test image was "
    "defective, and the confidence limit was tuned to reduce missed defects. "
    "'Defective' means the model flagged at least one detection; the number of "
    "detections can be larger than the number of real defects (overlapping boxes)."
)


@st.cache_resource
def get_db():
    return InspectionDB()


def get_cfg():
    return load_config()



@st.cache_resource(show_spinner="Loading model...")
def get_detector():
    import numpy as np
    from src.inference.detector import DefectDetector   # imported here so Home loads fast
    det = DefectDetector()
    det.inspect(np.full((200, 200, 3), 128, np.uint8))   # warm-up call (not saved anywhere)
    return det