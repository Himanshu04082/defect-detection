"""Reusable inference: image in -> Good / Defective decision + detections out.

Decision rule (frozen in Phase 8): at least one detection with confidence above the
threshold T means DEFECTIVE, otherwise GOOD. T comes from config/config.yaml.
No denoising or CLAHE (frozen in Phase 5); YOLO does its own resize to imgsz.
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils.config import load_config

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_BYTES = 10 * 1024 * 1024
MIN_SIDE = 32


class InspectionError(Exception):
    """Base class for errors the app can show to the user."""


class InvalidImageError(InspectionError):
    pass


class ModelNotFoundError(InspectionError):
    pass


@dataclass
class Detection:
    defect_type: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    area_px: float
    area_pct: float


@dataclass
class InspectionResult:
    decision: str                 # "Good" or "Defective"
    threshold: float
    detections: list              # sorted by confidence, highest first
    annotated_rgb: np.ndarray
    image_width: int
    image_height: int
    inference_ms: float
    model_name: str
    model_version: str

    @property
    def n_defects(self):
        return len(self.detections)

    @property
    def top_detection(self):
        return self.detections[0] if self.detections else None


def decode_image(data, filename=""):
    """Validate raw bytes and return a BGR image (numpy array)."""
    if filename:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise InvalidImageError(f"Unsupported file type '{ext}'. Use JPG, PNG or BMP.")
    if not data:
        raise InvalidImageError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise InvalidImageError("The file is larger than 10 MB.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError("The file could not be read as an image (it may be corrupt).")
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE:
        raise InvalidImageError(f"The image is too small ({w}x{h}).")
    return img


class DefectDetector:
    def __init__(self, cfg=None):
        cfg = cfg or load_config()
        inf = cfg["inference"]
        self.model_path = Path(inf["model_path"])
        self.threshold = float(inf["conf_threshold"])
        self.device = inf["device"]
        self.model_name = inf["model_name"]
        self.model_version = inf["model_version"]
        self.imgsz = cfg["preprocessing"]["imgsz"]
        if not self.model_path.exists():
            raise ModelNotFoundError(f"Model file not found: {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def inspect(self, image_bgr, threshold=None):
        t = self.threshold if threshold is None else float(threshold)
        if not 0.0 < t < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        h, w = image_bgr.shape[:2]
        t0 = time.perf_counter()
        r = self.model.predict(image_bgr, imgsz=self.imgsz, conf=t,
                               device=self.device, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000

        dets = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            dets.append(Detection(
                defect_type=self.model.names[int(b.cls)], confidence=round(float(b.conf), 4),
                x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1),
                area_px=round(area, 1), area_pct=round(100 * area / (w * h), 2)))
        dets.sort(key=lambda d: -d.confidence)

        return InspectionResult(
            decision="Defective" if dets else "Good", threshold=t, detections=dets,
            annotated_rgb=cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB),
            image_width=w, image_height=h, inference_ms=round(ms, 1),
            model_name=self.model_name, model_version=self.model_version)

    def inspect_bytes(self, data, filename="", threshold=None):
        return self.inspect(decode_image(data, filename), threshold)

    def inspect_path(self, path, threshold=None):
        path = Path(path)
        if not path.exists():
            raise InvalidImageError(f"File not found: {path}")
        return self.inspect_bytes(path.read_bytes(), path.name, threshold)


if __name__ == "__main__":
    # quick manual test:  python -m src.inference.detector <image path>
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m src.inference.detector <image path>")
    res = DefectDetector().inspect_path(sys.argv[1])
    print(f"Decision: {res.decision} | threshold {res.threshold} | "
          f"{res.n_defects} detection(s) | {res.inference_ms} ms | "
          f"image {res.image_width}x{res.image_height} | model {res.model_name} ({res.model_version})")
    for d in res.detections:
        print(f"  {d.defect_type:16s} conf={d.confidence:.2f} "
              f"box=({d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f}) area={d.area_px:.0f}px ({d.area_pct}%)")
    out = Path("reports/figures/detector_demo.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(res.annotated_rgb, cv2.COLOR_RGB2BGR))
    print("Saved:", out)