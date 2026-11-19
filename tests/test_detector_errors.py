"""Error-handling tests for src/inference/detector.py (plain script, no pytest needed)."""
import cv2
import numpy as np

from src.inference.detector import (DefectDetector, InvalidImageError,
                                    ModelNotFoundError, decode_image)
from src.utils.config import load_config

passed, failed = 0, 0


def check(name, ok, detail=""):
    global passed, failed
    passed += ok
    failed += (not ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def expect(exc, fn):
    try:
        fn()
    except exc as e:
        return True, f"-> {type(e).__name__}: {e}"
    except Exception as e:                       # wrong kind of error
        return False, f"-> unexpected {type(e).__name__}: {e}"
    return False, "-> no error raised"


def png_bytes(h, w, gray=True):
    img = np.random.randint(0, 255, (h, w) if gray else (h, w, 3), dtype=np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


# ---- invalid inputs must raise InvalidImageError, never crash ----
for name, fn in [
    ("empty file", lambda: decode_image(b"", "a.jpg")),
    ("wrong type (.txt)", lambda: decode_image(b"hello", "notes.txt")),
    ("corrupt .jpg", lambda: decode_image(b"this is not an image", "bad.jpg")),
    ("tiny image 10x10", lambda: decode_image(png_bytes(10, 10), "tiny.png")),
    ("file over 10 MB", lambda: decode_image(b"0" * (11 * 1024 * 1024), "big.jpg")),
]:
    ok, detail = expect(InvalidImageError, fn)
    check(name, ok, detail)

# ---- valid inputs must decode (grayscale, colour, upper-case extension) ----
for name, data, fname in [("grayscale png", png_bytes(200, 200), "g.png"),
                          ("colour png", png_bytes(200, 200, gray=False), "c.png"),
                          ("upper-case extension", png_bytes(200, 200), "X.PNG")]:
    img = decode_image(data, fname)
    check(name, img.ndim == 3 and img.shape[2] == 3, f"-> shape {img.shape}")

# ---- model and detector level ----
cfg = load_config()
cfg_bad = {**cfg, "inference": {**cfg["inference"], "model_path": "models/does_not_exist.pt"}}
ok, detail = expect(ModelNotFoundError, lambda: DefectDetector(cfg_bad))
check("missing model file", ok, detail)

det = DefectDetector()
ok, detail = expect(InvalidImageError, lambda: det.inspect_path("data/yolo/images/val/nope.jpg"))
check("missing image path", ok, detail)
for bad_t in (0, 1.5, -0.2):
    ok, detail = expect(ValueError, lambda t=bad_t: det.inspect(np.zeros((200, 200, 3), np.uint8), t))
    check(f"bad threshold {bad_t}", ok, detail)

# ---- a valid blank image must run and give a Good/Defective answer ----
res = det.inspect(np.full((200, 200, 3), 128, np.uint8))
check("blank grey image runs", res.decision in ("Good", "Defective"),
      f"-> {res.decision}, {res.n_defects} detection(s) at T={res.threshold}")

# ---- changing threshold must not crash and stays consistent (higher T -> not more detections) ----
a = det.inspect_path("data/yolo/images/val/scratches_9.jpg", threshold=0.15)
b = det.inspect_path("data/yolo/images/val/scratches_9.jpg", threshold=0.60)
check("higher T gives no more detections", b.n_defects <= a.n_defects,
      f"-> T=0.15: {a.n_defects}, T=0.60: {b.n_defects}")

print(f"\nPassed: {passed} | Failed: {failed}")