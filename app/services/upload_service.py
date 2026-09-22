import os
import time
import base64

import cv2
import numpy as np

from app.services.logging_service import log_detection
from app.utils.drawing import draw_all_detections


# ── Constants ────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}


# ── Validation helpers ────────────────────────────────────────────────────────

def _allowed_file(filename: str) -> bool:
    """Returns True if the filename has a permitted image extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def validate_upload(file) -> tuple[bool, str]:

    if file is None or file.filename == "":
        return False, "No file selected."
    if not _allowed_file(file.filename):
        return False, (
            "Unsupported file type. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return True, ""


# ── Image helpers ─────────────────────────────────────────────────────────────

def decode_image(file_bytes: bytes) -> np.ndarray | None:
    npimg = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img  


def encode_image_base64(img: np.ndarray) -> str:

    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


# ── Persistence helpers ───────────────────────────────────────────────────────

def generate_filename(prefix: str = "upload") -> str:
    return f"{prefix}_{int(time.time())}.jpg"


def save_annotated_image(img: np.ndarray, upload_dir: str, filename: str) -> str:
   
    filepath = os.path.join(upload_dir, filename)
    if not cv2.imwrite(filepath, img):
        raise OSError(f"cv2.imwrite failed for path: {filepath}")
    return filepath


# ── Main service entry-point ─────────────────────────────────────────────────

def process_upload(file, pipeline, upload_dir: str) -> dict:

    # ── 1. Validate ───────────────────────────────────────────────────────────
    is_valid, error_msg = validate_upload(file)
    if not is_valid:
        return {"error": error_msg, "code": 400}

    # ── 2. Decode ─────────────────────────────────────────────────────────────
    try:
        file_bytes = file.read()
    except Exception as exc:
        return {"error": f"Failed to read uploaded file: {exc}", "code": 500}

    img = decode_image(file_bytes)
    if img is None:
        return {"error": "Could not decode image. Ensure the file is a valid image.", "code": 400}

    # ── 3. Detect ─────────────────────────────────────────────────────────────
    try:
        result = pipeline.process_image(img)
    except Exception as exc:
        return {"error": f"Detection pipeline error: {exc}", "code": 500}

    detections = result["detections"]
    stats = result["stats"]

    # ── 4. Draw ───────────────────────────────────────────────────────────────
    annotated = draw_all_detections(img.copy(), detections)

    # ── 5. Save ───────────────────────────────────────────────────────────────
    filename = generate_filename("upload")
    try:
        save_annotated_image(annotated, upload_dir, filename)
    except OSError as exc:
        return {"error": str(exc), "code": 500}

    # ── 6. Log to DB ──────────────────────────────────────────────────────────
    log_detection(
        with_mask=stats["with_mask"],
        without_mask=stats["without_mask"],
        source="upload",
        image_filename=filename
    )

    # ── 7. Build response ─────────────────────────────────────────────────────
    img_base64 = encode_image_base64(annotated)

    return {
        "status": "success",
        "image": img_base64,
        "detections": len(detections),
        "with_mask": stats["with_mask"],
        "without_mask": stats["without_mask"]
    }
