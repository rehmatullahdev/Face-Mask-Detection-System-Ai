import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"

# Model Paths
RESOURCES_DIR = APP_DIR / "resources" / "models"
FACE_MODEL_PATH = str(RESOURCES_DIR / "blaze_face_full_range.tflite")
MASK_MODEL_PATH = str(RESOURCES_DIR / "mask_detector.keras")

# Detection Settings
MIN_FACE_CONFIDENCE = 0.5
MASK_THRESHOLD = 0.6
MODEL_INPUT_SIZE = (224, 224)
DETECTION_LABELS = ["Mask", "No Mask"]

# Flask Configuration
class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "super-secret-admin-key")
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + str(APP_DIR / 'detections.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOADS_DIR = APP_DIR / "static" / "uploads"
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
