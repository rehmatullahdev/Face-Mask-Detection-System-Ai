import os
import time
from app.models import db, DetectionLog
from flask import current_app

def log_detection(with_mask: int, without_mask: int, source: str, image_filename: str = None):

    if with_mask == 0 and without_mask == 0:
        return None

    try:
        log = DetectionLog(
            with_mask=with_mask,
            without_mask=without_mask,
            source=source,
            image_filename=image_filename
        )
        db.session.add(log)
        db.session.commit()
        return log
    except Exception as exc:
        db.session.rollback()
        current_app.logger.error(f"Failed to log detection to database: {exc}")
        return None

def save_webcam_snapshot(frame: bytes):
    """
    Saves a webcam frame to the uploads directory.
    """
    try:
        filename = f"webcam_{int(time.time())}.jpg"
        filepath = os.path.join(current_app.config['UPLOADS_DIR'], filename)
        with open(filepath, 'wb') as f:
            f.write(frame)
        return filename
    except Exception as exc:
        current_app.logger.error(f"Failed to save webcam snapshot: {exc}")
        return None
