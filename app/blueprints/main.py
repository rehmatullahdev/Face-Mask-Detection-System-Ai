from flask import Blueprint, render_template, Response, request, jsonify, current_app

from app.services.camera import VideoCamera
from app.services.upload_service import process_upload
from app.services.logging_service import log_detection, save_webcam_snapshot
from app.models import db, DetectionLog

import os
import time

main_bp = Blueprint('main', __name__)

# Single shared camera instance
camera = VideoCamera()


# ── Page routes ───────────────────────────────────────────────────────────────

@main_bp.route('/')
def index():
    return render_template('index.html')


@main_bp.route('/upload_page')
def upload_page():
    return render_template('upload.html')


# ── Camera control routes ─────────────────────────────────────────────────────

@main_bp.route('/start_feed', methods=['POST'])
def start_feed():
    camera.start()
    return jsonify({"status": "started"})


@main_bp.route('/stop_feed', methods=['POST'])
def stop_feed():
    camera.stop()
    return jsonify({"status": "stopped"})


# ── Upload route ──────────────────────────────────────────────────────────────

@main_bp.route('/upload', methods=['POST'])
def upload_image():
    file = request.files.get('file')
    upload_dir = str(current_app.config['UPLOADS_DIR'])

    result = process_upload(file, camera.pipeline, upload_dir)

    # Service signals errors via a 'code' key
    if "error" in result:
        return jsonify({"error": result["error"]}), result.get("code", 400)

    return jsonify(result)


# ── Webcam streaming ──────────────────────────────────────────────────────────

def _log_webcam_snapshot(frame: bytes, detections: list) -> None:
    stats = camera.pipeline.get_stats(detections)
    if stats["with_mask"] == 0 and stats["without_mask"] == 0:
        return

    filename = save_webcam_snapshot(frame)
    if filename:
        log_detection(
            with_mask=stats["with_mask"],
            without_mask=stats["without_mask"],
            source='webcam',
            image_filename=filename
        )


def _generate_frames(app):
    """Video streaming generator with periodic database snapshot logging."""
    frame_count = 0
    while True:
        frame = camera.get_frame()
        if frame is not None and camera.is_running:
            frame_count += 1
            if frame_count % 30 == 0 and camera.last_detections:
                with app.app_context():
                    _log_webcam_snapshot(frame, camera.last_detections)

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'
        )


@main_bp.route('/video_feed')
def video_feed():
    return Response(
        _generate_frames(current_app._get_current_object()),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
