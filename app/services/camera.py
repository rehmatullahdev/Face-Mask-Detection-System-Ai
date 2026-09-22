import cv2
import numpy as np
import os
from app.config import FACE_MODEL_PATH, MASK_MODEL_PATH
from app.services.mask_detection_pipeline import MaskDetectionPipeline
from app.utils.drawing import draw_all_detections


class VideoCamera:
    def __init__(self):
        self.video = None
        self.is_running = False
        self.last_detections = []

        if not os.path.exists(FACE_MODEL_PATH):
            print(f"Face model not found: {FACE_MODEL_PATH}")
        if not os.path.exists(MASK_MODEL_PATH):
            print(f"Mask model not found: {MASK_MODEL_PATH}")

        print("Initializing Mask Detection Pipeline...")
        self.pipeline = MaskDetectionPipeline(FACE_MODEL_PATH, MASK_MODEL_PATH)

    def start(self):
        if not self.is_running:
            self.video = cv2.VideoCapture(0)
            self.is_running = True

    def stop(self):
        if self.is_running:
            self.is_running = False
            if self.video:
                self.video.release()
                self.video = None
            self.last_detections = []

    def __del__(self):
        self.stop()

    def get_frame(self):
        if not self.is_running or not self.video:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Camera is stopped. Click 'Start Webcam'.",
                        (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()

        success, image = self.video.read()
        if not success:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Failed to grab frame.",
                        (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()

        # Run pipeline
        try:
            result = self.pipeline.process_image(image)
            self.last_detections = result["detections"]
        except Exception as exc:
            print(f"Pipeline processing error: {exc}")
            self.last_detections = []

        # Draw bounding boxes and labels
        image = draw_all_detections(image, self.last_detections)

        ret, jpeg = cv2.imencode('.jpg', image)
        if ret:
            return jpeg.tobytes()
        return None
