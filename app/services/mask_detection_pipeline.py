import numpy as np
import cv2
from app.detectors.face_detector import FaceDetector
from app.detectors.mask_classifier import MaskClassifier


class MaskDetectionPipeline:
    def __init__(self, face_model_path: str, mask_model_path: str):
        self.face_detector = FaceDetector(face_model_path)
        self.mask_classifier = MaskClassifier(mask_model_path)

    def get_stats(self, detections: list):

        stats = {"with_mask": 0, "without_mask": 0}
        for det in detections:
            if "No Mask" in det["label"]:
                stats["without_mask"] += 1
            else:
                stats["with_mask"] += 1
        return stats


    def process_image(self, image_np: np.ndarray):

        results = self.face_detector.detect(image_np)
        detections = []

        if results.detections:
            ih, iw, _ = image_np.shape
            for detection in results.detections:
                bbox = detection.bounding_box
                x = int(bbox.origin_x)
                y = int(bbox.origin_y)
                w = int(bbox.width)
                h = int(bbox.height)

                # Clip to image boundaries
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(iw, x + w), min(ih, y + h)

                face_crop = image_np[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                label, confidence = self.mask_classifier.predict(face_crop)
                detections.append({
                    "bbox": (x, y, w, h),
                    "label": label,
                    "confidence": confidence
                })

        return {
            "detections": detections,
            "stats": self.get_stats(detections)
        }
