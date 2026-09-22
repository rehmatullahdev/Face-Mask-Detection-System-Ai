import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2


from app.config import MIN_FACE_CONFIDENCE


class FaceDetector:
    def __init__(self, model_path: str):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=MIN_FACE_CONFIDENCE
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect(self, image_np: np.ndarray):

        # MediaPipe requires RGB format
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object, using a copy to ensure memory safety
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, 
            data=image_rgb.copy()
        )
        
        return self.detector.detect(mp_image)
