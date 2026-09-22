import tensorflow as tf
import numpy as np
import cv2


from app.config import MODEL_INPUT_SIZE, DETECTION_LABELS, MASK_THRESHOLD


class MaskClassifier:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        self.input_size = MODEL_INPUT_SIZE
        self.labels = DETECTION_LABELS
        self.threshold = MASK_THRESHOLD

    def preprocess(self, face_img: np.ndarray):

        face_img = cv2.resize(face_img, self.input_size)
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def predict(self, face_img: np.ndarray):

        processed_img = self.preprocess(face_img)
        prediction = self.model.predict(processed_img, verbose=0)

        if prediction.shape[1] == 1:
            # Binary sigmoid output
            prob = prediction[0][0]
            label_idx = 0 if prob < self.threshold else 1
            confidence = float(1 - prob if label_idx == 0 else prob)
        else:
            # Multi-class softmax output
            label_idx = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][label_idx])

        return self.labels[label_idx], confidence
