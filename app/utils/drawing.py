import cv2
import numpy as np


''' responsible for drawing a single bounding box and its
 corresponding label on the image.'''

def draw_detection(image: np.ndarray, bbox, label: str, confidence: float):
    """
    Draws a bounding box and label on the image.
    """
    x, y, w, h = bbox
    
    # Define color based on label
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    
    # Draw Bounding Box
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    
    # Prepare label text
    text = f"{label} ({confidence*100:.1f}%)"
    
    # Get text size for background rectangle
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    # Draw background rectangle for text
    cv2.rectangle(image, (x, y - text_h - 10), (x + text_w, y), color, -1)
    
    # Draw text
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image

def draw_all_detections(image: np.ndarray, detections: list):

    for det in detections:
        image = draw_detection(image, det["bbox"], det["label"], det["confidence"])
    return image
