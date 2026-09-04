from mtcnn import MTCNN
import cv2
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class MTCNNFaceDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces and return bounding boxes + landmarks"""
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            faces = self.detector.detect_faces(image_rgb)
            logger.info(f"🔍 Detected {len(faces)} faces")

            return self._format_results(faces)
        except Exception as e:
            logger.error(f"MTCNN detection failed: {e}")
            return []

    def _format_results(self, faces: List[Dict]) -> List[Dict]:
        formatted = []
        for face in faces:
            formatted.append({
                "box": {
                    "x": int(face['box'][0]),
                    "y": int(face['box'][1]),
                    "width": int(face['box'][2]),
                    "height": int(face['box'][3])
                },
                "confidence": float(face['confidence']),
                "landmarks": {
                    "left_eye": [int(face['keypoints']['left_eye'][0]),
                                 int(face['keypoints']['left_eye'][1])],
                    "right_eye": [int(face['keypoints']['right_eye'][0]),
                                  int(face['keypoints']['right_eye'][1])],
                    "nose": [int(face['keypoints']['nose'][0]),
                             int(face['keypoints']['nose'][1])],
                    "mouth_left": [int(face['keypoints']['mouth_left'][0]),
                                   int(face['keypoints']['mouth_left'][1])],
                    "mouth_right": [int(face['keypoints']['mouth_right'][0]),
                                    int(face['keypoints']['mouth_right'][1])]
                }
            })
        return formatted
