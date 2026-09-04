import os
import time
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from celery import Celery
from .core.config import settings
from .services.cache import Cache
from image_processing import process_request,  create_face_mask, process_mask_for_contours, extract_mask_contours
from .services.svg_renderer import generate_svg_from_contours, contours_to_mask_contours
from .services.face_detection import MTCNNFaceDetector
from image_processing import compute_phash_from_b64
from .utils import b64_to_pil


# Ensure project root is in Python path

logger = logging.getLogger(__name__)

# read broker/backend from settings
BROKER = getattr(settings, "celery_broker_url", os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"))
BACKEND = getattr(settings, "celery_backend", os.getenv("CELERY_BACKEND", "redis://localhost:6379/2"))

celery_app = Celery('tasks')


class CeleryConfig:
    broker_url = BROKER
    result_backend = BACKEND
    task_serializer = 'json'
    result_serializer = 'json'
    accept_content = ['json']
    enable_utc = True
    task_track_started = True
    worker_disable_rate_limits = True
    task_ignore_result = False
    broker_connection_retry_on_startup = True


celery_app.config_from_object(CeleryConfig)

if __name__ == '__main__':
    celery_app.start()

@celery_app.task
def run_processing(payload):
    """
    Celery task wrapper around the synchronous processing function.
    """
    try:
        logger.info(f"Processing task with payload: {payload.keys()}")

        # Initialize Cache with use_async=False for sync operations
        cache = Cache(settings.database_url, use_async=False)
        cache.init_db_sync()  # Use sync initialization

        # Validate input
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a dictionary")

        image_b64 = payload.get("image")
        landmarks = payload.get("landmarks", [])
        seg_b64 = payload.get("segmentation_map")

        if not image_b64 or not seg_b64:
            raise ValueError("Missing required image data")

        # Check cache first
        try:
            phash = compute_phash_from_b64(image_b64)
            if phash:
                cached_result = cache.get_sync(phash)
                if cached_result:
                    logger.info(f"Cache hit for phash: {phash}")
                    return {
                        "status": "success",
                        "result": cached_result,
                        "phash": phash,
                        "cached": True
                    }
        except Exception as e:
            logger.warning(f"Cache check failed: {str(e)}")

        # Process request
        result, computed_phash = process_request(
            image_b64,
            landmarks,
            seg_b64,
            db_cache=cache
        )

        # Cache the result
        if result and computed_phash:
            try:
                cache.set_sync(computed_phash, result)  # Use set_sync for sync operations
                logger.info(f"Cached result for phash: {computed_phash}")
            except Exception as e:
                logger.warning(f"Failed to cache result: {str(e)}")

        return {
            "status": "success",
            "result": result,
            "phash": computed_phash,
            "cached": False
        }

    except Exception as e:
        logger.exception(f"Task failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


@celery_app.task(name='app.task.detect_faces_mtcnn')
def detect_faces_mtcnn(payload):
    """MTCNN face detection task"""
    try:
        logger.info(f"🚀 Starting MTCNN detection")

        detector = MTCNNFaceDetector()
        image_b64 = payload.get("image")
        min_confidence = payload.get("min_confidence", 0.7)

        if not image_b64:
            raise ValueError("Missing image data")

        # Decode to PIL then numpy
        image_pil = b64_to_pil(image_b64)
        image_np = np.array(image_pil.convert('RGB'))

        logger.info(f"📷 Image shape: {image_np.shape}")

        # Detect faces
        faces = detector.detect_faces(image_np)

        logger.info(f"✓ Detection complete: {len(faces)} faces found")

        return {
            "status": "success",
            "faces": faces,
            "total_faces": len(faces),
            "min_confidence": min_confidence,
            "original_image": image_b64  # ← Store for visualization
        }
    except Exception as e:
        logger.exception(f"❌ MTCNN task failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


@celery_app.task(name='app.task.process_frontal_crop', bind=True)
def process_frontal_crop(self, job_id: str, payload: dict):
    """Process frontal crop with complete face boundary extraction"""
    try:
        logger.info(f"🚀 Starting frontal crop processing for job: {job_id}")

        # Initialize detector and cache
        detector = MTCNNFaceDetector()
        cache = Cache(settings.database_url, use_async=False)
        cache.init_db_sync()

        image_b64 = payload.get("image")
        if not image_b64:
            raise ValueError("Missing image data")

        # Decode image
        image_pil = b64_to_pil(image_b64)
        image_np = np.array(image_pil.convert('RGB'))

        logger.info(f"📷 Image shape: {image_np.shape}")

        # Detect faces
        faces = detector.detect_faces(image_np)
        logger.info(f"✓ Detected {len(faces)} faces")

        if not faces:
            result = {
                "status": "completed",
                "job_id": job_id,
                "svg": None,
                "mask_contours": {},
                "faces": [],
                "message": "No faces detected"
            }
            cache.set_sync(f"job:{job_id}", result)
            return result

        # Create combined mask for all detected faces
        combined_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

        for face in faces:
            face_mask = create_face_mask(image_np.shape, face)
            combined_mask = cv2.bitwise_or(combined_mask, face_mask)

        # Process mask to get complete contours
        processed_mask = process_mask_for_contours(combined_mask)
        contours = extract_mask_contours(processed_mask)

        logger.info(f"✓ Extracted {len(contours)} contours")

        # Generate SVG
        svg_output = generate_svg_from_contours(
            contours,
            image_np.shape[1],
            image_np.shape[0]
        )

        # Generate mask_contours for API
        mask_contours_data = contours_to_mask_contours(contours)

        # Store results in cache
        result = {
            "status": "completed",
            "job_id": job_id,
            "svg": svg_output,
            "mask_contours": mask_contours_data,
            "faces": faces,
            "total_faces": len(faces),
            "original_image": image_b64
        }

        cache.set_sync(f"job:{job_id}", result)
        logger.info(f"✓ Frontal crop processing complete for job: {job_id}")

        return result

    except Exception as e:
        logger.exception(f"❌ Frontal crop task failed: {str(e)}")
        error_result = {
            "status": "error",
            "job_id": job_id,
            "error": str(e)
        }

        try:
            cache = Cache(settings.database_url, use_async=False)
            cache.init_db_sync()
            cache.set_sync(f"job:{job_id}", error_result)
        except:
            pass

        return error_result


@celery_app.task
def health_check():
    return {"status": "healthy", "timestamp": time.time()}
