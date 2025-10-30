import os
import time
import logging
from celery import Celery
from .core.config import settings
from .services.cache import Cache
from .services.image_processing import process_request

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
            from app.services.image_processing import compute_phash_from_b64
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


@celery_app.task
def health_check():
    return {"status": "healthy", "timestamp": time.time()}
