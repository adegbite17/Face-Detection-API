import pytest
from fastapi.testclient import TestClient
from app.main import app
import base64
from PIL import Image
import io
import redis
import time
import json
import logging
from app.task import celery_app
from celery.backends.redis import RedisBackend

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

client = TestClient(app)


def configure_celery():
    """Configure Celery with proper settings"""
    celery_app.conf.update(
        result_backend='redis://localhost:6379/1',
        broker_url='redis://localhost:6379/1',
        redis_backend_use_ssl=False,
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        enable_utc=True,
        result_expires=3600,
        task_track_started=True,
        broker_connection_retry_on_startup=True,
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        task_ignore_result=False,
        task_store_errors_even_if_ignored=True
    )


def get_redis_client(max_retries=3):
    for attempt in range(max_retries):
        try:
            client = redis.Redis(
                host='localhost',
                port=6379,
                db=1,
                decode_responses=False,
                socket_timeout=5
            )
            client.ping()
            return client
        except (redis.ConnectionError, ConnectionRefusedError) as e:
            logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)


def is_redis_available():
    try:
        client = get_redis_client()
        return client.ping()
    except Exception as e:
        logger.error(f"Redis availability check failed: {e}")
        return False


def verify_celery_config():
    try:
        logger.info("Checking Celery configuration:")
        logger.info(f"Broker URL: {celery_app.conf.broker_url}")
        logger.info(f"Result Backend: {celery_app.conf.result_backend}")

        i = celery_app.control.inspect(timeout=3)
        availability = i.ping()
        logger.info(f"Worker ping result: {availability}")
        return bool(availability)
    except Exception as e:
        logger.error(f"Failed to verify Celery config: {e}")
        return False


def check_celery_worker_status():
    try:
        i = celery_app.control.inspect(timeout=3)
        active = i.active()
        reserved = i.reserved()
        scheduled = i.scheduled()
        registered = i.registered()

        if active or reserved or scheduled:
            return True

        if registered:
            logger.info("Worker is registered but idle")
            return True

        logger.warning("No active Celery workers found")
        return False
    except Exception as e:
        logger.error(f"Error checking Celery status: {e}")
        return False


requires_redis = pytest.mark.skipif(
    not is_redis_available(),
    reason="Redis is not available"
)


@pytest.fixture(scope="session", autouse=True)
def verify_celery():
    configure_celery()
    assert verify_celery_config(), "Celery is not properly configured"


@pytest.fixture(scope="session")
def redis_client():
    if is_redis_available():
        client = get_redis_client()
        logger.info("Redis connection established")
        yield client
        client.flushall()
        logger.info("Redis flushed after tests")
    else:
        logger.error("Redis connection failed")
        yield None


@pytest.fixture
def mock_image():
    img = Image.new('RGBA', (100, 100), 'white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode()


@pytest.fixture
def mock_payload(mock_image):
    return {
        "image": mock_image,
        "landmarks": [{"x": i, "y": i} for i in range(68)],
        "segmentation_map": mock_image
    }


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "version": "0.1.0"}


@requires_redis
def test_direct_task_execution(mock_payload):
    """Test direct task execution without API"""
    from app.task import run_processing
    result = run_processing(mock_payload)
    assert result['status'] == 'success'


@requires_redis
def test_redis_connection(redis_client):
    """Verify Redis connection is working"""
    assert redis_client is not None
    test_key = b"test_connection"
    test_value = b"test_value"
    redis_client.set(test_key, test_value)
    assert redis_client.get(test_key) == test_value
    logger.info("Redis connection test passed")


@requires_redis
def test_celery_worker_health():
    """Verify Celery worker is running and healthy"""
    assert check_celery_worker_status(), "No active Celery workers found"


@requires_redis
def test_celery_health():
    """Verify Celery is working with a simple task"""
    from app.task import health_check
    result = health_check.delay()
    assert result.get(timeout=10)['status'] == 'healthy'
