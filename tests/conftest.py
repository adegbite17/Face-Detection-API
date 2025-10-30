import pytest
import os
from app.core.config import settings


@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables"""
    os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:postgres@localhost:5432/test_db"
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
    os.environ["CELERY_BROKER_URL"] = "redis://localhost:6379/1"
    os.environ["CELERY_BACKEND"] = "redis://localhost:6379/2"
    os.environ["ENABLE_CELERY"] = "1"
    os.environ["LOADTEST_MODE"] = "1"
    yield
