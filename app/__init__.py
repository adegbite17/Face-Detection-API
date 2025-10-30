from .core.config import settings
from .task import celery_app

# Export the celery app
__all__ = ['celery_app']
