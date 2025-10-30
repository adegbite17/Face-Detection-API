from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    redis_url: str
    celery_broker_url: str
    celery_backend: str
    perceptual_cache_ttl: int = 86400
    loadtest_mode: bool = False
    enable_celery: bool = True

    class Config:
        env_file = '.env'

settings = Settings()
