import asyncio
from fastapi import FastAPI
from prometheus_client import make_asgi_app
from app.api import router as api_router
from app.core.logging_setup import setup_logging
from app.services.cache import Cache
from app.core.config import settings

logger = setup_logging()

app = FastAPI(title="ooves Face Crop API", version="0.1.0")
app.include_router(api_router)

# mount metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.on_event("startup")
async def startup_event():
    # create DB table
    try:
        cache = Cache(settings.database_url)
        try:
            await cache.init_db()
            logger.info("Cache table initialized (async).")
        except AttributeError:
            # Cache might only have sync init; run it in threadpool might be needed
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, cache.init_db_sync)
            logger.info("Cache table initialized (sync fallback).")
    except Exception as e:
        logger.exception("Failed to init cache DB on startup: %s", e)

@app.get("/", tags=["health"])
async def root():
    return {"ok": True, "version": "0.1.0"}
