import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from app.api import router as api_router
from app.core.logging_setup import setup_logging
from app.services.cache import Cache
from app.core.config import settings
from .v1.endpoints import detection
from app.v1.endpoints.detection import router as detection_router

logger = setup_logging()

app = FastAPI(title="Image Processing API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)
app.include_router(detection_router)

# mount metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get("/")
def read_root():
    return {"message": "Image Processing API"}

@app.on_event("startup")
async def startup_event():
    # create DB table
    try:
        cache = Cache(settings.database_url)
        try:
            await cache.init_db()
            logger.info("Cache table initialized (async).")
        except AttributeError:
            # Cache might only have sync init, running it in threadpool might be needed
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, cache.init_db_sync)
            logger.info("Cache table initialized (sync fallback).")
    except Exception as e:
        logger.exception("Failed to init cache DB on startup: %s", e)

@app.get("/", tags=["health"])
async def root():
    return {"ok": True, "version": "0.1.0"}
