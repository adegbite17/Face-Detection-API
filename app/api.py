import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException
from app.schema import SubmitPayload, SubmitResponse, SvgResponse
from app.core.logging_setup import setup_logging
from app.core.config import settings
from prometheus_client import Counter
from app.services.image_processing import process_request
from app.services.cache import Cache
import uuid

logger = setup_logging()
router = APIRouter()
REQUEST_COUNTER = Counter("ooves_requests_total", "Total requests to ooves API")

# Threadpool for CPU-bound synchronous processing
EXECUTOR = ThreadPoolExecutor(max_workers=3)

# import celery if enabled
if getattr(settings, "enable_celery", True):
    from app.task import run_processing

# In-memory job tracking
JOB_MAP = {}


@router.post("/api/v1/frontal/crop/submit", response_model=SubmitResponse)
async def submit(payload: SubmitPayload):
    """
    Submit an image processing request.
    Returns either immediate result or job ID if using Celery.
    """
    REQUEST_COUNTER.inc()
    logger.info("Received processing request")

    try:
        if getattr(settings, "enable_celery", True):
            # Celery processing path
            payload_dict = {
                "image": payload.image,
                "landmarks": [landmark.to_tuple() for landmark in payload.landmarks],
                "segmentation_map": payload.segmentation_map
            }

            try:
                task = run_processing.delay(payload_dict)
                task_id = str(task.id)

                # generate a more manageable job ID
                job_id = int(uuid.UUID(task_id).int % (2 ** 31))

                # store both IDs for lookup
                JOB_MAP[str(job_id)] = task_id
                JOB_MAP[task_id] = {
                    "status": "pending",
                    "job_id": job_id,
                    "task": task
                }

                logger.info(f"Created Celery task {task_id} with job ID {job_id}")
                return SubmitResponse(id=job_id, status="pending")

            except Exception as e:
                logger.exception("Failed to create Celery task")
                raise HTTPException(
                    status_code=500,
                    detail=f"Task creation failed: {str(e)}"
                )
        else:
            # Synchronous processing path
            try:
                sync_cache = Cache(settings.database_url, use_async=False)
                sync_cache.init_db_sync()

                def sync_worker():
                    return process_request(
                        payload.image,
                        payload.landmarks,
                        payload.segmentation_map,
                        db_cache=sync_cache
                    )

                loop = asyncio.get_running_loop()
                result, phash = await loop.run_in_executor(EXECUTOR, sync_worker)

                # for sync processing, return immediate result
                return SubmitResponse(status="completed")

            except Exception as e:
                logger.exception("Synchronous processing failed")
                raise HTTPException(
                    status_code=500,
                    detail=f"Processing failed: {str(e)}"
                )

    except Exception as e:
        logger.exception("Submit endpoint failed")
        raise HTTPException(
            status_code=500,
            detail=f"Request failed: {str(e)}"
        )


@router.get("/api/v1/frontal/crop/status/{job_id}", response_model=SvgResponse)
async def status(job_id: str):
    """
    Query job status and return result when available.
    """
    if not getattr(settings, "enable_celery", True):
        raise HTTPException(
            status_code=404,
            detail="Status checking not available in synchronous mode"
        )

    try:
        # get the Celery task ID from the job ID
        task_id = JOB_MAP.get(job_id)
        if not task_id:
            raise HTTPException(status_code=404, detail="Job not found")

        # get task info
        task_info = JOB_MAP.get(task_id)
        if not task_info:
            raise HTTPException(status_code=404, detail="Task information not found")

        task = task_info["task"]

        # check if task is ready
        if not task.ready():
            return SvgResponse(svg="", mask_contours={"1": []})

        # get task result
        result = task.get()

        if result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Unknown processing error")
            )

        # extract the result data
        result_data = result.get("result", {})

        return SvgResponse(
            svg=result_data.get("svg", ""),
            mask_contours={"1": result_data.get("mask_contours", [])}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to check status for job {job_id}")
        raise HTTPException(
            status_code=500,
            detail=f"Status check failed: {str(e)}"
        )
