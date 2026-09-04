from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from celery.result import AsyncResult
from PIL import Image, ImageDraw
import io
import logging
from typing import List

from app.task import detect_faces_mtcnn, celery_app
from app.utils import b64_to_pil
import numpy as np

router = APIRouter(prefix="/api/v1/detection", tags=["Face Detection"])
logger = logging.getLogger(__name__)


class MTCNNRequest(BaseModel):
    image: str
    min_confidence: float = 0.7


class MTCNNResponse(BaseModel):
    task_id: str
    status: str


class FrontalCropRequest(BaseModel):
    image: str


class FrontalCropResponse(BaseModel):
    job_id: str
    status: str


@router.post("/mtcnn/submit", response_model=MTCNNResponse)
async def submit_mtcnn_detection(request: MTCNNRequest):
    """Submit async MTCNN face detection task"""
    try:
        payload = {
            "image": request.image,
            "min_confidence": request.min_confidence
        }
        task = detect_faces_mtcnn.delay(payload)

        return MTCNNResponse(
            task_id=task.id,
            status="pending"
        )
    except Exception as e:
        logger.error(f"Failed to submit MTCNN task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mtcnn/status/{task_id}")
async def get_mtcnn_status(task_id: str):
    """Check status of MTCNN detection task"""
    try:
        result = AsyncResult(task_id, app=celery_app)  # ← Fixed: use imported celery_app

        if result.ready():
            return {
                "status": "completed",
                "result": result.get()
            }

        return {
            "status": "pending",
            "task_id": task_id
        }
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mtcnn/visualize/{task_id}")
async def visualize_detection(task_id: str):
    """Return image with face bounding boxes drawn on it"""
    try:
        result = AsyncResult(task_id, app=celery_app)

        if result.state != 'SUCCESS':
            raise HTTPException(
                status_code=400,
                detail=f"Task not completed. Current status: {result.state}"
            )

        data = result.get()
        if data.get('status') != 'success':
            raise HTTPException(
                status_code=400,
                detail=f"Detection failed: {data.get('error', 'Unknown error')}"
            )

        faces = data.get('faces', [])
        original_image_b64 = data.get('original_image')  # ← Get stored image

        if not original_image_b64:
            raise HTTPException(
                status_code=404,
                detail="Original image not found in task result"
            )

        # Decode image
        img = b64_to_pil(original_image_b64)
        draw = ImageDraw.Draw(img)

        # Draw bounding boxes and landmarks
        for i, face in enumerate(faces):
            box = face['box']
            confidence = face['confidence']

            # Draw bounding box (red rectangle)
            draw.rectangle(
                [(box['x'], box['y']),
                 (box['x'] + box['width'], box['y'] + box['height'])],
                outline='red',
                width=3
            )

            # Draw confidence label
            draw.text(
                (box['x'], box['y'] - 10),
                f"Face {i+1}: {confidence:.2%}",
                fill='red'
            )

            # Draw landmarks (facial keypoints)
            landmarks = face.get('landmarks', {})
            for landmark_name, (lx, ly) in landmarks.items():
                # Draw small circles for each landmark
                radius = 5
                draw.ellipse(
                    [(lx - radius, ly - radius),
                     (lx + radius, ly + radius)],
                    fill='blue',
                    outline='yellow'
                )

        # Convert to bytes
        img_bytes_io = io.BytesIO()
        img.save(img_bytes_io, format='PNG')
        img_bytes = img_bytes_io.getvalue()

        return Response(content=img_bytes, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mtcnn/batch")
async def batch_detect(images: List[str]):
    """Process multiple images in parallel"""
    try:
        tasks = []
        for img_b64 in images:
            task = detect_faces_mtcnn.delay({"image": img_b64})
            tasks.append({
                "task_id": task.id,
                "status": "pending"
            })

        return {
            "tasks": tasks,
            "total": len(tasks)
        }
    except Exception as e:
        logger.error(f"Batch submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/frontal/crop/submit", response_model=FrontalCropResponse)
async def submit_frontal_crop(request: FrontalCropRequest):
    """Submit frontal crop processing task with complete face boundary extraction"""
    try:
        import uuid
        job_id = str(uuid.uuid4())

        payload = {
            "image": request.image
        }

        # Import the task
        from app.task import process_frontal_crop

        # Submit task with job_id
        task = process_frontal_crop.delay(job_id, payload)

        return FrontalCropResponse(
            job_id=job_id,
            status="pending"
        )
    except Exception as e:
        logger.error(f"Failed to submit frontal crop task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/frontal/crop/status/{job_id}")
async def get_crop_status(job_id: str):
    """Get status and results of frontal crop processing"""
    try:
        # Initialize cache
        from app.services.cache import Cache
        from app.core.config import settings

        cache = Cache(settings.database_url, use_async=False)
        cache.init_db_sync()

        # Get result from cache
        result = cache.get_sync(f"job:{job_id}")

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found. It may have expired or never existed."
            )

        return {
            "job_id": job_id,
            "status": result.get("status"),
            "svg": result.get("svg"),
            "mask_contours": result.get("mask_contours", {}),
            "faces": result.get("faces", []),
            "total_faces": result.get("total_faces", 0),
            "message": result.get("message"),
            "error": result.get("error")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get crop status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/frontal/crop/svg/{job_id}")
async def get_crop_svg(job_id: str):
    """Return the SVG visualization of face boundaries"""
    try:
        from app.services.cache import Cache
        from app.core.config import settings

        cache = Cache(settings.database_url, use_async=False)
        cache.init_db_sync()

        result = cache.get_sync(f"job:{job_id}")

        if not result:
            raise HTTPException(status_code=404, detail="Job not found")

        if result.get("status") != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Job not completed. Status: {result.get('status')}"
            )

        svg_content = result.get("svg")
        if not svg_content:
            raise HTTPException(status_code=404, detail="SVG not available")

        return Response(content=svg_content, media_type="image/svg+xml")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get SVG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/frontal/crop/visualize/{job_id}")
async def visualize_crop(job_id: str):
    """Return original image with face boundaries overlaid"""
    try:
        from app.services.cache import Cache
        from app.core.config import settings
        import cv2

        cache = Cache(settings.database_url, use_async=False)
        cache.init_db_sync()

        result = cache.get_sync(f"job:{job_id}")

        if not result:
            raise HTTPException(status_code=404, detail="Job not found")

        if result.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Job not completed")

        original_image_b64 = result.get("original_image")
        mask_contours = result.get("mask_contours", {})

        if not original_image_b64:
            raise HTTPException(status_code=404, detail="Original image not found")

        # Decode image
        img_pil = b64_to_pil(original_image_b64)
        img_np = np.array(img_pil.convert('RGB'))

        # Draw contours
        for face_key, points in mask_contours.items():
            pts = np.array([[p['x'], p['y']] for p in points], dtype=np.int32)
            cv2.polylines(img_np, [pts], True, (0, 255, 0), 2)

        # Convert back to PIL and return
        result_img = Image.fromarray(img_np)
        img_bytes_io = io.BytesIO()
        result_img.save(img_bytes_io, format='PNG')
        img_bytes = img_bytes_io.getvalue()

        return Response(content=img_bytes, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to visualize crop: {e}")
        raise HTTPException(status_code=500, detail=str(e))
