# Face Processing API

An async face detection and segmentation pipeline built with FastAPI, Celery, and PostgreSQL. Accepts a frontal face image with landmarks and a segmentation map, extracts per-region contours, and returns an SVG overlay with mask coordinates.

## Architecture

Client → FastAPI → Celery Worker → PostgreSQL (phash cache)
↓
Redis (broker + backend)
↓
MTCNN + OpenCV pipeline

- **FastAPI** — async HTTP layer, two route groups (`/api/v1/frontal` and `/api/v1/detection`)
- **Celery + Redis** — offloads CPU-bound vision work off the request thread
- **PostgreSQL** — perceptual hash cache (24h TTL) to skip reprocessing identical images
- **MTCNN** — face detection with bounding boxes and 5-point landmarks
- **OpenCV** — contour extraction per segmentation region, Gaussian smoothing, SVG generation
- **Prometheus** — request metrics at `/metrics`

## Endpoints

### Frontal Crop Pipeline
POST /api/v1/frontal/crop/submit
GET /api/v1/frontal/crop/status/{id}

Accepts image + 68-point landmarks + segmentation map (PNG, label-indexed). Returns SVG overlay and per-region mask contours.

### Face Detection
POST /api/v1/detection/mtcnn/submit
GET /api/v1/detection/mtcnn/status/{task_id}
GET /api/v1/detection/mtcnn/visualize/{task_id}
POST /api/v1/detection/mtcnn/batch
MTCNN-based face detection returning bounding boxes, 5-point landmarks, and confidence scores. Visualize endpoint returns the original image with boxes and keypoints drawn.

### Frontal Crop (image-only)
POST /api/v1/detection/frontal/crop/submit
GET /api/v1/detection/frontal/crop/status/{job_id}
GET /api/v1/detection/frontal/crop/svg/{job_id}
GET /api/v1/detection/frontal/crop/visualize/{job_id}

Detects faces via MTCNN, creates elliptical face masks, extracts contours, returns SVG and mask data stored in PostgreSQL.

## Running Locally

**Prerequisites:** Docker and Docker Compose

```bash
# 1. Copy environment file
cp .env.example .env   # or use the provided .env

# 2. Build and start all services
docker-compose up --build
Services started:

app — FastAPI + Celery worker on port 8000

db — PostgreSQL 16 on port 5432

redis — Redis 7 on port 6379

prometheus — Prometheus on port 9090

Interactive API docs: http://localhost:8000/docs
Metrics: http://localhost:8000/metrics

### Request Format

POST /api/v1/frontal/crop/submit
{
  "image": "<base64 PNG>",
  "landmarks": [{"x": 120.5, "y": 340.2}, ...],
  "segmentation_map": "<base64 PNG, label-indexed>"
}
### Response
{"id": 1034347484, "status": "pending"}

### Poll status
GET /api/v1/frontal/crop/status/1034347484

{
  "svg": "<base64 SVG>",
  "mask_contours": {
    "region_1": [[x, y], ...],
    "region_2": [[x, y], ...]
  }
}

Segmentation Map Requirements
The segmentation map should be a lossless PNG with label-indexed pixel values (not RGB). Each pixel value corresponds to a face region:

Value	Region
0	Background
1	Skin / face
2	Left eyebrow
3	Right eyebrow
4	Left eye
5	Right eye
6	Nose
7–9	Lips / mouth
10–11	Ears
12+	Hair / neck
JPEG segmentation maps might be corrupt label values and produce incomplete results.

### Configuration
Variable	Description	Default
DATABASE_URL	PostgreSQL connection string	required
REDIS_URL	Redis URL	required
CELERY_BROKER_URL	Celery broker	required
CELERY_BACKEND	Celery result backend	required
ENABLE_CELERY	1 = async, 0 = sync	1
LOADTEST_MODE	1 = skip artificial delay	0
PERCEPTUAL_CACHE_TTL	Cache TTL in seconds	86400

## Running Tests
# Requires Redis running locally
pytest tests/

Tech Stack
Python 3.11

FastAPI 0.115 / Uvicorn

Celery 5.3 / Redis 7

PostgreSQL 16 / SQLAlchemy 2 / Asyncpg

OpenCV 4.8 / MTCNN / Pillow / ImageHash

Prometheus Client

Docker Compose


