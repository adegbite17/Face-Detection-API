# Qooves FastAPI Take-home

Run locally:
1. use  .env as environment variables
2. docker-compose up --build
3. POST to `http://localhost:8000/api/v1/frontal/crop/submit` with payload:
   {
     "image": "<base64 image>",
     "landmarks":  [{x: .., y: ...}, {x: ..., y: ...}, ...],
     "segmentation_map": "<base64 mask>"
   }
4. For async mode you'll receive `{"id": <int>, "status":"pending"}`. Query `GET /api/v1/frontal/crop/status/{id}`.

Notes:
- Set `LOAD_TEST=1` to run synchronous fast path (no 20s delay).
- Metrics on `http://localhost:8000/metrics`
- DB cache stores perceptual hash to avoid reprocessing identical images.
