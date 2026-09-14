# AI service

The FastAPI layer accepts uploads and exposes the existing end-to-end analysis
as asynchronous jobs. Each job is isolated under `outputs/api_jobs/<job-id>`.
The default executor runs one analysis at a time to avoid loading competing
models onto the same GPU.

## Install and run

```bash
python -m pip install -e ".[api]"
badminton-ai-api
```

The upload and result interface is available at `http://127.0.0.1:8000/`.
Interactive API documentation is at `http://127.0.0.1:8000/docs` and the
readiness endpoint is `GET /health`.

Create an analysis with multipart form data:

```bash
curl -X POST http://127.0.0.1:8000/v1/analyses \
  -F "video=@stroke.mp4" \
  -F "view=front" \
  -F "handedness=right"
```

Technique defaults to `auto`: the accepted ST-GCN++ prediction selects the
matching assessment rules. A client may still submit `forehand_clear` or
`backhand_drive` as a manual fallback when classification is unavailable or
uncertain. Camera view and racket hand remain explicit because the current
action classifier does not predict them.

Poll the returned `status_url`. When `status` is `succeeded`, read
`result_url` for the complete JSON response or use an entry in
`artifact_urls` to download a report or preview.

## Configuration

All settings are optional environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `AI_CLASSIFIER_HOST` | `127.0.0.1` | Bind address |
| `AI_CLASSIFIER_PORT` | `8000` | HTTP port |
| `AI_CLASSIFIER_JOBS_ROOT` | `outputs/api_jobs` | Persistent job storage |
| `AI_CLASSIFIER_MAX_UPLOAD_BYTES` | `524288000` | Streaming upload limit |
| `AI_CLASSIFIER_MAX_CONCURRENT_JOBS` | `1` | Local inference concurrency |
| `AI_CLASSIFIER_INFERENCE_TIMEOUT_SECONDS` | `1800` | Per-job timeout |
| `AI_CLASSIFIER_ENABLE_WSL_CLASSIFIER` | `false` | Allow the Windows/WSL classifier bridge |

The classifier bridge is disabled by default because it is machine-specific.
Requests with `run_classifier=true` fail explicitly unless the setting is
enabled.

The service uses `configs/pose/yolov8.yaml` for the complete pipeline. That
profile matches the training input of the current ST-GCN++ checkpoint and is
reused for geometry, so one request performs pose extraction only once.

## Deployment boundary

`CliInferenceBackend` is a compatibility adapter around the current CLI. The
HTTP routes and job store depend only on the `InferenceBackend` protocol. A
future in-process model runner or Redis-backed worker can therefore replace
the adapter without changing clients.

The default bind address is loopback-only and the service does not implement
authentication. Put authentication, TLS, request throttling, and upload-body
limits at an API gateway or reverse proxy before exposing it publicly.

The bundled filesystem store and thread executor are intended for one API
process. Do not start multiple Uvicorn workers against it. For horizontal
scaling, implement the same backend boundary with a shared database/object
store and a durable queue such as Redis plus Celery, RQ, or Arq.
