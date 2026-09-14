"""FastAPI application factory."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from ai_classifier.inference.backend import CliInferenceBackend, InferenceBackend

from .config import ApiSettings
from .job_store import JobNotFoundError, JobStore
from .schemas import (
    AnalysisOptions,
    CameraView,
    Handedness,
    HealthResponse,
    JobAccepted,
    JobRecord,
    JobResponse,
    JobStatus,
    SegmentationMode,
    TargetPlayer,
    Technique,
)
from .service import AnalysisJobService

VIDEO_EXTENSIONS = frozenset({".avi", ".mkv", ".mov", ".mp4"})
UPLOAD_CHUNK_BYTES = 1024 * 1024


def create_app(
    settings: ApiSettings | None = None,
    *,
    backend: InferenceBackend | None = None,
) -> FastAPI:
    settings = settings or ApiSettings.from_env()
    store = JobStore(settings.jobs_root)
    backend = backend or CliInferenceBackend(
        settings.repo_root,
        timeout_seconds=settings.inference_timeout_seconds,
        enable_wsl_classifier=settings.enable_wsl_classifier,
    )
    service = AnalysisJobService(
        store, backend, max_workers=settings.max_concurrent_jobs
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        store.mark_interrupted_jobs_failed()
        yield
        service.shutdown()

    app = FastAPI(
        title="Badminton AI Classifier",
        version="1.0.0",
        description="Asynchronous phone-video technique analysis API.",
        lifespan=lifespan,
    )
    app.state.job_service = service
    app.state.settings = settings

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            accepting_jobs=True,
            classifier_enabled=settings.enable_wsl_classifier,
            max_upload_bytes=settings.max_upload_bytes,
        )

    @app.get("/", include_in_schema=False, response_class=FileResponse)
    def web_app() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.post(
        "/v1/analyses",
        response_model=JobAccepted,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["analyses"],
    )
    async def create_analysis(
        request: Request,
        video: Annotated[UploadFile, File(description="Phone video to analyze")],
        technique: Annotated[Technique, Form()] = Technique.AUTO,
        view: Annotated[CameraView, Form()] = CameraView.FRONT,
        handedness: Annotated[Handedness, Form()] = Handedness.RIGHT,
        target: Annotated[TargetPlayer, Form()] = TargetPlayer.SINGLE,
        segmentation_mode: Annotated[SegmentationMode, Form()] = SegmentationMode.SINGLE,
        floor_angle_degrees: Annotated[float, Form(ge=-45.0, le=45.0)] = 0.0,
        swing_peak_frame: Annotated[int | None, Form(ge=0)] = None,
        generate_preview: Annotated[bool, Form()] = True,
        run_classifier: Annotated[bool, Form()] = False,
    ) -> JobAccepted:
        suffix = Path(video.filename or "").suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Supported video extensions: {', '.join(sorted(VIDEO_EXTENSIONS))}",
            )
        options = AnalysisOptions(
            technique=technique,
            view=view,
            handedness=handedness,
            target=target,
            segmentation_mode=segmentation_mode,
            floor_angle_degrees=floor_angle_degrees,
            swing_peak_frame=swing_peak_frame,
            generate_preview=generate_preview,
            run_classifier=run_classifier or technique == Technique.AUTO,
        )
        if options.run_classifier and not settings.enable_wsl_classifier:
            raise HTTPException(
                status_code=422,
                detail="Classifier execution is disabled on this API instance.",
            )
        record, input_path = service.prepare(options, input_suffix=suffix)
        received = 0
        try:
            with input_path.open("xb") as destination:
                while chunk := await video.read(UPLOAD_CHUNK_BYTES):
                    received += len(chunk)
                    if received > settings.max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Video exceeds {settings.max_upload_bytes} bytes.",
                        )
                    destination.write(chunk)
        except HTTPException:
            input_path.unlink(missing_ok=True)
            store.update(
                record.id,
                status=JobStatus.FAILED,
                error="Upload exceeded the configured size limit.",
            )
            raise
        except OSError as exc:
            input_path.unlink(missing_ok=True)
            store.update(record.id, status=JobStatus.FAILED, error="Upload failed.")
            raise HTTPException(status_code=500, detail="Could not store upload.") from exc
        finally:
            await video.close()

        if received == 0:
            store.update(record.id, status=JobStatus.FAILED, error="Upload was empty.")
            raise HTTPException(status_code=422, detail="Uploaded video is empty.")
        service.submit(record.id)
        return JobAccepted(
            id=record.id,
            status=JobStatus.QUEUED,
            status_url=str(request.url_for("get_analysis", job_id=record.id)),
        )

    @app.get(
        "/v1/analyses/{job_id}",
        response_model=JobResponse,
        tags=["analyses"],
    )
    def get_analysis(job_id: str, request: Request) -> JobResponse:
        record = _get_record(store, job_id)
        artifact_urls = {
            name: str(request.url_for("get_artifact", job_id=job_id, name=name))
            for name in record.artifacts
        }
        return JobResponse(
            **record.model_dump(),
            result_url=(
                str(request.url_for("get_result", job_id=job_id))
                if record.status == JobStatus.SUCCEEDED
                else None
            ),
            artifact_urls=artifact_urls,
        )

    @app.get("/v1/analyses/{job_id}/result", tags=["analyses"])
    def get_result(job_id: str) -> dict:
        record = _get_record(store, job_id)
        if record.status != JobStatus.SUCCEEDED:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Analysis is {record.status.value}.",
            )
        result = service.artifact_path(job_id, "analysis.json")
        if result is None:
            raise HTTPException(status_code=500, detail="Result artifact is missing.")
        try:
            return json.loads(result.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail="Result artifact is invalid.") from exc

    @app.get(
        "/v1/analyses/{job_id}/artifacts/{name}",
        name="get_artifact",
        response_class=FileResponse,
        tags=["analyses"],
    )
    def get_artifact(job_id: str, name: str) -> FileResponse:
        _get_record(store, job_id)
        artifact = service.artifact_path(job_id, name)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Artifact not found.")
        return FileResponse(artifact)

    return app


def _get_record(store: JobStore, job_id: str) -> JobRecord:
    try:
        return store.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Analysis job not found.") from exc


app = create_app()
