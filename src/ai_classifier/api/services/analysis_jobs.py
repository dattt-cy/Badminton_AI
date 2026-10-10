from __future__ import annotations

import json
import shutil
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable
from uuid import uuid4

from ai_classifier.api.schemas.job import AnalysisJob, JobStatus

PipelineRunner = Callable[..., dict[str, Any]]


class JobNotFoundError(KeyError):
    pass


class UploadTooLargeError(ValueError):
    pass


class AnalysisJobService:
    """Stores job metadata on disk and serializes GPU-heavy pipeline execution."""

    def __init__(
        self,
        storage_dir: Path,
        max_upload_bytes: int,
        runner: PipelineRunner | None = None,
    ) -> None:
        self.storage_dir = storage_dir.resolve()
        self.max_upload_bytes = max_upload_bytes
        self._runner = runner
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")
        self._lock = threading.Lock()
        self._futures: set[Future[None]] = set()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def create(
        self,
        source: BinaryIO,
        filename: str,
        player_upper: str,
        player_lower: str,
    ) -> AnalysisJob:
        job_id = uuid4().hex
        job_dir = self.storage_dir / job_id
        job_dir.mkdir(parents=True)
        safe_suffix = Path(filename).suffix.lower() or ".mp4"
        video_path = job_dir / f"input{safe_suffix}"
        self._copy_limited(source, video_path)

        now = datetime.now(timezone.utc)
        job = AnalysisJob(
            id=job_id,
            status=JobStatus.queued,
            filename=Path(filename).name,
            player_upper=player_upper,
            player_lower=player_lower,
            created_at=now,
            updated_at=now,
        )
        self._write_job(job)
        future = self._executor.submit(self._run, job, video_path)
        with self._lock:
            self._futures.add(future)
        future.add_done_callback(self._discard_future)
        return job

    def get(self, job_id: str) -> AnalysisJob:
        metadata_path = self._job_dir(job_id) / "job.json"
        if not metadata_path.is_file():
            raise JobNotFoundError(job_id)
        return AnalysisJob.model_validate_json(metadata_path.read_text(encoding="utf-8"))

    def get_result(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job.status is not JobStatus.completed:
            raise RuntimeError(job.status.value)
        return json.loads(
            (self._job_dir(job_id) / "result.json").read_text(encoding="utf-8")
        )

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        if job.status is JobStatus.processing:
            raise RuntimeError("processing")
        shutil.rmtree(self._job_dir(job_id))

    def _run(self, job: AnalysisJob, video_path: Path) -> None:
        try:
            self._set_status(job, JobStatus.processing)
            runner = self._runner or self._default_runner()
            result_path = video_path.parent / "result.json"
            runner(
                video_path=video_path,
                output_path=result_path,
                player_upper=job.player_upper,
                player_lower=job.player_lower,
            )
            self._set_status(
                job,
                JobStatus.completed,
                result_url=f"/api/v1/analyses/{job.id}/result",
            )
        except Exception as exc:  # the error is surfaced through job status
            self._set_status(job, JobStatus.failed, error=str(exc))

    @staticmethod
    def _default_runner() -> PipelineRunner:
        from run_pipeline import analyze_badminton_video

        return analyze_badminton_video

    def _copy_limited(self, source: BinaryIO, destination: Path) -> None:
        total = 0
        try:
            with destination.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > self.max_upload_bytes:
                        raise UploadTooLargeError(
                            f"Upload exceeds {self.max_upload_bytes} bytes"
                        )
                    target.write(chunk)
        except Exception:
            shutil.rmtree(destination.parent, ignore_errors=True)
            raise

    def _set_status(
        self,
        job: AnalysisJob,
        status: JobStatus,
        *,
        error: str | None = None,
        result_url: str | None = None,
    ) -> None:
        updated = job.model_copy(
            update={
                "status": status,
                "updated_at": datetime.now(timezone.utc),
                "error": error,
                "result_url": result_url,
            }
        )
        self._write_job(updated)

    def _write_job(self, job: AnalysisJob) -> None:
        path = self.storage_dir / job.id / "job.json"
        temporary = path.with_suffix(".tmp")
        with self._lock:
            temporary.write_text(job.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(path)

    def _job_dir(self, job_id: str) -> Path:
        if not job_id.isalnum():
            raise JobNotFoundError(job_id)
        return self.storage_dir / job_id

    def _discard_future(self, future: Future[None]) -> None:
        with self._lock:
            self._futures.discard(future)

