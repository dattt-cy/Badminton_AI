"""Job lifecycle and bounded background execution."""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from ai_classifier.inference.backend import InferenceBackend

from .job_store import JobStore
from .schemas import AnalysisOptions, JobRecord, JobStatus

LOGGER = logging.getLogger(__name__)

ARTIFACT_NAMES = frozenset(
    {
        "analysis.json",
        "classification.json",
        "pose.npz",
        "pose_viewer.json",
        "pose_preview.mp4",
        "quality.json",
        "technique_report.json",
        "user_report.json",
        "user_report.md",
    }
)


class AnalysisJobService:
    def __init__(
        self,
        store: JobStore,
        backend: InferenceBackend,
        *,
        max_workers: int = 1,
    ) -> None:
        self.store = store
        self.backend = backend
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="inference"
        )
        self._futures: dict[str, Future[None]] = {}

    def prepare(
        self, options: AnalysisOptions, *, input_suffix: str
    ) -> tuple[JobRecord, Path]:
        filename = f"input{input_suffix.lower()}"
        record, job_dir = self.store.create(options, input_filename=filename)
        return record, job_dir / filename

    def submit(self, job_id: str) -> None:
        future = self._executor.submit(self._execute, job_id)
        self._futures[job_id] = future
        future.add_done_callback(lambda _: self._futures.pop(job_id, None))

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def artifact_path(self, job_id: str, name: str) -> Path | None:
        if name not in ARTIFACT_NAMES:
            return None
        candidate = self.store.job_dir(job_id) / "artifacts" / name
        return candidate if candidate.is_file() else None

    def _execute(self, job_id: str) -> None:
        record = self.store.update(job_id, status=JobStatus.RUNNING)
        job_dir = self.store.job_dir(job_id)
        output_dir = job_dir / "artifacts"
        output_dir.mkdir(parents=False, exist_ok=True)
        try:
            self.backend.analyze(
                input_path=job_dir / record.input_filename,
                output_dir=output_dir,
                options=record.options,
            )
            artifacts = sorted(
                path.name
                for path in output_dir.iterdir()
                if path.is_file() and path.name in ARTIFACT_NAMES
            )
            self.store.update(
                job_id, status=JobStatus.SUCCEEDED, artifacts=artifacts
            )
        except Exception as exc:  # boundary: persist all worker failures
            LOGGER.exception("Analysis job %s failed", job_id)
            self.store.update(
                job_id,
                status=JobStatus.FAILED,
                error=str(exc)[:4000] or type(exc).__name__,
            )
