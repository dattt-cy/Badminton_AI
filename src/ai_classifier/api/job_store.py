"""Small filesystem-backed job store for a single service instance."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .schemas import AnalysisOptions, JobRecord, JobStatus


class JobNotFoundError(KeyError):
    pass


class JobStore:
    """Persist job metadata atomically so results survive API restarts."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def create(
        self, options: AnalysisOptions, *, input_filename: str
    ) -> tuple[JobRecord, Path]:
        now = datetime.now(UTC)
        job_id = uuid4().hex
        job_dir = self.root / job_id
        job_dir.mkdir(parents=False, exist_ok=False)
        record = JobRecord(
            id=job_id,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            options=options,
            input_filename=input_filename,
        )
        self._write(record)
        return record, job_dir

    def get(self, job_id: str) -> JobRecord:
        path = self.job_dir(job_id) / "job.json"
        try:
            payload = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise JobNotFoundError(job_id) from exc
        return JobRecord.model_validate_json(payload)

    def update(
        self,
        job_id: str,
        *,
        status: JobStatus,
        error: str | None = None,
        artifacts: list[str] | None = None,
    ) -> JobRecord:
        with self._lock:
            current = self.get(job_id)
            updated = current.model_copy(
                update={
                    "status": status,
                    "updated_at": datetime.now(UTC),
                    "error": error,
                    "artifacts": current.artifacts if artifacts is None else artifacts,
                }
            )
            self._write(updated)
            return updated

    def mark_interrupted_jobs_failed(self) -> None:
        """Fail unfinished jobs; the local executor cannot resume subprocesses."""
        for metadata_path in self.root.glob("*/job.json"):
            try:
                record = JobRecord.model_validate_json(
                    metadata_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if record.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                self.update(
                    record.id,
                    status=JobStatus.FAILED,
                    error="Service restarted before the analysis completed.",
                )

    def job_dir(self, job_id: str) -> Path:
        if len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id):
            raise JobNotFoundError(job_id)
        path = (self.root / job_id).resolve()
        if path.parent != self.root:
            raise JobNotFoundError(job_id)
        return path

    def _write(self, record: JobRecord) -> None:
        with self._lock:
            path = self.job_dir(record.id) / "job.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                record.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            temporary.replace(path)
