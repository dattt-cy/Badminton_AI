"""Stable request and response contracts for the AI service."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from ai_classifier.inference.models import (
    AnalysisOptions,
    CameraView,
    Handedness,
    SegmentationMode,
    TargetPlayer,
    Technique,
)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    options: AnalysisOptions
    input_filename: str
    error: str | None = None
    artifacts: list[str] = Field(default_factory=list)


class JobAccepted(BaseModel):
    id: str
    status: JobStatus
    status_url: str


class JobResponse(JobRecord):
    result_url: str | None = None
    artifact_urls: dict[str, str] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    accepting_jobs: bool
    classifier_enabled: bool
    max_upload_bytes: int
