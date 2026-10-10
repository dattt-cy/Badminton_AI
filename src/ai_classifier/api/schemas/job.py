from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class AnalysisJob(BaseModel):
    id: str
    status: JobStatus
    filename: str
    player_upper: str
    player_lower: str
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    result_url: str | None = None


class DeleteJobResponse(BaseModel):
    id: str
    deleted: bool

