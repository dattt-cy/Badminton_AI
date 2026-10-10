from typing import Any

from pydantic import BaseModel


class TimingResponse(BaseModel):
    job_id: str
    timing_seconds: dict[str, Any]
