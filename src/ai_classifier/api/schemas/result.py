from typing import Any

from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    job_id: str
    data: dict[str, Any] = Field(description="Complete pipeline output")

