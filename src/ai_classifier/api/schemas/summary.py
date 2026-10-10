from typing import Any

from pydantic import BaseModel


class SummaryResponse(BaseModel):
    job_id: str
    summary: dict[str, Any]

