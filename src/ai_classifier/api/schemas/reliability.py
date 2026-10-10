from typing import Any

from pydantic import BaseModel


class ReliabilityResponse(BaseModel):
    job_id: str
    reliability: dict[str, Any]
