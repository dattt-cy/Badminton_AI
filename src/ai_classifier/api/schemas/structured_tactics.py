from typing import Any

from pydantic import BaseModel


class StructuredTacticsResponse(BaseModel):
    job_id: str
    structured_tactics: dict[str, Any]
