from typing import Any

from pydantic import BaseModel, ConfigDict


class CoachInsight(BaseModel):
    model_config = ConfigDict(extra="allow")

    category: str
    target: str
    insight: str


class InsightListResponse(BaseModel):
    job_id: str
    items: list[CoachInsight | dict[str, Any]]
    total: int

