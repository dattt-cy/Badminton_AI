from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrokeItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    hit_index: int
    frame: int
    time_seconds: float
    hitter: str
    hitter_name: str
    stroke: str
    confidence: float
    rally: int | None = None
    analytics: dict[str, Any] = Field(default_factory=dict)


class StrokeFilters(BaseModel):
    player: str | None = None
    stroke: str | None = None
    rally: int | None = None
    min_confidence: float = 0.0


class StrokeListResponse(BaseModel):
    job_id: str
    items: list[StrokeItem]
    page: int
    page_size: int
    total: int
    total_pages: int
    filters: StrokeFilters

