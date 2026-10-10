from pydantic import BaseModel, ConfigDict, Field


class RallyItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    rally_id: int
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    hits_count: int
    winner: str
    winner_name: str
    stroke_sequence: list[str] = Field(default_factory=list)


class RallyListResponse(BaseModel):
    job_id: str
    items: list[RallyItem]
    total: int

