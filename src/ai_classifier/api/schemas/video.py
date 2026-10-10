from typing import Any

from pydantic import BaseModel


class VideoResponse(BaseModel):
    job_id: str
    video: dict[str, Any]
    players: dict[str, str]
    schema_version: str | None = None

