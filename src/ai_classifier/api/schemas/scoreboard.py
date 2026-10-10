from typing import Any

from pydantic import BaseModel


class ScoreboardResponse(BaseModel):
    job_id: str
    scoreboard: dict[str, Any]

