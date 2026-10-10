from typing import Any

from pydantic import BaseModel


class ScoreboardOcrResponse(BaseModel):
    job_id: str
    scoreboard_ocr: dict[str, Any]
