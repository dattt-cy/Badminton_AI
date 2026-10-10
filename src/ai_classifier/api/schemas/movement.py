from typing import Any

from pydantic import BaseModel


class MovementResponse(BaseModel):
    job_id: str
    movement: dict[str, Any]

