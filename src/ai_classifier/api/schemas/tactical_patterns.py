from typing import Any

from pydantic import BaseModel


class TacticalPatternsResponse(BaseModel):
    job_id: str
    tactical_patterns: dict[str, Any]
