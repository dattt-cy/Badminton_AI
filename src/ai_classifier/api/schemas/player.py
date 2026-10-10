from typing import Any

from pydantic import BaseModel


class PlayerProfilesResponse(BaseModel):
    job_id: str
    profiles: dict[str, Any]

