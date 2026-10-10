from typing import Any

from pydantic import BaseModel


class CourtDistributionResponse(BaseModel):
    job_id: str
    spatial_distribution_3x3: dict[str, Any]

