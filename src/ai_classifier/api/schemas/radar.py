from typing import Any

from pydantic import BaseModel


class RadarChartResponse(BaseModel):
    job_id: str
    radar_chart: dict[str, Any]
