from typing import Annotated

from fastapi import APIRouter, Depends

from ai_classifier.api.dependencies import get_result_query
from ai_classifier.api.schemas.radar import RadarChartResponse
from ai_classifier.api.services.result_queries import ResultQueryService

router = APIRouter(prefix="/analyses", tags=["charts"])
Query = Annotated[ResultQueryService, Depends(get_result_query)]


@router.get("/{job_id}/charts/radar", response_model=RadarChartResponse)
def get_radar_chart(job_id: str, query: Query) -> RadarChartResponse:
    return RadarChartResponse(job_id=job_id, radar_chart=query.section(job_id, "radar_chart"))
