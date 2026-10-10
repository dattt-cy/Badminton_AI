from typing import Annotated

from fastapi import APIRouter, Depends

from ai_classifier.api.dependencies import get_result_query
from ai_classifier.api.schemas.court import CourtDistributionResponse
from ai_classifier.api.services.result_queries import ResultQueryService

router = APIRouter(prefix="/analyses", tags=["court"])
Query = Annotated[ResultQueryService, Depends(get_result_query)]


@router.get("/{job_id}/court-distribution", response_model=CourtDistributionResponse)
def get_court_distribution(job_id: str, query: Query) -> CourtDistributionResponse:
    return CourtDistributionResponse(
        job_id=job_id,
        spatial_distribution_3x3=query.section(job_id, "spatial_distribution_3x3"),
    )
