from typing import Annotated

from fastapi import APIRouter, Depends

from ai_classifier.api.dependencies import get_result_query
from ai_classifier.api.schemas.reliability import ReliabilityResponse
from ai_classifier.api.services.result_queries import ResultQueryService

router = APIRouter(prefix="/analyses", tags=["reliability"])
Query = Annotated[ResultQueryService, Depends(get_result_query)]


@router.get("/{job_id}/reliability", response_model=ReliabilityResponse)
def get_reliability(job_id: str, query: Query) -> ReliabilityResponse:
    return ReliabilityResponse(
        job_id=job_id,
        reliability=query.section(job_id, "reliability"),
    )
