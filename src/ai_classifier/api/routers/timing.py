from typing import Annotated

from fastapi import APIRouter, Depends

from ai_classifier.api.dependencies import get_result_query
from ai_classifier.api.schemas.timing import TimingResponse
from ai_classifier.api.services.result_queries import ResultQueryService

router = APIRouter(prefix="/analyses", tags=["timing"])
Query = Annotated[ResultQueryService, Depends(get_result_query)]


@router.get("/{job_id}/timing", response_model=TimingResponse)
def get_timing(job_id: str, query: Query) -> TimingResponse:
    return TimingResponse(job_id=job_id, timing_seconds=query.section(job_id, "timing_seconds"))
