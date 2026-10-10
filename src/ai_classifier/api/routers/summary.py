from typing import Annotated

from fastapi import APIRouter, Depends

from ai_classifier.api.dependencies import get_result_query
from ai_classifier.api.schemas.summary import SummaryResponse
from ai_classifier.api.services.result_queries import ResultQueryService

router = APIRouter(prefix="/analyses", tags=["summary"])
Query = Annotated[ResultQueryService, Depends(get_result_query)]


@router.get("/{job_id}/summary", response_model=SummaryResponse)
def get_summary(job_id: str, query: Query) -> SummaryResponse:
    return SummaryResponse(job_id=job_id, summary=query.summary(job_id))
