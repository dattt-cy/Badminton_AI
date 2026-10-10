from typing import Annotated

from fastapi import APIRouter, Depends

from ai_classifier.api.dependencies import get_result_query
from ai_classifier.api.schemas.insight import InsightListResponse
from ai_classifier.api.services.result_queries import ResultQueryService

router = APIRouter(prefix="/analyses", tags=["insights"])
Query = Annotated[ResultQueryService, Depends(get_result_query)]


@router.get("/{job_id}/insights", response_model=InsightListResponse)
def list_insights(job_id: str, query: Query) -> InsightListResponse:
    items = query.section(job_id, "coach_insights")
    return InsightListResponse(job_id=job_id, items=items, total=len(items))
