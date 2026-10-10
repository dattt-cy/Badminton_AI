from typing import Annotated

from fastapi import APIRouter, Depends

from ai_classifier.api.dependencies import get_result_query
from ai_classifier.api.schemas.rally import RallyListResponse
from ai_classifier.api.services.result_queries import ResultQueryService

router = APIRouter(prefix="/analyses", tags=["rallies"])
Query = Annotated[ResultQueryService, Depends(get_result_query)]


@router.get("/{job_id}/rallies", response_model=RallyListResponse)
def list_rallies(job_id: str, query: Query) -> RallyListResponse:
    items = query.rallies(job_id)
    return RallyListResponse(job_id=job_id, items=items, total=len(items))
