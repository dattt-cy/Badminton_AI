from typing import Annotated

from fastapi import APIRouter, Depends

from ai_classifier.api.dependencies import get_result_query
from ai_classifier.api.schemas.structured_tactics import StructuredTacticsResponse
from ai_classifier.api.services.result_queries import ResultQueryService

router = APIRouter(prefix="/analyses", tags=["tactics"])
Query = Annotated[ResultQueryService, Depends(get_result_query)]


@router.get("/{job_id}/structured-tactics", response_model=StructuredTacticsResponse)
def get_structured_tactics(job_id: str, query: Query) -> StructuredTacticsResponse:
    return StructuredTacticsResponse(
        job_id=job_id,
        structured_tactics=query.section(job_id, "structured_tactics"),
    )
