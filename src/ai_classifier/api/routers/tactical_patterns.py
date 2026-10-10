from typing import Annotated

from fastapi import APIRouter, Depends

from ai_classifier.api.dependencies import get_result_query
from ai_classifier.api.schemas.tactical_patterns import TacticalPatternsResponse
from ai_classifier.api.services.result_queries import ResultQueryService

router = APIRouter(prefix="/analyses", tags=["tactics"])
Query = Annotated[ResultQueryService, Depends(get_result_query)]


@router.get("/{job_id}/tactical-patterns", response_model=TacticalPatternsResponse)
def get_tactical_patterns(job_id: str, query: Query) -> TacticalPatternsResponse:
    return TacticalPatternsResponse(
        job_id=job_id,
        tactical_patterns=query.section(job_id, "tactical_patterns"),
    )
