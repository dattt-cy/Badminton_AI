from typing import Annotated

from fastapi import APIRouter, Depends, Query as QueryParam

from ai_classifier.api.dependencies import get_result_query
from ai_classifier.api.schemas.stroke import StrokeFilters, StrokeListResponse
from ai_classifier.api.services.result_queries import ResultQueryService

router = APIRouter(prefix="/analyses", tags=["strokes"])
QueryService = Annotated[ResultQueryService, Depends(get_result_query)]


@router.get("/{job_id}/strokes", response_model=StrokeListResponse)
def list_strokes(
    job_id: str,
    query: QueryService,
    page: Annotated[int, QueryParam(ge=1)] = 1,
    page_size: Annotated[int, QueryParam(ge=1, le=100)] = 20,
    player: Annotated[str | None, QueryParam(pattern="^(upper|lower)$")] = None,
    stroke: str | None = None,
    rally: Annotated[int | None, QueryParam(ge=1)] = None,
    min_confidence: Annotated[float, QueryParam(ge=0.0, le=1.0)] = 0.0,
) -> StrokeListResponse:
    filters = StrokeFilters(
        player=player,
        stroke=stroke,
        rally=rally,
        min_confidence=min_confidence,
    )
    return StrokeListResponse(
        job_id=job_id,
        **query.strokes(job_id, page=page, page_size=page_size, filters=filters),
    )
