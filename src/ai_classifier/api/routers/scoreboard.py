from typing import Annotated

from fastapi import APIRouter, Depends

from ai_classifier.api.dependencies import get_result_query
from ai_classifier.api.schemas.scoreboard import ScoreboardResponse
from ai_classifier.api.services.result_queries import ResultQueryService

router = APIRouter(prefix="/analyses", tags=["scoreboard"])
Query = Annotated[ResultQueryService, Depends(get_result_query)]


@router.get("/{job_id}/scoreboard", response_model=ScoreboardResponse)
def get_scoreboard(job_id: str, query: Query) -> ScoreboardResponse:
    return ScoreboardResponse(job_id=job_id, scoreboard=query.scoreboard(job_id))
