from typing import Annotated

from fastapi import APIRouter, Depends

from ai_classifier.api.dependencies import get_result_query
from ai_classifier.api.schemas.scoreboard_ocr import ScoreboardOcrResponse
from ai_classifier.api.services.result_queries import ResultQueryService

router = APIRouter(prefix="/analyses", tags=["scoreboard"])
Query = Annotated[ResultQueryService, Depends(get_result_query)]


@router.get("/{job_id}/scoreboard-ocr", response_model=ScoreboardOcrResponse)
def get_scoreboard_ocr(job_id: str, query: Query) -> ScoreboardOcrResponse:
    return ScoreboardOcrResponse(
        job_id=job_id,
        scoreboard_ocr=query.section(job_id, "scoreboard_ocr"),
    )
