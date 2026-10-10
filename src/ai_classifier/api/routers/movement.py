from typing import Annotated

from fastapi import APIRouter, Depends

from ai_classifier.api.dependencies import get_result_query
from ai_classifier.api.schemas.movement import MovementResponse
from ai_classifier.api.services.result_queries import ResultQueryService

router = APIRouter(prefix="/analyses", tags=["movement"])
Query = Annotated[ResultQueryService, Depends(get_result_query)]


@router.get("/{job_id}/movement", response_model=MovementResponse)
def get_movement(job_id: str, query: Query) -> MovementResponse:
    return MovementResponse(job_id=job_id, movement=query.section(job_id, "movement"))
