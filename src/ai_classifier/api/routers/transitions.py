from typing import Annotated

from fastapi import APIRouter, Depends

from ai_classifier.api.dependencies import get_result_query
from ai_classifier.api.schemas.transition import TransitionMatrixResponse
from ai_classifier.api.services.result_queries import ResultQueryService

router = APIRouter(prefix="/analyses", tags=["charts"])
Query = Annotated[ResultQueryService, Depends(get_result_query)]


@router.get("/{job_id}/charts/transitions", response_model=TransitionMatrixResponse)
def get_transition_matrix(job_id: str, query: Query) -> TransitionMatrixResponse:
    return TransitionMatrixResponse(
        job_id=job_id,
        response_transition_matrix=query.section(job_id, "response_transition_matrix"),
    )
