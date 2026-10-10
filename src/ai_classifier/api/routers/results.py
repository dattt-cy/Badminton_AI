from typing import Annotated

from fastapi import APIRouter, Depends

from ai_classifier.api.dependencies import get_result_query
from ai_classifier.api.schemas.result import AnalysisResult
from ai_classifier.api.services.result_queries import ResultQueryService

router = APIRouter(prefix="/analyses", tags=["results"])
Query = Annotated[ResultQueryService, Depends(get_result_query)]


@router.get("/{job_id}/result", response_model=AnalysisResult)
def get_analysis_result(job_id: str, query: Query) -> AnalysisResult:
    return AnalysisResult(job_id=job_id, data=query.complete_result(job_id))
