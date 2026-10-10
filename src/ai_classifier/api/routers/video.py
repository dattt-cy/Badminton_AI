from typing import Annotated

from fastapi import APIRouter, Depends

from ai_classifier.api.dependencies import get_result_query
from ai_classifier.api.schemas.video import VideoResponse
from ai_classifier.api.services.result_queries import ResultQueryService

router = APIRouter(prefix="/analyses", tags=["video"])
Query = Annotated[ResultQueryService, Depends(get_result_query)]


@router.get("/{job_id}/video", response_model=VideoResponse)
def get_video(job_id: str, query: Query) -> VideoResponse:
    return VideoResponse(job_id=job_id, **query.video(job_id))
