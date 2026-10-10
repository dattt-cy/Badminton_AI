from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from ai_classifier.api.dependencies import get_job_service
from ai_classifier.api.schemas.job import AnalysisJob, DeleteJobResponse
from ai_classifier.api.services.analysis_jobs import (
    AnalysisJobService,
    JobNotFoundError,
    UploadTooLargeError,
)

router = APIRouter(prefix="/analyses", tags=["analyses"])
Service = Annotated[AnalysisJobService, Depends(get_job_service)]


@router.post("", response_model=AnalysisJob, status_code=status.HTTP_202_ACCEPTED)
def create_analysis(
    service: Service,
    video: Annotated[UploadFile, File(description="Badminton match video")],
    player_upper: Annotated[str, Form(min_length=1, max_length=100)] = "Player Upper",
    player_lower: Annotated[str, Form(min_length=1, max_length=100)] = "Player Lower",
) -> AnalysisJob:
    if not video.filename:
        raise HTTPException(status_code=400, detail="A filename is required")
    try:
        return service.create(video.file, video.filename, player_upper, player_lower)
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc


@router.get("/{job_id}", response_model=AnalysisJob)
def get_analysis(job_id: str, service: Service) -> AnalysisJob:
    try:
        return service.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Analysis job not found") from exc


@router.delete("/{job_id}", response_model=DeleteJobResponse)
def delete_analysis(job_id: str, service: Service) -> DeleteJobResponse:
    try:
        service.delete(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Analysis job not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail="Cannot delete a processing job") from exc
    return DeleteJobResponse(id=job_id, deleted=True)

