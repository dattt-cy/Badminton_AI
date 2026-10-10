from typing import Annotated

from fastapi import APIRouter, Depends

from ai_classifier.api.dependencies import get_result_query
from ai_classifier.api.schemas.player import PlayerProfilesResponse
from ai_classifier.api.services.result_queries import ResultQueryService

router = APIRouter(prefix="/analyses", tags=["players"])
Query = Annotated[ResultQueryService, Depends(get_result_query)]


@router.get("/{job_id}/players", response_model=PlayerProfilesResponse)
def get_player_profiles(job_id: str, query: Query) -> PlayerProfilesResponse:
    return PlayerProfilesResponse(
        job_id=job_id,
        profiles=query.section(job_id, "player_profiles"),
    )
