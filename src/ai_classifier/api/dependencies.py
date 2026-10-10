from fastapi import Request

from ai_classifier.api.services.analysis_jobs import AnalysisJobService
from ai_classifier.api.services.result_queries import ResultQueryService


def get_job_service(request: Request) -> AnalysisJobService:
    return request.app.state.job_service


def get_result_query(request: Request) -> ResultQueryService:
    return ResultQueryService(get_job_service(request))

