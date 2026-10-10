from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ai_classifier.api.services.analysis_jobs import JobNotFoundError
from ai_classifier.api.services.result_queries import ResultNotReadyError


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(JobNotFoundError)
    async def job_not_found(_: Request, __: JobNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": "Analysis job not found"})

    @app.exception_handler(ResultNotReadyError)
    async def result_not_ready(_: Request, exc: ResultNotReadyError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": f"Analysis result is not ready ({exc})"},
        )
