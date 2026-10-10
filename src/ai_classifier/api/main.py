from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_classifier.api.core.config import Settings, settings
from ai_classifier.api.core.exception_handlers import register_exception_handlers
from ai_classifier.api.routers import (
    analyses,
    court,
    health,
    insights,
    movement,
    players,
    radar,
    rallies,
    reliability,
    results,
    scoreboard,
    scoreboard_ocr,
    strokes,
    structured_tactics,
    summary,
    tactical_patterns,
    timing,
    transitions,
    video,
)
from ai_classifier.api.services.analysis_jobs import AnalysisJobService, PipelineRunner


def create_app(
    config: Settings = settings,
    *,
    runner: PipelineRunner | None = None,
    storage_dir: Path | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.job_service = AnalysisJobService(
            storage_dir=storage_dir or config.storage_dir,
            max_upload_bytes=config.max_upload_bytes,
            runner=runner,
        )
        yield
        app.state.job_service.close()

    app = FastAPI(
        title=config.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(health.router)
    for api_router in (
        analyses.router,
        results.router,
        video.router,
        summary.router,
        reliability.router,
        strokes.router,
        rallies.router,
        scoreboard.router,
        scoreboard_ocr.router,
        tactical_patterns.router,
        structured_tactics.router,
        movement.router,
        court.router,
        players.router,
        radar.router,
        transitions.router,
        insights.router,
        timing.router,
    ):
        app.include_router(api_router, prefix=config.api_prefix)
    return app


app = create_app()

