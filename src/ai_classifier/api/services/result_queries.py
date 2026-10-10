from __future__ import annotations

import math
from typing import Any

from ai_classifier.api.schemas.stroke import StrokeFilters
from ai_classifier.api.services.analysis_jobs import AnalysisJobService


class ResultNotReadyError(RuntimeError):
    pass


class ResultQueryService:
    """Read-only queries over a completed pipeline result."""

    def __init__(self, jobs: AnalysisJobService) -> None:
        self.jobs = jobs

    def complete_result(self, job_id: str) -> dict[str, Any]:
        try:
            return self.jobs.get_result(job_id)
        except RuntimeError as exc:
            raise ResultNotReadyError(str(exc)) from exc

    def video(self, job_id: str) -> dict[str, Any]:
        result = self.complete_result(job_id)
        return {
            "video": result.get("video", {}),
            "players": result.get("players", {}),
            "schema_version": result.get("schema_version"),
        }

    def summary(self, job_id: str) -> dict[str, Any]:
        return self.section(job_id, "summary")

    def strokes(
        self,
        job_id: str,
        *,
        page: int,
        page_size: int,
        filters: StrokeFilters,
    ) -> dict[str, Any]:
        rows = list(self.complete_result(job_id).get("predictions", []))
        if filters.player:
            rows = [row for row in rows if row.get("hitter") == filters.player]
        if filters.stroke:
            normalized = filters.stroke.casefold()
            rows = [
                row for row in rows if str(row.get("stroke", "")).casefold() == normalized
            ]
        if filters.rally is not None:
            rows = [row for row in rows if row.get("rally") == filters.rally]
        rows = [
            row
            for row in rows
            if float(row.get("confidence", 0.0)) >= filters.min_confidence
        ]
        total = len(rows)
        start = (page - 1) * page_size
        return {
            "items": rows[start : start + page_size],
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": math.ceil(total / page_size),
            "filters": filters,
        }

    def rallies(self, job_id: str) -> list[dict[str, Any]]:
        scoreboard = self.complete_result(job_id).get("scoreboard", {})
        return list(scoreboard.get("rallies", []))

    def scoreboard(self, job_id: str) -> dict[str, Any]:
        return self.section(job_id, "scoreboard")

    def section(self, job_id: str, key: str) -> Any:
        return self.complete_result(job_id).get(key, {})
