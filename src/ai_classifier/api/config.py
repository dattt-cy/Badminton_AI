"""Environment-driven API configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _integer(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """Runtime settings kept separate from the web framework."""

    repo_root: Path
    jobs_root: Path
    max_upload_bytes: int = 500 * 1024 * 1024
    max_concurrent_jobs: int = 1
    inference_timeout_seconds: int = 30 * 60
    enable_wsl_classifier: bool = False

    @classmethod
    def from_env(cls) -> "ApiSettings":
        default_repo = Path(__file__).resolve().parents[3]
        repo_root = Path(os.getenv("AI_CLASSIFIER_REPO_ROOT", default_repo)).resolve()
        jobs_root = Path(
            os.getenv("AI_CLASSIFIER_JOBS_ROOT", repo_root / "outputs" / "api_jobs")
        ).resolve()
        return cls(
            repo_root=repo_root,
            jobs_root=jobs_root,
            max_upload_bytes=_integer(
                "AI_CLASSIFIER_MAX_UPLOAD_BYTES", 500 * 1024 * 1024
            ),
            max_concurrent_jobs=_integer("AI_CLASSIFIER_MAX_CONCURRENT_JOBS", 1),
            inference_timeout_seconds=_integer(
                "AI_CLASSIFIER_INFERENCE_TIMEOUT_SECONDS", 30 * 60
            ),
            enable_wsl_classifier=_boolean(
                "AI_CLASSIFIER_ENABLE_WSL_CLASSIFIER", False
            ),
        )
