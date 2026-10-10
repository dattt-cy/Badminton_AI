from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _origins_from_env() -> tuple[str, ...]:
    raw = os.getenv(
        "AI_CLASSIFIER_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    )
    return tuple(origin.strip() for origin in raw.split(",") if origin.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "Badminton AI Classifier API"
    api_prefix: str = "/api/v1"
    storage_dir: Path = Path("work_dirs/api")
    cors_origins: tuple[str, ...] = _origins_from_env()
    max_upload_bytes: int = int(
        os.getenv("AI_CLASSIFIER_MAX_UPLOAD_BYTES", str(2 * 1024**3))
    )


settings = Settings()

