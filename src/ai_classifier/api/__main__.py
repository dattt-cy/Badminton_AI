"""Development/standalone entry point for the API service."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "ai_classifier.api.app:app",
        host=os.getenv("AI_CLASSIFIER_HOST", "127.0.0.1"),
        port=int(os.getenv("AI_CLASSIFIER_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
