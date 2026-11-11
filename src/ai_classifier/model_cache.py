"""Process-wide cache of loaded models.

The AI worker analyses many videos in one long-running process. Loading HitNet,
TrackNet, YOLO pose, KAP, the rally model and EasyOCR again for every job costs
several seconds per video, so each one is loaded once and reused. The key holds
the checkpoint's path, modification time and size, so replacing a checkpoint
file is picked up by the next job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

_CACHE: dict[str, tuple[tuple, Any]] = {}


def file_key(path: str | Path) -> tuple:
    """Identity of a checkpoint file: resolved path, modification time and size."""
    candidate = Path(path)
    try:
        stat = candidate.stat()
    except OSError:
        return (str(path),)
    return (str(candidate.resolve()), stat.st_mtime_ns, stat.st_size)


def cached(kind: str, key: tuple, load: Callable[[], Any]) -> Any:
    """Return the object stored for ``kind`` under ``key``, loading it on a miss.

    Each kind keeps one entry; a different key replaces it, which bounds GPU memory
    when a checkpoint is swapped.
    """
    entry = _CACHE.get(kind)
    if entry is not None and entry[0] == key:
        return entry[1]
    _CACHE.pop(kind, None)
    value = load()
    _CACHE[kind] = (key, value)
    return value


def clear() -> None:
    _CACHE.clear()
