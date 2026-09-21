"""Match downloaded ShuttleSet videos to match IDs and audit their metadata."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.datasets.build_shuttleset_manifest import split_for_match


VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_dir", type=Path)
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path(
            "external/BST-Badminton-Stroke-type-Transformer/ShuttleSet/set"
        ),
    )
    parser.add_argument(
        "--resolution-csv",
        type=Path,
        default=Path(
            "external/BST-Badminton-Stroke-type-Transformer/ShuttleSet/"
            "my_raw_video_resolution.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/shuttleset_video_audit.csv"),
    )
    return parser.parse_args()


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def leading_id(path: Path) -> int | None:
    match = re.match(r"^(\d+)(?:\s|\s*-\s*)", path.name)
    return int(match.group(1)) if match else None


def similarity(match: dict[str, str], path: Path) -> float:
    source = normalize(path.stem.removesuffix(".mp4"))
    canonical = normalize(match["video"])
    canonical_score = SequenceMatcher(None, canonical, source).ratio()
    players = [normalize(match["winner"]), normalize(match["loser"])]
    player_score = sum(1.0 for player in players if player in source) / 2.0
    tournament_tokens = set(normalize(match["tournament"]).split())
    source_tokens = set(source.split())
    tournament_score = len(tournament_tokens & source_tokens) / max(
        len(tournament_tokens), 1
    )
    return 0.55 * canonical_score + 0.35 * player_score + 0.10 * tournament_score


def assign_videos(
    matches: list[dict[str, str]], videos: list[Path]
) -> dict[int, tuple[Path, str, float]]:
    assigned: dict[int, tuple[Path, str, float]] = {}
    used: set[Path] = set()
    by_id: dict[int, list[Path]] = {}
    for video in videos:
        match_id = leading_id(video)
        if match_id is not None:
            by_id.setdefault(match_id, []).append(video)
    for match in matches:
        match_id = int(match["id"])
        candidates = by_id.get(match_id, [])
        if len(candidates) == 1:
            assigned[match_id] = (candidates[0], "filename_id", 1.0)
            used.add(candidates[0])
    for match in matches:
        match_id = int(match["id"])
        if match_id in assigned:
            continue
        ranked = sorted(
            ((similarity(match, video), video) for video in videos if video not in used),
            key=lambda item: item[0],
            reverse=True,
        )
        if not ranked:
            continue
        score, video = ranked[0]
        assigned[match_id] = (video, "metadata_name", score)
        used.add(video)
    return assigned


def probe(path: Path) -> dict[str, object]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,codec_name",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    fps = float(numerator) / float(denominator)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "codec": stream["codec_name"],
        "duration_seconds": float(payload["format"]["duration"]),
    }


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    matches = load_csv(args.annotation_root / "match.csv")
    required_matches = [
        match for match in matches if split_for_match(int(match["id"])) != "excluded"
    ]
    resolutions = {
        int(row["id"]): (int(row["width"]), int(row["height"]))
        for row in load_csv(args.resolution_csv)
    }
    videos = sorted(
        path
        for path in args.video_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    assignments = assign_videos(required_matches, videos)
    records: list[dict[str, object]] = []
    for match in required_matches:
        match_id = int(match["id"])
        assigned = assignments.get(match_id)
        if assigned is None:
            records.append(
                {
                    "match_id": match_id,
                    "split": split_for_match(match_id),
                    "status": "missing",
                    "match_score": 0.0,
                    "match_method": "",
                    "local_video_path": "",
                    "expected_video_name": match["video"],
                    "expected_url": match["url"],
                }
            )
            continue
        path, method, score = assigned
        metadata = probe(path)
        expected_width, expected_height = resolutions[match_id]
        resolution_ok = (
            metadata["width"] == expected_width
            and metadata["height"] == expected_height
        )
        match_confident = method == "filename_id" or score >= 0.70
        records.append(
            {
                "match_id": match_id,
                "split": split_for_match(match_id),
                "status": "ok" if match_confident and resolution_ok else "review",
                "match_score": round(score, 4),
                "match_method": method,
                "local_video_path": str(path.resolve()),
                "expected_video_name": match["video"],
                "expected_url": match["url"],
                "width": metadata["width"],
                "height": metadata["height"],
                "expected_width": expected_width,
                "expected_height": expected_height,
                "resolution_ok": resolution_ok,
                "fps": round(float(metadata["fps"]), 6),
                "duration_seconds": round(float(metadata["duration_seconds"]), 3),
                "codec": metadata["codec"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    for record in records:
        for fieldname in fieldnames:
            record.setdefault(fieldname, "")
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "required_matches": len(required_matches),
        "downloaded_video_files": len(videos),
        "status_counts": {
            status: sum(record["status"] == status for record in records)
            for status in ("ok", "review", "missing")
        },
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
