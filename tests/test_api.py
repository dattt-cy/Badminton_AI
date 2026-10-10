import time
from pathlib import Path

from fastapi.testclient import TestClient

from ai_classifier.api.core.config import Settings
from ai_classifier.api.main import create_app


def fake_runner(*, video_path: Path, output_path: Path, **_: object) -> dict[str, object]:
    result = {
        "schema_version": "test",
        "video": {"filename": video_path.name, "fps": 30.0},
        "players": {"upper": "Alice", "lower": "Bob"},
        "summary": {"hit_count": 2, "estimated_rally_count": 1},
        "reliability": {"score_is_estimated": True},
        "predictions": [
            {
                "hit_index": 1,
                "frame": 10,
                "time_seconds": 0.33,
                "hitter": "upper",
                "hitter_name": "Alice",
                "stroke": "smash",
                "confidence": 0.9,
                "rally": 1,
                "analytics": {},
            },
            {
                "hit_index": 2,
                "frame": 20,
                "time_seconds": 0.67,
                "hitter": "lower",
                "hitter_name": "Bob",
                "stroke": "drop",
                "confidence": 0.6,
                "rally": 1,
                "analytics": {},
            },
        ],
        "scoreboard": {
            "estimated_score": {"text": "1 - 0"},
            "rallies": [
                {
                    "rally_id": 1,
                    "start_time_seconds": 0.33,
                    "end_time_seconds": 0.67,
                    "duration_seconds": 0.34,
                    "hits_count": 2,
                    "winner": "upper",
                    "winner_name": "Alice",
                    "stroke_sequence": ["smash", "drop"],
                }
            ],
        },
        "scoreboard_ocr": {"enabled": False},
        "tactical_patterns": {"top_sequences_3gram": []},
        "structured_tactics": {"upper": []},
        "movement": {"upper": {"distance_meters": 1.2}},
        "spatial_distribution_3x3": {"upper": {"1": 1}},
        "player_profiles": {"upper": {"attacking_ratio": 0.5}},
        "radar_chart": {"axes": ["attack"]},
        "response_transition_matrix": {"upper": {}},
        "coach_insights": [
            {"category": "attack", "target": "upper", "insight": "Keep attacking"}
        ],
        "timing_seconds": {"total": 1.0},
    }
    output_path.write_text(__import__("json").dumps(result), encoding="utf-8")
    return result


def test_health_and_analysis_lifecycle(tmp_path: Path) -> None:
    app = create_app(Settings(max_upload_bytes=1024), runner=fake_runner, storage_dir=tmp_path)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"

        response = client.post(
            "/api/v1/analyses",
            files={"video": ("match.mp4", b"video", "video/mp4")},
            data={"player_upper": "Alice", "player_lower": "Bob"},
        )
        assert response.status_code == 202
        job = response.json()

        for _ in range(50):
            job = client.get(f"/api/v1/analyses/{job['id']}").json()
            if job["status"] == "completed":
                break
            time.sleep(0.01)

        assert job["status"] == "completed"
        result = client.get(job["result_url"])
        assert result.status_code == 200
        assert result.json()["data"]["schema_version"] == "test"

        endpoints = {
            "video": "video",
            "summary": "summary",
            "reliability": "reliability",
            "rallies": "items",
            "scoreboard": "scoreboard",
            "scoreboard-ocr": "scoreboard_ocr",
            "tactical-patterns": "tactical_patterns",
            "structured-tactics": "structured_tactics",
            "movement": "movement",
            "court-distribution": "spatial_distribution_3x3",
            "players": "profiles",
            "charts/radar": "radar_chart",
            "charts/transitions": "response_transition_matrix",
            "insights": "items",
            "timing": "timing_seconds",
        }
        for path, expected_key in endpoints.items():
            response = client.get(f"/api/v1/analyses/{job['id']}/{path}")
            assert response.status_code == 200, path
            assert expected_key in response.json(), path

        strokes = client.get(
            f"/api/v1/analyses/{job['id']}/strokes",
            params={"player": "upper", "min_confidence": 0.8, "page_size": 1},
        )
        assert strokes.status_code == 200
        assert strokes.json()["total"] == 1
        assert strokes.json()["items"][0]["stroke"] == "smash"

        deleted = client.delete(f"/api/v1/analyses/{job['id']}")
        assert deleted.json() == {"id": job["id"], "deleted": True}
        assert client.get(f"/api/v1/analyses/{job['id']}").status_code == 404


def test_rejects_oversized_upload(tmp_path: Path) -> None:
    app = create_app(Settings(max_upload_bytes=3), runner=fake_runner, storage_dir=tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyses",
            files={"video": ("large.mp4", b"large", "video/mp4")},
        )
    assert response.status_code == 413
