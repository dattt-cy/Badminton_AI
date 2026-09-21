import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from ai_classifier.api.app import create_app
from ai_classifier.api.config import ApiSettings
from ai_classifier.inference.models import AnalysisOptions


class FakeBackend:
    def analyze(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        options: AnalysisOptions,
    ) -> Path:
        assert input_path.read_bytes() == b"video-data"
        payload = {
            "overall": "review_available",
            "selection": {
                "technique": options.technique.value,
                "view": options.view.value,
            },
        }
        result = output_dir / "analysis.json"
        result.write_text(json.dumps(payload), encoding="utf-8")
        (output_dir / "user_report.md").write_text("# Report\n", encoding="utf-8")
        (output_dir / "pose_viewer.json").write_text(
            '{"version":1,"layout":"coco17","fps":30,"frame_count":1,"frames":[]}'
        )
        return result


def _settings(tmp_path: Path, *, max_upload_bytes: int = 1024) -> ApiSettings:
    return ApiSettings(
        repo_root=tmp_path,
        jobs_root=tmp_path / "jobs",
        max_upload_bytes=max_upload_bytes,
        max_concurrent_jobs=1,
        inference_timeout_seconds=10,
    )


def _wait_for_completion(client: TestClient, status_url: str) -> dict:
    for _ in range(100):
        response = client.get(status_url)
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("job did not complete")


def test_create_poll_and_download_analysis(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), backend=FakeBackend())
    with TestClient(app) as client:
        response = client.post(
            "/v1/analyses",
            files={"video": ("stroke.mp4", b"video-data", "video/mp4")},
            data={"technique": "forehand_clear", "view": "front"},
        )
        assert response.status_code == 202

        job = _wait_for_completion(client, response.json()["status_url"])
        assert job["status"] == "succeeded"
        assert set(job["artifact_urls"]) == {
            "analysis.json",
            "pose_viewer.json",
            "user_report.md",
        }

        result = client.get(job["result_url"])
        assert result.status_code == 200
        assert result.json()["selection"] == {
            "technique": "forehand_clear",
            "view": "front",
        }

        report = client.get(job["artifact_urls"]["user_report.md"])
        assert report.status_code == 200
        assert report.text.replace("\r\n", "\n") == "# Report\n"


def test_web_app_and_health_expose_runtime_capabilities(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), backend=FakeBackend())
    with TestClient(app) as client:
        page = client.get("/")
        health = client.get("/health")

    assert page.status_code == 200
    assert "Badminton Form Lab" in page.text
    assert "id=\"analysisForm\"" in page.text
    assert "id=\"evidenceModal\"" in page.text
    assert "id=\"poseCanvas\"" in page.text
    assert "Skeleton 3D Viewer" in page.text
    assert "Khuỷu trái" in page.text
    assert "Gối phải" in page.text
    assert "id=\"angleLeftElbow\"" in page.text
    assert "id=\"angleRightKnee\"" in page.text
    assert "Đã kiểm tra" in page.text
    assert "Phát lại 2 lần" in page.text
    assert "AI chưa thể kết luận" in page.text
    assert health.json() == {
        "status": "ok",
        "accepting_jobs": True,
        "classifier_enabled": False,
        "max_upload_bytes": 1024,
    }


def test_rejects_unsupported_extension_before_creating_job(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), backend=FakeBackend())
    with TestClient(app) as client:
        response = client.post(
            "/v1/analyses",
            files={"video": ("stroke.txt", b"video-data", "text/plain")},
            data={"technique": "forehand_clear", "view": "front"},
        )

    assert response.status_code == 415
    assert list((tmp_path / "jobs").iterdir()) == []


def test_auto_technique_requires_classifier_capability(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), backend=FakeBackend())
    with TestClient(app) as client:
        response = client.post(
            "/v1/analyses",
            files={"video": ("stroke.mp4", b"video-data", "video/mp4")},
            data={"view": "front"},
        )

    assert response.status_code == 422
    assert "classifier" in response.json()["detail"].lower()


def test_enforces_streaming_upload_limit(tmp_path: Path) -> None:
    app = create_app(
        _settings(tmp_path, max_upload_bytes=4), backend=FakeBackend()
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/analyses",
            files={"video": ("stroke.mp4", b"video-data", "video/mp4")},
            data={"technique": "forehand_clear", "view": "front"},
        )

    assert response.status_code == 413
    metadata = next((tmp_path / "jobs").glob("*/job.json"))
    assert json.loads(metadata.read_text(encoding="utf-8"))["status"] == "failed"
