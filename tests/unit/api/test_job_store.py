from pathlib import Path

from ai_classifier.api.job_store import JobStore
from ai_classifier.api.schemas import (
    AnalysisOptions,
    CameraView,
    JobStatus,
    Technique,
)


def _options() -> AnalysisOptions:
    return AnalysisOptions(
        technique=Technique.FOREHAND_CLEAR,
        view=CameraView.FRONT,
    )


def test_job_store_persists_updates(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    record, job_dir = store.create(_options(), input_filename="input.mp4")

    updated = store.update(record.id, status=JobStatus.RUNNING)

    assert job_dir.is_dir()
    assert updated.status == JobStatus.RUNNING
    assert JobStore(tmp_path).get(record.id).status == JobStatus.RUNNING
    assert not (job_dir / "job.json.tmp").exists()


def test_restart_marks_unfinished_jobs_failed(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    queued, _ = store.create(_options(), input_filename="input.mp4")
    succeeded, _ = store.create(_options(), input_filename="input.mov")
    store.update(succeeded.id, status=JobStatus.SUCCEEDED)

    store.mark_interrupted_jobs_failed()

    interrupted = store.get(queued.id)
    assert interrupted.status == JobStatus.FAILED
    assert "restarted" in (interrupted.error or "").lower()
    assert store.get(succeeded.id).status == JobStatus.SUCCEEDED
