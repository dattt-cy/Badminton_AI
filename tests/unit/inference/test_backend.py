import subprocess
from pathlib import Path

import pytest

from ai_classifier.inference.backend import CliInferenceBackend, InferenceError
from ai_classifier.inference.models import AnalysisOptions, CameraView, Technique


def _options(**overrides) -> AnalysisOptions:
    values = {
        "technique": Technique.FOREHAND_CLEAR,
        "view": CameraView.FRONT,
    }
    values.update(overrides)
    return AnalysisOptions(**values)


def test_cli_backend_builds_argument_list_without_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    output = tmp_path / "output"
    output.mkdir()
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        (output / "analysis.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "done", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = CliInferenceBackend(repo, timeout_seconds=60)

    result = backend.analyze(
        input_path=tmp_path / "clip with spaces.mp4",
        output_dir=output,
        options=_options(generate_preview=False, swing_peak_frame=42),
    )

    command = captured["command"]
    assert result == output / "analysis.json"
    assert captured["kwargs"]["cwd"] == repo.resolve()
    assert captured["kwargs"]["check"] is False
    assert "--no-preview" in command
    assert command[command.index("--swing-peak-frame") + 1] == "42"


def test_cli_backend_rejects_classifier_when_bridge_is_disabled(
    tmp_path: Path,
) -> None:
    backend = CliInferenceBackend(tmp_path, timeout_seconds=60)

    with pytest.raises(InferenceError, match="disabled"):
        backend.analyze(
            input_path=tmp_path / "clip.mp4",
            output_dir=tmp_path / "output",
            options=_options(run_classifier=True),
        )


def test_cli_backend_turns_timeout_into_domain_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 5)

    monkeypatch.setattr(subprocess, "run", timeout)
    backend = CliInferenceBackend(tmp_path, timeout_seconds=5)

    with pytest.raises(InferenceError, match="5-second"):
        backend.analyze(
            input_path=tmp_path / "clip.mp4",
            output_dir=tmp_path / "output",
            options=_options(),
        )
