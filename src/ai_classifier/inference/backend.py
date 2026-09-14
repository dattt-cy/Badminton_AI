"""Replaceable execution backends for end-to-end inference."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Protocol

from ai_classifier.inference.models import AnalysisOptions


class InferenceError(RuntimeError):
    """An expected failure while executing the analysis pipeline."""


class InferenceBackend(Protocol):
    def analyze(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        options: AnalysisOptions,
    ) -> Path: ...


class CliInferenceBackend:
    """Compatibility backend around the existing, tested end-to-end CLI.

    Keeping this adapter isolated lets a future in-process or remote queue
    backend replace it without changing the HTTP contract.
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        timeout_seconds: int,
        enable_wsl_classifier: bool = False,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.timeout_seconds = timeout_seconds
        self.enable_wsl_classifier = enable_wsl_classifier

    def analyze(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        options: AnalysisOptions,
    ) -> Path:
        command = [
            sys.executable,
            str(self.repo_root / "scripts" / "inference" / "analyze_technique.py"),
            str(input_path),
            options.technique.value,
            "--view",
            options.view.value,
            "--handedness",
            options.handedness.value,
            "--target",
            options.target.value,
            "--segmentation-mode",
            options.segmentation_mode.value,
            "--floor-angle-degrees",
            str(options.floor_angle_degrees),
            "--output-dir",
            str(output_dir),
        ]
        if options.swing_peak_frame is not None:
            command.extend(["--swing-peak-frame", str(options.swing_peak_frame)])
        if not options.generate_preview:
            command.append("--no-preview")
        if options.run_classifier:
            if not self.enable_wsl_classifier:
                raise InferenceError(
                    "Classifier execution is disabled on this API instance."
                )
            command.append("--run-classifier-wsl")

        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_root,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise InferenceError(
                f"Analysis exceeded the {self.timeout_seconds}-second timeout."
            ) from exc
        except OSError as exc:
            raise InferenceError(f"Could not start inference: {exc}") from exc

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-4000:]
            raise InferenceError(detail or f"Inference exited with {completed.returncode}.")

        result = output_dir / "analysis.json"
        if not result.is_file():
            raise InferenceError("Inference completed without producing analysis.json.")
        return result
