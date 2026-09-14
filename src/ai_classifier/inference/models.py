"""Framework-independent inputs for end-to-end inference."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Technique(str, Enum):
    AUTO = "auto"
    FOREHAND_CLEAR = "forehand_clear"
    BACKHAND_DRIVE = "backhand_drive"


class CameraView(str, Enum):
    FRONT = "front"
    SIDE = "side"


class Handedness(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class TargetPlayer(str, Enum):
    ANY = "any"
    SINGLE = "single"
    FAR = "far"
    NEAR = "near"


class SegmentationMode(str, Enum):
    AUTO = "auto"
    SINGLE = "single"
    MULTI = "multi"


class AnalysisOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    technique: Technique
    view: CameraView
    handedness: Handedness = Handedness.RIGHT
    target: TargetPlayer = TargetPlayer.SINGLE
    segmentation_mode: SegmentationMode = SegmentationMode.SINGLE
    floor_angle_degrees: float = Field(default=0.0, ge=-45.0, le=45.0)
    swing_peak_frame: int | None = Field(default=None, ge=0)
    generate_preview: bool = True
    run_classifier: bool = False
