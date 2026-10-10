"""Neural network architectures and model checkpoints for Badminton AI Classifier."""

from pathlib import Path

from .taxonomy import SIDE_CLASSES, STROKE_CLASSES
from .multi_task_r2plus1d import MultiTaskR2Plus1D
from .kap_fusion import KAPScratchModel

CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"

DEFAULT_HIT_MODEL = CHECKPOINTS_DIR / "hit_detector.pth"
DEFAULT_VISUAL_BACKBONE = CHECKPOINTS_DIR / "visual_backbone.pth"
DEFAULT_KAP_MODEL = CHECKPOINTS_DIR / "kap_classifier.pth"
DEFAULT_RALLY_MODEL = CHECKPOINTS_DIR / "rally_boundary_detector.joblib"
DEFAULT_POSE_MODEL = CHECKPOINTS_DIR / "yolov8n-pose.pt"
DEFAULT_TRACKNET_MODEL = CHECKPOINTS_DIR / "tracknet_best.pt"

__all__ = [
    "MultiTaskR2Plus1D",
    "KAPScratchModel",
    "STROKE_CLASSES",
    "SIDE_CLASSES",
    "CHECKPOINTS_DIR",
    "DEFAULT_HIT_MODEL",
    "DEFAULT_VISUAL_BACKBONE",
    "DEFAULT_KAP_MODEL",
    "DEFAULT_RALLY_MODEL",
    "DEFAULT_POSE_MODEL",
    "DEFAULT_TRACKNET_MODEL",
]
