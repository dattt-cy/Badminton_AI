"""Neural network architectures for Badminton AI Classifier."""

from .fusion_head import FusionHead, SIDE_CLASSES, STROKE_CLASSES
from .multi_task_r2plus1d import MultiTaskR2Plus1D
from .cr_gated_fusion import CRGatedFusionModel

__all__ = [
    "FusionHead",
    "MultiTaskR2Plus1D",
    "CRGatedFusionModel",
    "STROKE_CLASSES",
    "SIDE_CLASSES",
]

