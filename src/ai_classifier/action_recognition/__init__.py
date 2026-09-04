"""Skeleton-based action recognition models and PySKL adapters."""

from .stgcnpp_classifier import (
    THREE_CLASS_LABELS,
    TWO_CLASS_LABELS,
    ActionPrediction,
    STGCNPPClassifier,
)

__all__ = [
    "THREE_CLASS_LABELS",
    "TWO_CLASS_LABELS",
    "ActionPrediction",
    "STGCNPPClassifier",
]
