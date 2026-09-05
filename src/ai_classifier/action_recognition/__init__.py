"""Skeleton-based action recognition models and PySKL adapters."""

from .stgcnpp_classifier import (
    TWO_CLASS_LABELS,
    ActionPrediction,
    STGCNPPClassifier,
)

__all__ = [
    "TWO_CLASS_LABELS",
    "ActionPrediction",
    "STGCNPPClassifier",
]
