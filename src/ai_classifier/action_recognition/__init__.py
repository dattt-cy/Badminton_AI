"""Skeleton-based action recognition models and PySKL adapters."""

from .stgcnpp_classifier import (
    DEFAULT_LABELS,
    ActionPrediction,
    STGCNPPClassifier,
)

__all__ = ["DEFAULT_LABELS", "ActionPrediction", "STGCNPPClassifier"]
