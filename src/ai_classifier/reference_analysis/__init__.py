"""Reference profile construction from confirmed-good sample strokes (mục 10)."""

from ai_classifier.error_detection import phase_feature_value

from .reference_profile import MIN_REFERENCE_SAMPLES, build_kinetic_chain_reference, build_range_checks

__all__ = [
    "MIN_REFERENCE_SAMPLES",
    "build_kinetic_chain_reference",
    "build_range_checks",
    "phase_feature_value",
]
