"""Joint-angle and motion-feature computation."""

from .correction import (
    ElbowCorrectionResult,
    predict_elbow_3d,
    predict_elbow_3d_from_features,
)
from .features import (
    GeometryQuality,
    KINETIC_CHAIN,
    GeometryFeatures,
    KineticChainTiming,
    PhaseBoundaries,
    StrokeGeometrySummary,
    assess_geometry_quality,
    detect_stroke_phases,
    estimate_pose_scale_px,
    extract_geometry_features,
    extract_kinetic_chain_timing,
    summarize_stroke_geometry,
)
from .technique_registry import (
    TechniqueDefinition,
    TechniqueRegistry,
    TechniqueRuleSpec,
    TechniqueView,
    load_technique_registry,
)
from .projection import (
    PerspectiveCamera,
    WeakPerspectiveCamera,
    fit_perspective_camera,
    fit_weak_perspective_camera,
)
from .observable_criteria import evaluate_observable_criteria

__all__ = [
    "GeometryQuality",
    "ElbowCorrectionResult",
    "KINETIC_CHAIN",
    "GeometryFeatures",
    "KineticChainTiming",
    "PhaseBoundaries",
    "StrokeGeometrySummary",
    "assess_geometry_quality",
    "detect_stroke_phases",
    "estimate_pose_scale_px",
    "extract_geometry_features",
    "extract_kinetic_chain_timing",
    "summarize_stroke_geometry",
    "TechniqueDefinition",
    "TechniqueRegistry",
    "TechniqueRuleSpec",
    "TechniqueView",
    "load_technique_registry",
    "predict_elbow_3d",
    "predict_elbow_3d_from_features",
    "WeakPerspectiveCamera",
    "PerspectiveCamera",
    "fit_perspective_camera",
    "fit_weak_perspective_camera",
    "evaluate_observable_criteria",
]
