"""Rule-engine checks for geometric and kinetic-chain technique errors."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ai_classifier.biomechanics import (
    KINETIC_CHAIN,
    GeometryFeatures,
    KineticChainTiming,
    PhaseBoundaries,
    load_technique_registry,
)

from .types import ErrorEvent, RangeCheck, RuleDefinition, RuleResult

# (rule_name, feature_name, phase_name) proxies for mục 12.1/12.2. Best-effort
# mappings onto existing GeometryFeatures columns; still need a badminton
# expert to confirm before being treated as ground truth. `elbow_too_close_to_torso`
# (mục 12.2) is intentionally omitted: no existing feature measures
# elbow-to-torso distance.
def evaluate_range(rule_name: str, value: float, check: RangeCheck) -> ErrorEvent | None:
    """Compare one observed value against its reference range.

    Returns `None` when the value is missing or inside the reference range.
    Values inside the buffer around the range are reported as `review`
    (mục 12.4); values further outside are reported as `deviation`.
    """
    if not math.isfinite(value):
        return None
    below = value < check.low and check.direction in {"lower", "two_sided"}
    above = value > check.high and check.direction in {"upper", "two_sided"}
    if not (below or above):
        return None

    span = check.high - check.low
    buffer = span * check.buffer_ratio
    severity = "review" if check.low - buffer <= value <= check.high + buffer else "deviation"
    direction = "thấp hơn" if value < check.low else "cao hơn"
    message = f"{rule_name}: giá trị {value:.3f} {direction} vùng tham chiếu [{check.low:.3f}, {check.high:.3f}]."
    return ErrorEvent(rule_name, message, value, check.low, check.high, severity)


def check_kinetic_chain(
    timing: KineticChainTiming,
    reference_lags: Mapping[str, RangeCheck],
) -> list[ErrorEvent]:
    """Check kinetic-chain force-transfer order and per-lag timing (mục 12.3).

    Requires `timing.valid`; otherwise no event is raised because the pose
    data was insufficient to locate the peaks in the first place.
    """
    if not timing.valid:
        return []

    events: list[ErrorEvent] = []
    if not timing.order_correct:
        joints = ", ".join(_joint_name(name) for name in KINETIC_CHAIN)
        events.append(
            ErrorEvent(
                rule_name="kinetic_chain_out_of_order",
                message=f"Thứ tự truyền lực không đúng chuỗi {joints}.",
                observed=math.nan,
                reference_low=math.nan,
                reference_high=math.nan,
                severity="deviation",
            )
        )

    for lag_name, check in reference_lags.items():
        if lag_name not in timing.lags:
            continue
        event = evaluate_range(lag_name, timing.lags[lag_name], check)
        if event is not None:
            events.append(event)

    return events


def _joint_name(column_name: str) -> str:
    return column_name.split("_", 1)[0]


def phase_feature_value(
    features: GeometryFeatures,
    phase_range: tuple[int, int],
    feature_name: str,
) -> float:
    """Median of one feature's finite values within a phase's frame range.

    Returns NaN for an empty phase range or when no frame in range is finite.
    """
    start, end = phase_range
    if end <= start:
        return float("nan")
    column = features.column(feature_name)[start:end]
    finite = column[np.isfinite(column)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def check_phase_features(
    features: GeometryFeatures,
    phases: PhaseBoundaries,
    technique: str,
    reference_ranges: Mapping[str, RangeCheck],
    *,
    registry_path: Path = Path("configs/biomechanics/techniques.yaml"),
) -> list[ErrorEvent]:
    """Check the mục 12.1/12.2 per-phase feature rules for one technique.

    Requires `phases.valid`; otherwise no event is raised because the phase
    boundaries could not be located in the first place.
    """
    if not phases.valid:
        return []

    definition = load_technique_registry(registry_path).technique(technique)
    events: list[ErrorEvent] = []
    for rule in definition.rules:
        check = reference_ranges.get(rule.name)
        if check is None:
            continue
        if rule.feature not in features.names or not hasattr(phases, rule.phase):
            continue
        phase_range = getattr(phases, rule.phase)
        value = phase_feature_value(features, phase_range, rule.feature)
        directed_check = RangeCheck(check.low, check.high, check.buffer_ratio, rule.direction)
        event = evaluate_range(rule.name, value, directed_check)
        if event is not None:
            events.append(event)
    return events


def evaluate_rule_definitions(
    features: GeometryFeatures,
    phases: PhaseBoundaries,
    rules: list[RuleDefinition],
    *,
    view: str,
) -> list[RuleResult]:
    """Evaluate declarative rules and retain pass/skip outcomes for reporting."""
    results: list[RuleResult] = []
    for rule in rules:
        if view not in rule.compatible_views:
            results.append(RuleResult(
                rule.name, "insufficient_data", None,
                rule.reference.low, rule.reference.high, 0,
                reason="incompatible_camera_view",
            ))
            continue
        if not phases.valid:
            results.append(RuleResult(
                rule.name, "insufficient_data", None,
                rule.reference.low, rule.reference.high, 0,
                reason=phases.invalid_reason or "invalid_stroke_phases",
            ))
            continue
        if rule.feature not in features.names or not hasattr(phases, rule.phase):
            results.append(RuleResult(
                rule.name, "insufficient_data", None,
                rule.reference.low, rule.reference.high, 0,
                reason="missing_feature_or_phase",
            ))
            continue
        start, end = getattr(phases, rule.phase)
        values = features.column(rule.feature)[start:end]
        confidence = features.confidence[:, features.names.index(rule.feature)][start:end]
        valid = np.isfinite(values) & (confidence >= rule.min_confidence)
        valid_count = int(valid.sum())
        phase_length = end - start
        if (
            valid_count < rule.min_valid_frames
            or valid_count / max(phase_length, 1) < rule.min_valid_ratio
        ):
            results.append(RuleResult(
                rule.name, "insufficient_data", None,
                rule.reference.low, rule.reference.high, valid_count,
                reason="too_few_confident_frames",
            ))
            continue
        observed = float(np.median(values[valid]))
        event = evaluate_range(rule.name, observed, rule.reference)
        status = "pass" if event is None else event.severity
        reason = None
        if rule.phase == "contact_estimated" and phases.contact_confidence < 0.5:
            status = "review"
            reason = "low_contact_confidence"
        results.append(RuleResult(
            rule.name,
            status,
            observed,
            rule.reference.low,
            rule.reference.high,
            valid_count,
            reason=reason,
        ))
    return results
