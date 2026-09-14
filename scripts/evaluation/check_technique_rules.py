"""Run the rule engine on a pose clip, with Motion Proposal + boundary refinement.

Usage:
    python scripts/evaluation/check_technique_rules.py <pose.npz> <technique> --view side

Short clips up to five seconds with at most one motion proposal are treated as
one complete stroke automatically. Use --segmentation-mode multi for long or
multi-stroke processing; --no-segment remains as a compatibility alias for
--segmentation-mode single.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from ai_classifier.biomechanics import (
    detect_stroke_phases,
    evaluate_observable_criteria,
    extract_geometry_features,
    load_technique_registry,
    summarize_stroke_geometry,
)
from ai_classifier.error_detection import (
    RangeCheck,
    RuleDefinition,
    RuleResult,
    evaluate_rule_definitions,
    phase_feature_value,
)
from ai_classifier.pose import PoseSequence
from ai_classifier.segmentation import (
    find_motion_proposals,
    merge_motion_proposals,
    refine_all_proposals,
)


def load_pose(path: Path) -> PoseSequence:
    with np.load(path) as data:
        return PoseSequence(
            keypoints=np.asarray(data["keypoints"], dtype=np.float32),
            fps=float(data["fps"]),
            frame_width=int(data["frame_width"]),
            frame_height=int(data["frame_height"]),
        )


def load_reference(path: Path) -> dict[str, RangeCheck]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        name: RangeCheck(entry["low"], entry["high"], entry["buffer_ratio"])
        for name, entry in document["phase_feature_ranges"].items()
    }


def load_rule_definitions(path: Path, rule_specs: tuple = ()) -> list[RuleDefinition]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "rules" not in document:
        raise ValueError(f"Reference has no declarative rules; rebuild it: {path}")
    policies = {rule.name: rule.orientation_requirement for rule in rule_specs}
    return [
        RuleDefinition(
            name=name,
            feature=entry["feature"],
            phase=entry["phase"],
            direction=entry["direction"],
            reference=RangeCheck(
                entry["low"], entry["high"], entry.get("buffer_ratio", 0.05),
                entry["direction"],
            ),
            min_confidence=entry.get("min_confidence", 0.6),
            min_valid_frames=entry.get("min_valid_frames", 3),
            min_valid_ratio=entry.get("min_valid_ratio", 0.6),
            compatible_views=tuple(entry.get("compatible_views", ("front", "side"))),
            orientation_requirement=entry.get(
                "orientation_requirement", policies.get(name, "aligned")
            ),
        )
        for name, entry in document["rules"].items()
    ]


SIDE_VIEW_MAX_RATIO = 0.30
FRONT_VIEW_MIN_RATIO = 0.45
SINGLE_STROKE_MAX_SECONDS = 5.0
PHASE_NAMES = (
    "preparation",
    "backswing",
    "forward_swing",
    "contact_estimated",
    "follow_through",
)


def estimate_view_with_ratio(
    sequence: PoseSequence,
    frame_range: tuple[int, int] | None = None,
) -> tuple[str, float | None]:
    """Estimate front versus side-on geometry from shoulder/torso projection.

    This describes the observable 2D body orientation, which is what matters
    for selecting compatible geometric thresholds. It is not a 3D camera-pose
    estimator.
    """
    keypoints = np.asarray(sequence.keypoints, dtype=np.float32)
    if frame_range is not None:
        start, end = frame_range
        keypoints = keypoints[start:end]
    if not len(keypoints):
        return "unknown", None
    shoulder_width = np.linalg.norm(keypoints[:, 5, :2] - keypoints[:, 6, :2], axis=1)
    mid_shoulder = (keypoints[:, 5, :2] + keypoints[:, 6, :2]) / 2
    mid_hip = (keypoints[:, 11, :2] + keypoints[:, 12, :2]) / 2
    torso_length = np.linalg.norm(mid_shoulder - mid_hip, axis=1)
    confidence = np.min(keypoints[:, [5, 6, 11, 12], 2], axis=1)
    valid = (confidence >= 0.3) & (torso_length > 1e-6)
    if not valid.any():
        return "unknown", None
    projected_ratio = float(np.median(shoulder_width[valid] / torso_length[valid]))
    if projected_ratio < SIDE_VIEW_MAX_RATIO:
        return "side", projected_ratio
    if projected_ratio > FRONT_VIEW_MIN_RATIO:
        return "front", projected_ratio
    return "oblique", projected_ratio


def estimate_view(sequence: PoseSequence) -> str:
    """Backward-compatible two-way view estimate for callers that require it."""
    view, ratio = estimate_view_with_ratio(sequence)
    if view in {"front", "side"}:
        return view
    if ratio is None:
        return "front"
    return "side" if ratio < 0.35 else "front"


def estimate_phase_views(sequence: PoseSequence, phases: object) -> dict[str, dict]:
    """Return projected orientation and ratio independently for every phase."""
    output = {}
    for phase_name in PHASE_NAMES:
        view, ratio = estimate_view_with_ratio(sequence, getattr(phases, phase_name))
        output[phase_name] = {"view": view, "projected_ratio": ratio}
    return output


def should_use_full_sequence(
    sequence: PoseSequence,
    merged_proposal_count: int,
    *,
    max_seconds: float = SINGLE_STROKE_MAX_SECONDS,
) -> bool:
    """Treat short uploads with at most one stroke proposal as pre-trimmed."""
    duration = len(sequence.keypoints) / sequence.fps if sequence.fps > 0 else float("inf")
    return duration <= max_seconds and merged_proposal_count <= 1


def evaluate_rules_fixed_view(
    features: object,
    phases: object,
    sequence: PoseSequence,
    rules: list[RuleDefinition],
    view: str,
    *,
    camera_view_reliable: bool = True,
    feature_validation: dict | None = None,
    technique: str | None = None,
) -> tuple[list[RuleResult], list[dict], dict[str, dict]]:
    """Use one camera reference and treat body rotation as uncertainty.

    Projected orientation can change as the athlete rotates, but the camera
    view cannot. Therefore it may downgrade or reject a measurement; it must
    never switch the reference profile in the middle of a stroke.
    """
    phase_orientations = estimate_phase_views(sequence, phases)
    base_results = evaluate_rule_definitions(features, phases, rules, view=view)
    results: list[RuleResult] = []
    contexts: list[dict] = []
    for rule, result in zip(rules, base_results):
        orientation = phase_orientations[rule.phase]["view"]
        if not camera_view_reliable:
            result = _replace_result(
                result, status="insufficient_data", reason="ambiguous_camera_view"
            )
        elif rule.orientation_requirement == "any":
            pass
        elif view in {"front", "side"} and orientation in {"front", "side"}:
            if (
                orientation != view
                and rule.orientation_requirement == "aligned"
                and result.status != "insufficient_data"
            ):
                result = _replace_result(
                    result, status="insufficient_data", reason="out_of_plane_rotation"
                )
            elif orientation != view and result.status != "insufficient_data":
                result = _replace_result(
                    result, status="review", reason="body_orientation_mismatch"
                )
        elif view in {"front", "side"} and orientation == "oblique":
            if result.status != "insufficient_data":
                result = _replace_result(
                    result, status="review", reason="oblique_body_orientation"
                )
        elif orientation == "unknown" and result.status != "insufficient_data":
            result = _replace_result(
                result, status="insufficient_data", reason="unknown_body_orientation"
            )
        validation_entry = _feature_validation_entry(
            feature_validation, technique, view, rule.feature
        )
        validation_status = validation_entry.get("status", "missing")
        if result.status != "insufficient_data":
            if validation_status in {"rejected", "missing"}:
                result = _replace_result(
                    result,
                    status="insufficient_data",
                    reason="feature_not_validated_2d_against_3d",
                )
            elif validation_status == "review":
                result = _replace_result(
                    result,
                    status="review",
                    reason="feature_requires_2d_3d_review",
                )
        results.append(result)
        contexts.append({
            "phase_view": orientation,
            "evaluated_views": [view],
            "rules": {view: rule},
            "phase": rule.phase,
            "feature_validation": validation_entry,
        })
    return results, contexts, phase_orientations


def _feature_validation_entry(
    document: dict | None, technique: str | None, view: str, feature: str
) -> dict:
    if document is None:
        return {"status": "not_applied"}
    try:
        return document["techniques"][technique]["views"][view]["features"][feature]
    except KeyError:
        return {"status": "missing"}


def evaluate_relative_2d_indicators(
    sequence: PoseSequence,
    phases: object,
    summary: object,
    technique: str,
    view: str,
    registry: dict | None,
    *,
    handedness: str = "right",
    floor_angle_degrees: float = 0.0,
) -> list[dict]:
    """Evaluate holdout-validated within-video indicators without hard grading."""
    if registry is None or not phases.valid:
        return []
    try:
        rules = registry["techniques"][technique]["views"][view]["rules"]
    except KeyError:
        return []
    features = extract_geometry_features(
        sequence, handedness=handedness,
        floor_angle_degrees=floor_angle_degrees,
    )
    prep_reach = phase_feature_value(
        features, phases.preparation, "wrist_shoulder_distance"
    )
    contact_reach = phase_feature_value(
        features, phases.contact_estimated, "wrist_shoulder_distance"
    )
    prep_height = phase_feature_value(features, phases.preparation, "wrist_height")
    contact_height = phase_feature_value(
        features, phases.contact_estimated, "wrist_height"
    )
    follow_height = phase_feature_value(features, phases.follow_through, "wrist_height")
    elbow_column = features.column("elbow_angle")
    reach_column = features.column("wrist_shoulder_distance")
    finite_elbow = elbow_column[np.isfinite(elbow_column)]
    finite_reach = reach_column[np.isfinite(reach_column)]
    elbow_p10, elbow_p90 = (
        np.percentile(finite_elbow, [10, 90])
        if finite_elbow.size else (float("nan"), float("nan"))
    )
    reach_p10, reach_p90 = (
        np.percentile(finite_reach, [10, 90])
        if finite_reach.size else (float("nan"), float("nan"))
    )
    values = {
        "elbow_extension_delta": summary.elbow_extension_delta,
        "contact_reach_gain": contact_reach - prep_reach,
        "contact_wrist_height": contact_height,
        "contact_wrist_height_gain": contact_height - prep_height,
        "followthrough_wrist_drop": contact_height - follow_height,
        "followthrough_elbow_angle": phase_feature_value(
            features, phases.follow_through, "elbow_angle"
        ),
        "wrist_vertical_excursion": summary.wrist_vertical_excursion,
        "wrist_path_length": summary.wrist_path_length,
        "elbow_angle_excursion": elbow_p90 - elbow_p10,
        "peak_elbow_extension": elbow_p90,
        "reach_excursion": reach_p90 - reach_p10,
        "peak_reach": reach_p90,
    }
    output = []
    for name, rule in rules.items():
        value = float(values.get(name, float("nan")))
        registry_status = rule["status"]
        if not np.isfinite(value):
            status, reason = "unavailable", "invalid_measurement"
        elif registry_status == "rejected":
            status, reason = "unavailable", "relative_indicator_not_validated"
        elif registry_status == "review":
            status, reason = "review", "indicator_requires_review"
        else:
            passes = (
                value >= float(rule["threshold"])
                if rule["direction"] == "higher"
                else value <= float(rule["threshold"])
            )
            status = "pass" if passes else "review"
            reason = None if passes else "below_validated_expert_indicator"
        output.append({
            "feature": name,
            "observed": value if np.isfinite(value) else None,
            "threshold": float(rule["threshold"]),
            "direction": rule["direction"],
            "status": status,
            "reason": reason,
            "registry_status": registry_status,
            "holdout_metrics": rule.get("test_metrics"),
        })
    return output


def _replace_result(result: RuleResult, *, status: str, reason: str) -> RuleResult:
    return RuleResult(
        result.rule_name, status, result.observed,
        result.reference_low, result.reference_high,
        result.valid_frames, reason=reason,
    )


def analyse_clip(
    sequence: PoseSequence,
    technique: str,
    rules_by_view: dict[str, list[RuleDefinition]],
    handedness: str,
    label: str,
    view: str,
    floor_angle_degrees: float = 0.0,
    camera_view_reliable: bool = True,
    swing_peak_frame: int | None = None,
    feature_validation: dict | None = None,
) -> tuple[object, object, list[RuleResult], list[dict], dict[str, dict]]:
    """Run phase detection + rule checks on one clean stroke clip."""
    features = extract_geometry_features(
        sequence,
        handedness=handedness,
        floor_angle_degrees=floor_angle_degrees,
    )
    phases = detect_stroke_phases(features, swing_peak_frame=swing_peak_frame)
    summary = summarize_stroke_geometry(features, phases)
    print(
        f"  phases_valid={phases.valid}  "
        f"prep={phases.preparation}  backswing={phases.backswing}  "
        f"forward_swing={phases.forward_swing}  contact={phases.contact_estimated}  "
        f"follow_through={phases.follow_through}  "
        f"swing_peak_confidence={phases.contact_confidence:.3f}  "
        f"source={phases.peak_source}  "
        f"candidates={list(phases.contact_candidates)}"
    )
    if not phases.valid:
        print(f"  {label}: Insufficient data: implausible or incomplete stroke phases.")
    rules = rules_by_view[view]
    results, contexts, phase_views = evaluate_rules_fixed_view(
        features, phases, sequence, rules, view,
        camera_view_reliable=camera_view_reliable,
        feature_validation=feature_validation,
        technique=technique,
    )
    rendered_views = ", ".join(
        f"{name}={entry['view']}"
        + (
            f"({entry['projected_ratio']:.3f})"
            if entry["projected_ratio"] is not None else ""
        )
        for name, entry in phase_views.items()
    )
    print(f"  body_orientation_by_phase: {rendered_views}")
    for result, context in zip(results, contexts):
        observed = "n/a" if result.observed is None else f"{result.observed:.3f}"
        reason = f" reason={result.reason}" if result.reason else ""
        accepted = _accepted_conditions(context)
        print(
            f"  {label}: [{result.status}] {result.rule_name}: observed={observed} "
            f"accepted {accepted} "
            f"valid_frames={result.valid_frames}{reason}"
        )
    return phases, summary, results, contexts, phase_views


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?",
                        default=Path("outputs/test__1__pose.npz"),
                        help="Pose .npz file")
    parser.add_argument("technique", nargs="?", default="forehand_lift",
                        help="Technique key declared in the registry")
    parser.add_argument("--registry", type=Path,
                        default=Path("configs/biomechanics/techniques.yaml"))
    parser.add_argument("--handedness", choices=("left", "right"), default="right")
    parser.add_argument(
        "--floor-angle-degrees", type=float, default=0.0,
        help="Observed slope of a horizontal court line, positive downward to the right.",
    )
    parser.add_argument(
        "--relative-2d-rules", type=Path,
        default=Path("configs/biomechanics/multisense_relative_2d_rules.yaml"),
    )
    parser.add_argument("--no-relative-2d-rules", action="store_true")
    parser.add_argument(
        "--view", choices=("front", "side", "generic"), required=True,
        help="Fixed camera view for the whole clip; this is never inferred from body rotation.",
    )
    parser.add_argument(
        "--reference", type=Path,
        help="Optional reference override for controlled A/B validation.",
    )
    parser.add_argument(
        "--swing-peak-frame", type=int,
        help="Manual swing-peak frame for a pre-trimmed single-stroke clip.",
    )
    parser.add_argument("--output", type=Path, help="Optional structured JSON report")
    parser.add_argument(
        "--feature-validation", type=Path,
        default=Path(
            "configs/biomechanics/multisense_feature_validation_projected.yaml"
        ),
        help="MultiSense 2D-vs-3D feature reliability registry",
    )
    parser.add_argument(
        "--no-feature-validation", action="store_true",
        help="Disable 3D-derived feature gates for controlled A/B audits only",
    )
    parser.add_argument(
        "--segmentation-mode",
        choices=("auto", "single", "multi"),
        default="auto",
        help="Auto uses the full sequence for clips up to 5 seconds with at most "
             "one proposal; single always uses the full clip; multi always segments.",
    )
    parser.add_argument(
        "--no-segment",
        action="store_true",
        help="Skip Motion Proposal; treat the entire clip as one stroke (for "
             "pre-trimmed clips only).",
    )
    args = parser.parse_args()

    feature_validation = None
    if not args.no_feature_validation:
        if not args.feature_validation.is_file():
            raise FileNotFoundError(
                f"Feature validation registry not found: {args.feature_validation}"
            )
        feature_validation = yaml.safe_load(
            args.feature_validation.read_text(encoding="utf-8")
        )
    relative_2d_registry = None
    if not args.no_relative_2d_rules:
        if not args.relative_2d_rules.is_file():
            raise FileNotFoundError(
                f"Relative 2D rule registry not found: {args.relative_2d_rules}"
            )
        relative_2d_registry = yaml.safe_load(
            args.relative_2d_rules.read_text(encoding="utf-8")
        )

    registry = load_technique_registry(args.registry)
    technique = registry.technique(args.technique)
    sequence = load_pose(args.input)
    detected_view, projected_ratio = estimate_view_with_ratio(sequence)
    camera_view_reliable = detected_view in {args.view, "oblique", "unknown"}
    selected_view = args.view
    requested_views = (selected_view,)
    missing_views = [view for view in requested_views if view not in technique.views]
    if missing_views:
        available = ", ".join(sorted(technique.views))
        raise ValueError(
            f"Technique '{args.technique}' has no '{missing_views[0]}' view; available: {available}"
        )
    reference_paths = {
        view: args.reference or technique.views[view].reference for view in requested_views
    }
    reference_documents = {}
    rules_by_view = {}
    for view, reference_path in reference_paths.items():
        if not reference_path.exists():
            print(f"Reference profile not found: {reference_path}", file=sys.stderr)
            print(
                "Run: python scripts/data/references/build_reference_profiles.py "
                f"{args.technique} "
                f"--view {view}", file=sys.stderr,
            )
            sys.exit(1)
        reference_documents[view] = yaml.safe_load(
            reference_path.read_text(encoding="utf-8")
        )
        rules_by_view[view] = load_rule_definitions(reference_path, technique.rules)
    reference_status = (
        "ready"
        if all(doc.get("reference_status") == "ready" for doc in reference_documents.values())
        else "provisional"
    )
    reference_sample_counts = {
        view: int(doc.get("sample_count", 0))
        for view, doc in reference_documents.items()
    }

    print(f"clip: {args.input.name}  frames={len(sequence.keypoints)}"
          f"  fps={sequence.fps:.1f}  technique={args.technique}  view={selected_view}"
          f"  references={','.join(path.name for path in reference_paths.values())}")
    print(
        f"Reference status: {reference_status}  samples={reference_sample_counts}/"
        f"{technique.minimum_reference_clips} required"
    )
    report = {
        "input": str(args.input),
        "technique": args.technique,
        "view": selected_view,
        "view_selection": {
            "requested": args.view,
            "detected_body_projection": detected_view,
            "projected_shoulder_torso_ratio": projected_ratio,
            "reliable": camera_view_reliable,
        },
        "references": {view: str(path) for view, path in reference_paths.items()},
        "reference_status": reference_status,
        "reference_sample_counts": reference_sample_counts,
        "minimum_reference_clips": technique.minimum_reference_clips,
        "feature_validation": (
            {"enabled": True, "path": str(args.feature_validation),
             "holdout_rule": feature_validation.get("holdout_rule")}
            if feature_validation is not None else {"enabled": False}
        ),
        "relative_2d_rules": {
            "enabled": relative_2d_registry is not None,
            "path": str(args.relative_2d_rules) if relative_2d_registry else None,
            "split_rule": (
                relative_2d_registry.get("split_rule")
                if relative_2d_registry else None
            ),
        },
        "frame_count": len(sequence.keypoints),
        "fps": sequence.fps,
        "floor_angle_degrees": args.floor_angle_degrees,
        "strokes": [],
    }

    features_full = extract_geometry_features(
        sequence,
        handedness=args.handedness,
        floor_angle_degrees=args.floor_angle_degrees,
    )
    proposals = find_motion_proposals(features_full, fps=sequence.fps)
    raw_proposal_count = len(proposals)
    proposals = merge_motion_proposals(proposals, features_full, fps=sequence.fps)
    segmentation_mode = "single" if args.no_segment else args.segmentation_mode
    use_full_sequence = segmentation_mode == "single" or (
        segmentation_mode == "auto"
        and should_use_full_sequence(sequence, len(proposals))
    )
    report["segmentation_mode"] = segmentation_mode
    report["raw_motion_proposals"] = raw_proposal_count
    report["merged_strokes"] = len(proposals)
    if args.swing_peak_frame is not None and not use_full_sequence:
        raise ValueError("--swing-peak-frame requires a single/full-sequence clip")

    if use_full_sequence:
        # Treat the entire sequence as one pre-trimmed stroke.
        mode_reason = "explicit" if segmentation_mode == "single" else "auto short-single-stroke"
        print(f"Mode: full sequence ({mode_reason})")
        phases, summary, results, contexts, phase_views = analyse_clip(
            sequence, args.technique, rules_by_view, args.handedness,
            "stroke-1", selected_view, args.floor_angle_degrees,
            camera_view_reliable, args.swing_peak_frame,
            feature_validation,
        )
        relative_indicators = evaluate_relative_2d_indicators(
            sequence, phases, summary, args.technique, selected_view,
            relative_2d_registry, handedness=args.handedness,
            floor_angle_degrees=args.floor_angle_degrees,
        )
        observable_criteria = evaluate_observable_criteria(
            sequence, phases, args.technique, selected_view,
            handedness=args.handedness,
            camera_view_reliable=camera_view_reliable,
        )
        report["strokes"].append({
            "stroke_id": 1,
            "start_frame": 0,
            "end_frame": len(sequence.keypoints) - 1,
            "phases": _serialize_phases(phases),
            "phase_views": phase_views,
            "phase_orientations": phase_views,
            "summary": _serialize_summary(summary, selected_view),
            "relative_2d_indicators": relative_indicators,
            "observable_criteria": observable_criteria,
            "user_feedback": _practical_user_feedback(
                results, contexts, relative_indicators, observable_criteria
            ),
            "summary_reference_comparison": _compare_summary(
                summary, reference_documents[selected_view], phase_views, selected_view
            ),
            "rules": _serialize_results(results, contexts, phases),
            "highlights": _build_highlights(results, contexts, phases),
            "geometry_assessment": _overall_assessment(results, reference_status),
        })
        _write_report(args.output, report)
        return

    # --- Full pipeline: Motion Proposal → Boundary Refinement → Rule check ---
    print(f"Motion proposals found: {raw_proposal_count}; after merge: {len(proposals)}")

    if not proposals:
        print("No high-motion intervals detected. Check pose quality or try --no-segment.")
        return

    strokes = refine_all_proposals(proposals, features_full, fps=sequence.fps)

    for i, stroke in enumerate(strokes, start=1):
        label = f"stroke-{i}"
        print(
            f"\n{label}: video frames [{stroke.start_frame}, {stroke.end_frame}]  "
            f"peak={stroke.peak_frame}  "
            f"duration={( stroke.end_frame - stroke.start_frame) / sequence.fps:.2f}s"
        )
        clip_seq = stroke.clip(sequence)
        phases, summary, results, contexts, phase_views = analyse_clip(
            clip_seq, args.technique, rules_by_view, args.handedness, label,
            selected_view, args.floor_angle_degrees, camera_view_reliable,
            feature_validation=feature_validation,
        )
        relative_indicators = evaluate_relative_2d_indicators(
            clip_seq, phases, summary, args.technique, selected_view,
            relative_2d_registry, handedness=args.handedness,
            floor_angle_degrees=args.floor_angle_degrees,
        )
        observable_criteria = evaluate_observable_criteria(
            clip_seq, phases, args.technique, selected_view,
            handedness=args.handedness,
            camera_view_reliable=camera_view_reliable,
        )
        report["strokes"].append({
            "stroke_id": i,
            "start_frame": stroke.start_frame,
            "end_frame": stroke.end_frame,
            "motion_peak_frame": stroke.peak_frame,
            "estimated_contact_frame": stroke.start_frame + phases.contact_frame,
            "estimated_swing_peak_frame": stroke.start_frame + phases.contact_frame,
            "candidate_contact_frames": [
                stroke.start_frame + frame for frame in phases.contact_candidates
            ],
            "candidate_swing_peak_frames": [
                stroke.start_frame + frame for frame in phases.contact_candidates
            ],
            "source_proposals": stroke.proposal.source_count,
            "phases": _serialize_phases(phases),
            "phase_views": phase_views,
            "phase_orientations": phase_views,
            "summary": _serialize_summary(summary, selected_view),
            "relative_2d_indicators": relative_indicators,
            "observable_criteria": observable_criteria,
            "user_feedback": _practical_user_feedback(
                results, contexts, relative_indicators, observable_criteria
            ),
            "summary_reference_comparison": _compare_summary(
                summary, reference_documents[selected_view], phase_views, selected_view
            ),
            "rules": _serialize_results(results, contexts, phases),
            "highlights": _build_highlights(results, contexts, phases),
            "geometry_assessment": _overall_assessment(results, reference_status),
        })
    report["merged_strokes"] = len(strokes)
    _write_report(args.output, report)


def _write_report(path: Path | None, report: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote JSON report: {path}")


def _accepted_condition(rule: RuleDefinition) -> str:
    if rule.direction == "lower":
        return f">= {rule.reference.low:.3f}"
    if rule.direction == "upper":
        return f"<= {rule.reference.high:.3f}"
    return f"[{rule.reference.low:.3f}, {rule.reference.high:.3f}]"


def _accepted_conditions(context: dict) -> str:
    rules = context["rules"]
    if not rules:
        return "n/a"
    rendered = [
        f"{view}:{_accepted_condition(rule)}"
        for view, rule in rules.items()
    ]
    return "; ".join(rendered)


RULE_LABELS = {
    "preparation_arm_position_outlier": "Vị trí tay khi chuẩn bị",
    "stance_too_narrow": "Độ rộng chân khi chuẩn bị",
    "elbow_too_close_to_torso": "Khoảng cách khuỷu với thân",
    "elbow_too_low_in_backswing": "Độ cao khuỷu khi backswing",
    "backswing_elbow_angle_outlier": "Góc khuỷu khi backswing",
    "backswing_wrist_height_outlier": "Độ cao cổ tay khi backswing",
    "excessive_torso_lean_in_forward_swing": "Độ nghiêng thân khi tăng tốc",
    "insufficient_reach": "Độ vươn tay gần swing peak",
    "arm_not_extended_near_contact": "Độ duỗi tay gần swing peak",
    "contact_wrist_height_outlier": "Độ cao cổ tay gần swing peak",
    "limited_follow_through": "Biên độ follow-through",
    "follow_through_elbow_angle_outlier": "Góc khuỷu khi follow-through",
}

FEATURE_LABELS_VI = {
    "elbow_angle": "Góc khuỷu",
    "wrist_shoulder_distance": "Độ vươn tay",
    "wrist_height": "Độ cao cổ tay",
    "elbow_torso_distance": "Khoảng cách khuỷu–thân",
    "torso_lean": "Độ nghiêng thân",
    "stance_width": "Độ rộng chân",
    "elbow_extension_delta": "Mức duỗi thêm của khuỷu",
    "contact_reach_gain": "Mức tăng độ vươn gần contact",
    "contact_wrist_height": "Độ cao cổ tay gần contact",
    "contact_wrist_height_gain": "Mức nâng cổ tay",
    "followthrough_wrist_drop": "Mức hạ cổ tay follow-through",
    "wrist_vertical_excursion": "Biên độ cổ tay theo chiều dọc",
    "wrist_path_length": "Quãng đường cổ tay",
    "elbow_angle_excursion": "Biên độ góc khuỷu",
    "peak_elbow_extension": "Góc khuỷu duỗi lớn nhất",
    "reach_excursion": "Biên độ vươn tay",
    "peak_reach": "Độ vươn tay lớn nhất",
}

REASON_LABELS_VI = {
    "feature_not_validated_2d_against_3d": "Phép đo chưa vượt kiểm định MultiSense.",
    "feature_requires_2d_3d_review": "Phép đo chỉ đủ dùng làm chỉ số tham khảo.",
    "out_of_plane_rotation": "Cơ thể xoay lệch khỏi mặt phẳng camera.",
    "incompatible_camera_view": "Góc camera này không phù hợp với phép đo.",
    "too_few_confident_frames": "Không đủ frame có keypoint đáng tin.",
    "low_contact_confidence": "Thời điểm contact chưa đủ chắc chắn.",
    "unknown_body_orientation": "Không xác định được hướng cơ thể.",
    "oblique_body_orientation": "Cơ thể đang ở góc chéo so với camera.",
    "body_orientation_mismatch": "Hướng cơ thể không khớp góc quay đã chọn.",
}


def _measurement_unit(feature: str) -> str:
    return "degree" if feature in {
        "elbow_angle", "torso_lean", "elbow_extension_delta",
        "elbow_angle_excursion", "peak_elbow_extension",
    } else "body_scale_ratio"


def _practical_user_feedback(
    results: list[RuleResult],
    contexts: list[dict],
    relative_indicators: list[dict],
    observable_criteria: list[dict] | None = None,
) -> dict:
    """Build conservative, score-free feedback for phone-video users."""
    good_signals, observations, unavailable = [], [], []
    for result, context in zip(results, contexts):
        rule = next(iter(context["rules"].values()))
        validation = context.get("feature_validation") or {"status": "missing"}
        item = {
            "feature": rule.feature,
            "label": FEATURE_LABELS_VI.get(rule.feature, rule.feature),
            "phase": rule.phase,
            "observed": result.observed,
            "unit": _measurement_unit(rule.feature),
            "validation_status": validation.get("status", "missing"),
        }
        if result.status == "pass" and validation.get("status") == "validated":
            item["message"] = "Có dấu hiệu phù hợp với vùng tham chiếu đã kiểm định."
            good_signals.append(item)
        elif result.status == "review":
            item["message"] = (
                "Chỉ số được hiển thị để tham khảo; chưa đủ bằng chứng kết luận đúng hoặc sai."
            )
            observations.append(item)
        else:
            item["reason"] = REASON_LABELS_VI.get(
                result.reason, "Phép đo chưa đủ tin cậy để đánh giá."
            )
            unavailable.append(item)

    for indicator in relative_indicators:
        item = {
            "feature": indicator["feature"],
            "label": FEATURE_LABELS_VI.get(
                indicator["feature"], indicator["feature"]
            ),
            "observed": indicator["observed"],
            "threshold": indicator["threshold"],
            "direction": indicator["direction"],
            "unit": _measurement_unit(indicator["feature"]),
            "validation_status": indicator["registry_status"],
        }
        if indicator["status"] == "pass" and indicator["registry_status"] == "validated":
            item["message"] = "Có dấu hiệu tương tự nhóm Expert trong kiểm định holdout."
            good_signals.append(item)
        elif indicator["status"] == "review":
            item["message"] = "Chỉ số tương đối cần được xem lại, không phải kết luận lỗi."
            observations.append(item)
        # Rejected experimental indicators remain in `relative_2d_indicators`
        # for audit, but are intentionally hidden from user-facing feedback.

    heuristic_observations = []
    for criterion in observable_criteria or []:
        if criterion["status"] == "unavailable":
            unavailable.append({
                "feature": criterion["criterion"],
                "label": criterion["label"],
                "observed": criterion["observed"],
                "unit": criterion["unit"],
                "validation_status": "experimental_heuristic",
                "reason": criterion["message"],
            })
        elif criterion["status"] in {"observed", "needs_review"}:
            heuristic_observations.append(criterion)

    if good_signals:
        overall = "observable_good_signals"
    elif observations or heuristic_observations:
        overall = "review_available"
    else:
        overall = "insufficient_data"
    return {
        "mode": "experimental_geometry_preview",
        "overall": overall,
        "score": None,
        "good_signals": good_signals,
        "observations_for_review": observations,
        "heuristic_observations": heuristic_observations,
        "not_assessed": unavailable,
        "disclaimer": (
            "Kết quả mô tả tín hiệu hình học quan sát được từ video; "
            "không phải điểm số hoặc chẩn đoán kỹ thuật chuyên môn."
        ),
    }


def _serialize_phases(phases: object) -> dict:
    item = asdict(phases)
    item["estimated_swing_peak_frame"] = phases.contact_frame
    item["swing_peak_confidence"] = phases.contact_confidence
    item["candidate_swing_peak_frames"] = list(phases.contact_candidates)
    return item


def _serialize_summary(summary: object, view: str | None = None) -> dict:
    item = asdict(summary)
    for name, value in list(item.items()):
        if isinstance(value, float) and not np.isfinite(value):
            item[name] = None
    item["balance_offset_interpretation"] = {
        "front": "lateral",
        "side": "forward",
    }.get(view, "image_horizontal")
    return item


def _compare_summary(
    summary: object,
    reference_document: dict,
    phase_orientations: dict[str, dict] | None = None,
    view: str = "generic",
) -> list[dict]:
    ranges = reference_document.get("stroke_summary_ranges", {})
    values = {
        "elbow_extension_delta": summary.elbow_extension_delta,
        "wrist_path_length": summary.wrist_path_length,
        "wrist_vertical_excursion": summary.wrist_vertical_excursion,
        "balance_offset_preparation": summary.balance_offset_preparation,
        **{
            f"phase_duration_ratio.{name}": value
            for name, value in summary.phase_duration_ratios.items()
        },
    }
    output = []
    for name, value in values.items():
        if name == "phase_duration_ratio.contact_estimated":
            output.append({
                "feature": name,
                "observed": float(value),
                "status": "unavailable",
                "reason": "algorithm_defined_window_not_coaching_metric",
            })
            continue
        reference = ranges.get(name)
        if reference is None or not np.isfinite(value):
            output.append({
                "feature": name,
                "observed": float(value) if np.isfinite(value) else None,
                "status": "unavailable",
                "reason": "reference_not_built" if reference is None else "invalid_measurement",
            })
            continue
        low, high = float(reference["low"]), float(reference["high"])
        position = "within_reference" if low <= value <= high else "outside_reference"
        reliability = _summary_reliability(name, phase_orientations, view)
        if reliability == "insufficient":
            output.append({
                "feature": name,
                "observed": float(value),
                "status": "unavailable",
                "reference_position": position,
                "reference_low": low,
                "reference_high": high,
                "reason": "out_of_plane_rotation",
                "sample_count": int(reference.get("sample_count", 0)),
            })
            continue
        output.append({
            "feature": name,
            "observed": float(value),
            "reference_low": low,
            "reference_high": high,
            "status": "review" if reliability == "review" else position,
            "reference_position": position,
            "measurement_confidence": reliability,
            "sample_count": int(reference.get("sample_count", 0)),
        })
    return output


def _summary_reliability(
    feature: str,
    phase_orientations: dict[str, dict] | None,
    view: str,
) -> str:
    if not phase_orientations or view not in {"front", "side"}:
        return "high"
    if feature == "elbow_extension_delta":
        phases, strict = ("backswing", "contact_estimated"), True
    elif feature == "balance_offset_preparation":
        phases, strict = ("preparation",), False
    elif feature in {"wrist_path_length", "wrist_vertical_excursion"}:
        phases, strict = tuple(PHASE_NAMES), False
    else:
        return "high"
    orientations = [phase_orientations[name]["view"] for name in phases]
    if strict and any(item not in {view, "oblique"} for item in orientations):
        return "insufficient"
    if any(item != view for item in orientations):
        return "review"
    return "high"


def _review_frame(context: dict, phases: object) -> int:
    phase = context.get("phase")
    if phase == "contact_estimated":
        return int(phases.contact_frame)
    if phase and hasattr(phases, phase):
        start, end = getattr(phases, phase)
        return int((start + max(start, end - 1)) // 2)
    return int(phases.contact_frame)


def _evidence_text(result: RuleResult) -> str:
    if result.observed is None:
        return "Không đủ phép đo tin cậy."
    value = result.observed
    if result.status == "pass":
        return (
            f"{value:.1f}, nằm trong vùng tham chiếu "
            f"[{result.reference_low:.1f}, {result.reference_high:.1f}]."
        )
    if value < result.reference_low:
        delta = result.reference_low - value
        return f"{value:.1f}, thấp hơn cận tham chiếu {delta:.1f}."
    if value > result.reference_high:
        delta = value - result.reference_high
        return f"{value:.1f}, cao hơn cận tham chiếu {delta:.1f}."
    return f"{value:.1f}; cần xem lại do độ tin cậy phép chiếu."


def _serialize_results(
    results: list, contexts: list[dict], phases: object
) -> list[dict]:
    output = []
    for result, context in zip(results, contexts):
        item = asdict(result)
        item["finding"] = RULE_LABELS.get(
            result.rule_name, result.rule_name.replace("_", " ")
        )
        item["evidence"] = _evidence_text(result)
        item["phase"] = context.get("phase")
        item["review_frame"] = _review_frame(context, phases)
        item["phase_view"] = context["phase_view"]
        item["evaluated_views"] = context["evaluated_views"]
        item["accepted_conditions"] = {
            view: _accepted_condition(rule)
            for view, rule in context["rules"].items()
        }
        item["feature_validation"] = context.get("feature_validation")
        output.append(item)
    return output


def _build_highlights(
    results: list[RuleResult], contexts: list[dict], phases: object
) -> dict[str, list[dict]]:
    groups = {"strengths": [], "needs_review": [], "insufficient_data": []}
    for result, context in zip(results, contexts):
        entry = {
            "rule_name": result.rule_name,
            "finding": RULE_LABELS.get(
                result.rule_name, result.rule_name.replace("_", " ")
            ),
            "evidence": _evidence_text(result),
            "review_frame": _review_frame(context, phases),
        }
        if result.status == "pass":
            groups["strengths"].append(entry)
        elif result.status == "insufficient_data":
            groups["insufficient_data"].append(entry)
        else:
            groups["needs_review"].append(entry)
    return groups


def _overall_assessment(results: list, reference_status: str = "ready") -> str:
    """Collapse rule outcomes without turning missing evidence into a pass."""
    statuses = {result.status for result in results}
    if not results or statuses == {"insufficient_data"}:
        return "insufficient_data"
    if reference_status != "ready":
        return "review"
    if "deviation" in statuses:
        return "deviation"
    if "review" in statuses:
        return "review"
    if "insufficient_data" in statuses:
        return "insufficient_data"
    return "pass"


if __name__ == "__main__":
    main()
