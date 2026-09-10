"""Run the rule engine on a pose clip, with Motion Proposal + boundary refinement.

Usage:
    python scripts/evaluation/check_technique_rules.py <pose.npz> <technique> [--handedness right]

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
    extract_geometry_features,
    load_technique_registry,
)
from ai_classifier.error_detection import (
    RangeCheck,
    RuleDefinition,
    RuleResult,
    evaluate_rule_definitions,
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


def load_rule_definitions(path: Path) -> list[RuleDefinition]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "rules" not in document:
        raise ValueError(f"Reference has no declarative rules; rebuild it: {path}")
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


def _combine_oblique_results(
    front: RuleResult,
    side: RuleResult,
) -> RuleResult:
    """Require front/side agreement before making an oblique-view assertion."""
    if front.status == side.status:
        return front
    usable = [item for item in (front, side) if item.status != "insufficient_data"]
    observed = next((item.observed for item in usable if item.observed is not None), None)
    valid_frames = max((item.valid_frames for item in usable), default=0)
    return RuleResult(
        front.rule_name,
        "review" if usable else "insufficient_data",
        observed,
        front.reference_low,
        front.reference_high,
        valid_frames,
        reason="view_disagreement" if usable else "insufficient_view_data",
    )


def evaluate_rules_by_phase_view(
    features: object,
    phases: object,
    sequence: PoseSequence,
    rules_by_view: dict[str, list[RuleDefinition]],
) -> tuple[list[RuleResult], list[dict], dict[str, dict]]:
    """Evaluate each phase against front, side, or both reference profiles."""
    phase_views = estimate_phase_views(sequence, phases)
    ordered_names = [rule.name for rule in rules_by_view["front"]]
    lookup = {
        view: {rule.name: rule for rule in rules}
        for view, rules in rules_by_view.items()
    }
    results: list[RuleResult] = []
    contexts: list[dict] = []
    for name in ordered_names:
        front_rule = lookup["front"][name]
        phase_view = phase_views[front_rule.phase]["view"]
        available = [
            view for view in ("front", "side")
            if name in lookup[view] and view in lookup[view][name].compatible_views
        ]
        if phase_view in {"front", "side"}:
            chosen = phase_view if phase_view in available else None
            if chosen is None:
                rule = front_rule
                result = evaluate_rule_definitions(features, phases, [rule], view=phase_view)[0]
                contexts.append({"phase_view": phase_view, "evaluated_views": [], "rules": {}})
            else:
                rule = lookup[chosen][name]
                result = evaluate_rule_definitions(features, phases, [rule], view=chosen)[0]
                contexts.append({
                    "phase_view": phase_view,
                    "evaluated_views": [chosen],
                    "rules": {chosen: rule},
                })
        elif phase_view == "oblique" and len(available) == 2:
            front_result = evaluate_rule_definitions(
                features, phases, [lookup["front"][name]], view="front"
            )[0]
            side_result = evaluate_rule_definitions(
                features, phases, [lookup["side"][name]], view="side"
            )[0]
            result = _combine_oblique_results(front_result, side_result)
            rule = front_rule
            contexts.append({
                "phase_view": "oblique",
                "evaluated_views": ["front", "side"],
                "rules": {
                    "front": lookup["front"][name],
                    "side": lookup["side"][name],
                },
            })
        elif available:
            chosen = available[0]
            rule = lookup[chosen][name]
            result = evaluate_rule_definitions(features, phases, [rule], view=chosen)[0]
            if phase_view in {"oblique", "unknown"} and result.status != "insufficient_data":
                result = RuleResult(
                    result.rule_name, "review", result.observed,
                    result.reference_low, result.reference_high,
                    result.valid_frames, reason="single_view_evidence",
                )
            contexts.append({
                "phase_view": phase_view,
                "evaluated_views": [chosen],
                "rules": {chosen: rule},
            })
        else:
            rule = front_rule
            result = evaluate_rule_definitions(features, phases, [rule], view="side")[0]
            contexts.append({"phase_view": phase_view, "evaluated_views": [], "rules": {}})
        results.append(result)
    return results, contexts, phase_views


def analyse_clip(
    sequence: PoseSequence,
    technique: str,
    rules_by_view: dict[str, list[RuleDefinition]],
    handedness: str,
    label: str,
    view: str,
    floor_angle_degrees: float = 0.0,
) -> tuple[object, list[RuleResult], list[dict], dict[str, dict]]:
    """Run phase detection + rule checks on one clean stroke clip."""
    features = extract_geometry_features(
        sequence,
        handedness=handedness,
        floor_angle_degrees=floor_angle_degrees,
    )
    phases = detect_stroke_phases(features)
    print(
        f"  phases_valid={phases.valid}  "
        f"prep={phases.preparation}  backswing={phases.backswing}  "
        f"forward_swing={phases.forward_swing}  contact={phases.contact_estimated}  "
        f"follow_through={phases.follow_through}  "
        f"contact_confidence={phases.contact_confidence:.3f}  "
        f"candidates={list(phases.contact_candidates)}"
    )
    if not phases.valid:
        print(f"  {label}: Insufficient data: implausible or incomplete stroke phases.")
    if view == "phase-aware":
        results, contexts, phase_views = evaluate_rules_by_phase_view(
            features, phases, sequence, rules_by_view
        )
        rendered_views = ", ".join(
            f"{name}={entry['view']}"
            + (
                f"({entry['projected_ratio']:.3f})"
                if entry["projected_ratio"] is not None else ""
            )
            for name, entry in phase_views.items()
        )
        print(f"  phase_views: {rendered_views}")
    else:
        rules = rules_by_view[view]
        results = evaluate_rule_definitions(features, phases, rules, view=view)
        contexts = [
            {"phase_view": view, "evaluated_views": [view], "rules": {view: rule}}
            for rule in rules
        ]
        phase_views = {name: {"view": view, "projected_ratio": None} for name in PHASE_NAMES}
    for result, context in zip(results, contexts):
        observed = "n/a" if result.observed is None else f"{result.observed:.3f}"
        reason = f" reason={result.reason}" if result.reason else ""
        accepted = _accepted_conditions(context)
        print(
            f"  {label}: [{result.status}] {result.rule_name}: observed={observed} "
            f"accepted {accepted} "
            f"valid_frames={result.valid_frames}{reason}"
        )
    return phases, results, contexts, phase_views


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
        "--view", choices=("auto", "front", "side", "generic"), default="auto",
        help="Reference geometry to use. Auto selects from projected body orientation.",
    )
    parser.add_argument("--output", type=Path, help="Optional structured JSON report")
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

    registry = load_technique_registry(args.registry)
    technique = registry.technique(args.technique)
    sequence = load_pose(args.input)
    phase_aware = args.view == "auto" and {"front", "side"} <= set(technique.views)
    selected_view = "phase-aware" if phase_aware else (
        estimate_view(sequence) if args.view == "auto" else args.view
    )
    requested_views = ("front", "side") if phase_aware else (selected_view,)
    missing_views = [view for view in requested_views if view not in technique.views]
    if missing_views:
        available = ", ".join(sorted(technique.views))
        raise ValueError(
            f"Technique '{args.technique}' has no '{missing_views[0]}' view; available: {available}"
        )
    reference_paths = {
        view: technique.views[view].reference for view in requested_views
    }
    reference_documents = {}
    rules_by_view = {}
    for view, reference_path in reference_paths.items():
        if not reference_path.exists():
            print(f"Reference profile not found: {reference_path}", file=sys.stderr)
            print(
                f"Run: python scripts/data/build_reference_profiles.py {args.technique} "
                f"--view {view}", file=sys.stderr,
            )
            sys.exit(1)
        reference_documents[view] = yaml.safe_load(
            reference_path.read_text(encoding="utf-8")
        )
        rules_by_view[view] = load_rule_definitions(reference_path)
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
        "references": {view: str(path) for view, path in reference_paths.items()},
        "reference_status": reference_status,
        "reference_sample_counts": reference_sample_counts,
        "minimum_reference_clips": technique.minimum_reference_clips,
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

    if use_full_sequence:
        # Treat the entire sequence as one pre-trimmed stroke.
        mode_reason = "explicit" if segmentation_mode == "single" else "auto short-single-stroke"
        print(f"Mode: full sequence ({mode_reason})")
        phases, results, contexts, phase_views = analyse_clip(
            sequence, args.technique, rules_by_view, args.handedness,
            "stroke-1", selected_view, args.floor_angle_degrees
        )
        report["strokes"].append({
            "stroke_id": 1,
            "start_frame": 0,
            "end_frame": len(sequence.keypoints) - 1,
            "phases": asdict(phases),
            "phase_views": phase_views,
            "rules": _serialize_results(results, contexts),
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
        phases, results, contexts, phase_views = analyse_clip(
            clip_seq, args.technique, rules_by_view, args.handedness, label,
            selected_view, args.floor_angle_degrees
        )
        report["strokes"].append({
            "stroke_id": i,
            "start_frame": stroke.start_frame,
            "end_frame": stroke.end_frame,
            "motion_peak_frame": stroke.peak_frame,
            "estimated_contact_frame": stroke.start_frame + phases.contact_frame,
            "candidate_contact_frames": [
                stroke.start_frame + frame for frame in phases.contact_candidates
            ],
            "source_proposals": stroke.proposal.source_count,
            "phases": asdict(phases),
            "phase_views": phase_views,
            "rules": _serialize_results(results, contexts),
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


def _serialize_results(results: list, contexts: list[dict]) -> list[dict]:
    output = []
    for result, context in zip(results, contexts):
        item = asdict(result)
        item["phase_view"] = context["phase_view"]
        item["evaluated_views"] = context["evaluated_views"]
        item["accepted_conditions"] = {
            view: _accepted_condition(rule)
            for view, rule in context["rules"].items()
        }
        output.append(item)
    return output


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
