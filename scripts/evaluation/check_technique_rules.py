"""Run the rule engine on a pose clip, with Motion Proposal + boundary refinement.

Usage:
    python scripts/evaluation/check_technique_rules.py <pose.npz> <technique> [--handedness right]

If the input is a single clean stroke clip, pass --no-segment to skip
Motion Proposal and process the entire clip directly (useful for comparing
against the reference-build workflow).
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


def estimate_view(sequence: PoseSequence) -> str:
    """Estimate front versus side-on geometry from shoulder/torso projection.

    This describes the observable 2D body orientation, which is what matters
    for selecting compatible geometric thresholds. It is not a 3D camera-pose
    estimator.
    """
    keypoints = np.asarray(sequence.keypoints, dtype=np.float32)
    shoulder_width = np.linalg.norm(keypoints[:, 5, :2] - keypoints[:, 6, :2], axis=1)
    mid_shoulder = (keypoints[:, 5, :2] + keypoints[:, 6, :2]) / 2
    mid_hip = (keypoints[:, 11, :2] + keypoints[:, 12, :2]) / 2
    torso_length = np.linalg.norm(mid_shoulder - mid_hip, axis=1)
    confidence = np.min(keypoints[:, [5, 6, 11, 12], 2], axis=1)
    valid = (confidence >= 0.3) & (torso_length > 1e-6)
    if not valid.any():
        return "front"
    projected_ratio = float(np.median(shoulder_width[valid] / torso_length[valid]))
    return "side" if projected_ratio < 0.35 else "front"


def analyse_clip(
    sequence: PoseSequence,
    technique: str,
    rules: list[RuleDefinition],
    handedness: str,
    label: str,
    view: str,
) -> tuple[object, list]:
    """Run phase detection + rule checks on one clean stroke clip."""
    features = extract_geometry_features(sequence, handedness=handedness)
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
    results = evaluate_rule_definitions(features, phases, rules, view=view)
    for result, rule in zip(results, rules):
        observed = "n/a" if result.observed is None else f"{result.observed:.3f}"
        reason = f" reason={result.reason}" if result.reason else ""
        accepted = _accepted_condition(rule)
        print(
            f"  {label}: [{result.status}] {result.rule_name}: observed={observed} "
            f"accepted {accepted} "
            f"valid_frames={result.valid_frames}{reason}"
        )
    return phases, results


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
        "--view", choices=("auto", "front", "side", "generic"), default="auto",
        help="Reference geometry to use. Auto selects from projected body orientation.",
    )
    parser.add_argument("--output", type=Path, help="Optional structured JSON report")
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
    selected_view = estimate_view(sequence) if args.view == "auto" else args.view
    if selected_view not in technique.views:
        available = ", ".join(sorted(technique.views))
        raise ValueError(
            f"Technique '{args.technique}' has no '{selected_view}' view; available: {available}"
        )
    reference_path = technique.views[selected_view].reference
    if not reference_path.exists():
        print(f"Reference profile not found: {reference_path}", file=sys.stderr)
        print(
            f"Run: python scripts/data/build_reference_profiles.py {args.technique} "
            f"--view {selected_view}", file=sys.stderr,
        )
        sys.exit(1)

    reference_document = yaml.safe_load(reference_path.read_text(encoding="utf-8"))
    reference_status = reference_document.get("reference_status", "provisional")
    reference_sample_count = int(reference_document.get("sample_count", 0))
    rules = load_rule_definitions(reference_path)

    print(f"clip: {args.input.name}  frames={len(sequence.keypoints)}"
          f"  fps={sequence.fps:.1f}  technique={args.technique}  view={selected_view}"
          f"  reference={reference_path.name}")
    print(
        f"Reference status: {reference_status}  samples={reference_sample_count}/"
        f"{technique.minimum_reference_clips} required"
    )
    report = {
        "input": str(args.input),
        "technique": args.technique,
        "view": selected_view,
        "reference": str(reference_path),
        "reference_status": reference_status,
        "reference_sample_count": reference_sample_count,
        "minimum_reference_clips": technique.minimum_reference_clips,
        "frame_count": len(sequence.keypoints),
        "fps": sequence.fps,
        "strokes": [],
    }

    if args.no_segment:
        # Treat the entire sequence as one pre-trimmed stroke.
        print("Mode: no-segment (treating full clip as one stroke)")
        phases, results = analyse_clip(
            sequence, args.technique, rules, args.handedness, "stroke-1", selected_view
        )
        report["strokes"].append({
            "stroke_id": 1,
            "start_frame": 0,
            "end_frame": len(sequence.keypoints) - 1,
            "phases": asdict(phases),
            "rules": _serialize_results(results, rules),
            "geometry_assessment": _overall_assessment(results, reference_status),
        })
        _write_report(args.output, report)
        return

    # --- Full pipeline: Motion Proposal → Boundary Refinement → Rule check ---
    features_full = extract_geometry_features(sequence, handedness=args.handedness)
    proposals = find_motion_proposals(features_full, fps=sequence.fps)
    raw_proposal_count = len(proposals)
    proposals = merge_motion_proposals(proposals, features_full, fps=sequence.fps)
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
        phases, results = analyse_clip(
            clip_seq, args.technique, rules, args.handedness, label, selected_view
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
            "rules": _serialize_results(results, rules),
            "geometry_assessment": _overall_assessment(results, reference_status),
        })
    report["raw_motion_proposals"] = raw_proposal_count
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


def _serialize_results(results: list, rules: list[RuleDefinition]) -> list[dict]:
    output = []
    for result, rule in zip(results, rules):
        item = asdict(result)
        item["direction"] = rule.direction
        item["accepted_condition"] = _accepted_condition(rule)
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
