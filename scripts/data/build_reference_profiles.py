"""Build a reference profile (mục 10) from confirmed single-stroke clips.

Loads every pose `.npz` in a directory (one stroke per clip, per
`data/raw/README.md`), computes per-phase feature medians for the mục 12
rule proxies below and the kinetic-chain timing diagnostic, then writes the
aggregated percentile-based ranges to a YAML config for `evaluate_range`.

Phase boundaries come from `detect_stroke_phases`, a semi-automatic proposal
(mục 10.2) — spot-check a few clips before trusting the output on new data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from ai_classifier.biomechanics import (
    assess_geometry_quality,
    detect_stroke_phases,
    estimate_pose_scale_px,
    extract_geometry_features,
    extract_kinetic_chain_timing,
    load_technique_registry,
)
from ai_classifier.error_detection import phase_feature_value
from ai_classifier.pose import PoseSequence
from ai_classifier.reference_analysis import build_kinetic_chain_reference, build_range_checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a geometry reference from curated single-stroke pose clips."
    )
    parser.add_argument("technique", help="Technique key declared in the registry")
    parser.add_argument("--registry", type=Path,
                        default=Path("configs/biomechanics/techniques.yaml"))
    parser.add_argument("--view", choices=("generic", "front", "side"), default="generic",
                        help="View-specific clip list and output reference to build")
    parser.add_argument(
        "--pose-dir",
        type=Path,
        help="Directory of single-stroke pose .npz files "
        "(default: data/processed/poses/<technique>/single_player)",
    )
    parser.add_argument("--handedness", choices=("left", "right"), default="right")
    parser.add_argument("--buffer-ratio", type=float, default=0.05)
    parser.add_argument(
        "--min-scale-px",
        type=float,
        default=30.0,
        help="Skip clips whose median body scale (shoulder width px, or torso "
        "length fallback) is below this; small scale means keypoint jitter "
        "dominates the normalized signal.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Use only the N clips with the largest body scale as the curated "
        "reference set (about 10-20 confirmed-good clips are recommended). "
        "Default: use every clip that passes --min-scale-px.",
    )
    parser.add_argument(
        "--clip-list", type=Path,
        help="UTF-8 text file containing one visually approved .npz filename per line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output YAML path (default: configs/biomechanics/<technique>_reference.yaml)",
    )
    parser.add_argument(
        "--pending-review", action="store_true",
        help="Force a provisional profile when clips have only automatic quality review.",
    )
    return parser.parse_args()


def load_pose(path: Path) -> PoseSequence:
    with np.load(path) as data:
        return PoseSequence(
            keypoints=np.asarray(data["keypoints"], dtype=np.float32),
            fps=float(data["fps"]),
            frame_width=int(data["frame_width"]),
            frame_height=int(data["frame_height"]),
        )


def main() -> None:
    args = parse_args()
    registry = load_technique_registry(args.registry)
    technique = registry.technique(args.technique)
    if args.view not in technique.views:
        raise ValueError(f"Technique '{args.technique}' has no '{args.view}' view")
    view_config = technique.views[args.view]
    pose_dir = args.pose_dir or technique.pose_dir
    output_path = args.output or view_config.reference
    clip_list = args.clip_list or view_config.clip_list
    if not pose_dir.is_dir():
        raise FileNotFoundError(f"Pose directory not found: {pose_dir}")

    clip_paths = sorted(pose_dir.rglob("*_pose.npz"))
    if not clip_paths:
        raise FileNotFoundError(f"No .npz pose files found in {pose_dir}")

    if clip_list:
        approved = {
            line.strip() for line in clip_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        def approved_key(path: Path) -> str:
            return path.relative_to(pose_dir).as_posix()

        clip_paths = [
            path for path in clip_paths
            if path.name in approved or approved_key(path) in approved
        ]
        found = {path.name for path in clip_paths} | {approved_key(path) for path in clip_paths}
        missing_approved = approved - found
        if missing_approved:
            raise FileNotFoundError(f"Approved pose clips not found: {sorted(missing_approved)}")

    rule_specs = technique.rules
    rule_samples: dict[str, list[float]] = {rule.name: [] for rule in rule_specs}

    # Pass 1: load every clip and rank by pixel scale (mục 9.1 quality gate).
    loaded: list[tuple[Path, PoseSequence, float]] = []
    for clip_path in clip_paths:
        sequence = load_pose(clip_path)
        scale_px = estimate_pose_scale_px(sequence)
        loaded.append((clip_path, sequence, scale_px))

    eligible = []
    quality_by_path = {}
    for item in loaded:
        quality = assess_geometry_quality(item[1], handedness=args.handedness)
        quality_by_path[item[0]] = quality
        if quality.suitable_for_reference and item[2] >= args.min_scale_px:
            eligible.append(item)
    # Compare by path only: PoseSequence equality would compare keypoint
    # arrays elementwise and raise on the ambiguous truth value.
    eligible_paths = {clip_path for clip_path, _, _ in eligible}
    rejected = [item for item in loaded if item[0] not in eligible_paths]
    eligible.sort(key=lambda item: item[2], reverse=True)
    selected = eligible[: args.top_n] if args.top_n else eligible
    dropped_by_top_n = eligible[args.top_n :] if args.top_n else []

    print(f"{'clip':<40}{'scale_px':<10}{'status'}")
    for clip_path, _, scale_px in loaded:
        if clip_path in [p for p, _, _ in rejected]:
            quality = quality_by_path[clip_path]
            print(f"{clip_path.name:<40}{scale_px:<10.1f}skipped: quality "
                  f"valid={quality.valid_ratio:.2f} scale_cv={quality.scale_cv:.2f} "
                  f"spike={quality.speed_spike_ratio:.2f} phases={quality.phases_valid}")
    for clip_path, _, scale_px in dropped_by_top_n:
        print(f"{clip_path.name:<40}{scale_px:<10.1f}skipped: outside top {args.top_n}")

    timings = []
    phase_valid_count = 0
    clips_used: list[str] = []
    for clip_path, sequence, scale_px in selected:
        features = extract_geometry_features(sequence, handedness=args.handedness)
        phases = detect_stroke_phases(features)
        timing = extract_kinetic_chain_timing(features)
        timings.append(timing)
        print(
            f"{clip_path.name:<40}{scale_px:<10.1f}used  "
            f"phases_valid={phases.valid}  chain_valid={timing.valid}  "
            f"chain_order_ok={timing.order_correct}"
        )

        if not phases.valid:
            continue
        phase_valid_count += 1
        clips_used.append(clip_path.relative_to(pose_dir).as_posix())
        for rule in rule_specs:
            if rule.feature not in features.names or not hasattr(phases, rule.phase):
                raise ValueError(
                    f"Rule '{rule.name}' uses unavailable feature/phase "
                    f"'{rule.feature}/{rule.phase}'"
                )
            phase_range = getattr(phases, rule.phase)
            value = phase_feature_value(features, phase_range, rule.feature)
            if np.isfinite(value):
                rule_samples[rule.name].append(value)

    print(
        f"\n{len(selected)}/{len(clip_paths)} clips used as the reference set "
        f"({len(rejected)} skipped for scale, {len(dropped_by_top_n)} skipped outside top-n)."
    )

    chain_valid_count = sum(timing.valid for timing in timings)
    print(
        f"{phase_valid_count}/{len(selected)} used clips had usable phase boundaries; "
        f"{chain_valid_count}/{len(selected)} had usable kinetic-chain timing "
        "(diagnostic only, excluded from MVP rules)."
    )

    rule_ranges = build_range_checks(rule_samples, buffer_ratio=args.buffer_ratio)
    missing = sorted(set(rule_samples) - set(rule_ranges))
    if missing:
        print(f"Skipped rules with too few valid samples: {missing}")
    if not rule_ranges:
        raise RuntimeError("No rule produced a reference range; check phase detection and pose quality.")

    chain_reference = build_kinetic_chain_reference(timings, buffer_ratio=args.buffer_ratio)
    document = {
        "technique": args.technique,
        "view": args.view,
        "registry": str(args.registry),
        "handedness": args.handedness,
        "sample_count": phase_valid_count,
        "minimum_reference_clips": technique.minimum_reference_clips,
        "reference_status": (
            "ready"
            if phase_valid_count >= technique.minimum_reference_clips and not args.pending_review
            else "provisional"
        ),
        "min_scale_px": args.min_scale_px,
        "top_n": args.top_n,
        "clips_used": clips_used,
        "phase_feature_ranges": {
            name: {"low": check.low, "high": check.high, "buffer_ratio": check.buffer_ratio}
            for name, check in rule_ranges.items()
        },
        "rules": {
            rule.name: {
                "feature": rule.feature,
                "phase": rule.phase,
                "direction": rule.direction,
                "low": rule_ranges[rule.name].low,
                "high": rule_ranges[rule.name].high,
                "buffer_ratio": rule_ranges[rule.name].buffer_ratio,
                "min_confidence": rule.min_confidence,
                "min_valid_frames": rule.min_valid_frames,
                "min_valid_ratio": rule.min_valid_ratio,
                "compatible_views": list(rule.compatible_views),
            }
            for rule in rule_specs
            if rule.name in rule_ranges
        },
        "kinetic_chain_lags_diagnostic_only": {
            name: {"low": check.low, "high": check.high, "buffer_ratio": check.buffer_ratio}
            for name, check in chain_reference.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(document, output_file, allow_unicode=True, sort_keys=False)
    print(f"Wrote reference profile: {output_path}")


if __name__ == "__main__":
    main()
