"""Check whether a pose NPZ is suitable for geometric technique analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ai_classifier.biomechanics import assess_geometry_quality, extract_geometry_features
from ai_classifier.pose import PoseSequence
from ai_classifier.segmentation import find_motion_proposals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Pose .npz produced by this project")
    parser.add_argument("--handedness", choices=("left", "right"), default="right")
    parser.add_argument("--output", type=Path, help="Optional JSON output")
    args = parser.parse_args()

    with np.load(args.input) as data:
        sequence = PoseSequence(
            np.asarray(data["keypoints"], dtype=np.float32),
            float(data["fps"]),
            int(data["frame_width"]),
            int(data["frame_height"]),
        )
    features = extract_geometry_features(sequence, handedness=args.handedness)
    quality = assess_geometry_quality(sequence, handedness=args.handedness)
    proposals = find_motion_proposals(features, fps=sequence.fps)

    feature_quality = {
        name: float(np.isfinite(features.column(name)).mean())
        for name in features.names
    }
    essential = ("elbow_angle", "shoulder_angle", "wrist_speed")
    essential_quality = min(feature_quality[name] for name in essential)
    reasons: list[str] = []
    if essential_quality < 0.70:
        reasons.append("less than 70% of essential arm measurements are valid")
    if quality.scale_px < 30.0:
        reasons.append("player is too small in the frame for stable geometry")
    if quality.scale_cv > 0.25:
        reasons.append("body scale is unstable across the clip")
    if quality.speed_spike_ratio > 3.0:
        reasons.append("wrist speed contains a likely keypoint spike")
    if not quality.phases_valid:
        reasons.append("stroke phases are incomplete or implausible")
    if not proposals:
        reasons.append("no stable high-motion interval was found")

    result = {
        "input": str(args.input),
        "frame_count": int(len(sequence.keypoints)),
        "fps": sequence.fps,
        "duration_seconds": len(sequence.keypoints) / sequence.fps,
        "handedness": args.handedness,
        "geometry_feasible": not reasons,
        "reasons": reasons,
        "feature_valid_ratio": feature_quality,
        "trajectory_quality": {
            "essential_valid_ratio": quality.valid_ratio,
            "median_scale_px": quality.scale_px,
            "scale_cv": quality.scale_cv,
            "wrist_speed_spike_ratio": quality.speed_spike_ratio,
            "phases_valid": quality.phases_valid,
            "suitable_for_reference": quality.suitable_for_reference,
        },
        "motion_proposals": [
            {
                "start_frame": item.start_frame,
                "end_frame": item.end_frame,
                "peak_frame": item.peak_frame,
                "start_time": item.start_frame / sequence.fps,
                "end_time": item.end_frame / sequence.fps,
                "peak_time": item.peak_frame / sequence.fps,
                "peak_score": item.peak_score,
            }
            for item in proposals
        ],
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
