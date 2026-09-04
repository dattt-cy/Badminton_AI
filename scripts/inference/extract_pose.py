'''Extract a COCO-17 pose sequence from a video using YOLOv8 Pose.'''

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ai_classifier.pose import YOLOv8PoseEstimator
from ai_classifier.preprocessing import KeypointSmoother


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path, help='Input video path')
    parser.add_argument('output', type=Path, help='Output .npz path')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/pose/yolov8.yaml'),
        help='YOLOv8 Pose YAML configuration',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open('r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file) or {}

    estimator = YOLOv8PoseEstimator(
        model_path=config.get('model', 'yolov8n-pose.pt'),
        confidence=float(config.get('confidence', 0.25)),
        image_size=int(config.get('image_size', 640)),
        device=config.get('device'),
        tracking_distance_weight=float(config.get('tracking_distance_weight', 1.0)),
    )
    sequence = estimator.extract(args.video)
    smoothing = config.get('smoothing', {})
    if smoothing.get('enabled', True):
        sequence = KeypointSmoother(
            alpha=float(smoothing.get('alpha', 0.35)),
            min_confidence=float(smoothing.get('min_confidence', 0.3)),
            max_gap=int(smoothing.get('max_gap', 4)),
        ).smooth(sequence)
    destination = sequence.save(args.output)
    print(
        f'Saved {sequence.keypoints.shape[0]} frames with shape '
        f'{sequence.keypoints.shape} to {destination}'
    )


if __name__ == '__main__':
    main()
