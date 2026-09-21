import json
from pathlib import Path

import numpy as np

from ai_classifier.pose.viewer_export import build_viewer_payload, export_pose_viewer


def _keypoints() -> np.ndarray:
    values = np.zeros((3, 17, 3), dtype=np.float32)
    for frame in range(3):
        for joint in range(17):
            values[frame, joint] = (100 + joint + frame, 200 + joint, 0.9)
    return values


def test_build_viewer_payload_normalizes_and_adds_stable_depth() -> None:
    payload = build_viewer_payload(
        _keypoints(), fps=30.0, frame_width=640, frame_height=480
    )

    assert payload["layout"] == "coco17"
    assert payload["frame_count"] == 3
    assert len(payload["frames"][0]) == 17
    assert payload["frames"][0][5][2] > 0
    assert payload["frames"][0][6][2] < 0
    assert payload["frames"][0][5][3] == 0.9


def test_export_pose_viewer_writes_compact_json(tmp_path: Path) -> None:
    pose = tmp_path / "pose.npz"
    output = tmp_path / "pose_viewer.json"
    np.savez(
        pose,
        keypoints=_keypoints(),
        fps=np.float32(30),
        frame_width=np.int32(640),
        frame_height=np.int32(480),
    )

    assert export_pose_viewer(pose, output) == output
    assert json.loads(output.read_text(encoding="utf-8"))["frame_count"] == 3
