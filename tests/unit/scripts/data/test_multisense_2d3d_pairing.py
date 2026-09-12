import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[4]
    / "scripts/data/export/build_multisense_2d3d_pairs.py"
)
SPEC = importlib.util.spec_from_file_location("build_multisense_2d3d_pairs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_video_time():
    assert MODULE.parse_video_time("00.05.59.845") == pytest.approx(359.845)


def test_manual_clip_metadata_reads_mirrored_path(tmp_path):
    path = (
        tmp_path / "train/forehand_clear/side/Sub09"
        / "Forehand Clear Side Video-00.05.59.845-00.06.03.309-seg25.npz"
    )

    metadata = MODULE.manual_clip_metadata(path, tmp_path)

    assert metadata[:3] == ("Sub09", "forehand_clear", "side")
    assert metadata[3:] == pytest.approx((359.845, 363.309))


def test_estimate_video_epoch_start_aligns_clip_centers_to_annotations():
    clips = [(10.0, 13.0), (20.0, 23.0), (30.0, 33.0)]
    annotations = [
        {"start_time": "1010.0", "stop_time": "1013.0"},
        {"start_time": "1020.0", "stop_time": "1023.0"},
        {"start_time": "1030.0", "stop_time": "1033.0"},
    ]

    result = MODULE.estimate_video_epoch_start(clips, annotations, 995.0)

    assert result == pytest.approx(1000.0)


def test_multi_joint_alignment_recovers_shared_offset():
    np = MODULE.np
    timestamps = np.arange(0.0, 4.0, 0.02)
    pose_time = np.arange(0.0, 3.0, 1 / 30)
    expected_offset = 0.20
    signal_3d_a = np.sin(7 * timestamps) + 0.3 * np.sin(17 * timestamps)
    signal_3d_b = np.cos(5 * timestamps) + 0.2 * np.sin(13 * timestamps)
    signal_2d_a = np.interp(pose_time + expected_offset, timestamps, signal_3d_a)
    signal_2d_b = np.interp(pose_time + expected_offset, timestamps, signal_3d_b)

    offset, correlation = MODULE.multi_joint_motion_alignment_offset(
        pose_time, (signal_2d_a, signal_2d_b), timestamps,
        (signal_3d_a, signal_3d_b), 0.0,
    )

    assert offset == pytest.approx(expected_offset, abs=1 / 60)
    assert correlation > 0.99
