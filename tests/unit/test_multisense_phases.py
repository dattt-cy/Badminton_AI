import numpy as np

from ai_classifier.biomechanics.features import GeometryFeatures
from ai_classifier.biomechanics.phases_3d import detect_multisense_stroke_phases


def test_multisense_phase_detector_trims_long_annotation_around_motion_peak():
    frames = 300
    timestamps = np.arange(frames, dtype=np.float32) / 60
    names = (
        "wrist_speed", "elbow_angular_speed", "knee_angular_speed",
        "hip_angular_speed", "shoulder_angular_speed",
    )
    values = np.zeros((frames, len(names)), dtype=np.float32)
    values[:, 0] = 0.1
    values[205:220, 0] = np.linspace(1, 8, 15)
    values[220:235, 0] = np.linspace(8, 1, 15)
    values[:, 1] = values[:, 0] * 0.5
    features = GeometryFeatures(names, values, np.ones_like(values), timestamps)

    result = detect_multisense_stroke_phases(features)

    assert 140 <= result.start_frame <= 150
    assert 250 <= result.end_frame <= 260
    assert result.motion_peak_frame in range(215, 221)
    assert len(result.features.timestamps) < frames
