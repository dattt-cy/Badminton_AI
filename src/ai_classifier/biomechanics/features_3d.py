"""Geometry features for MultiSenseBadminton 21-joint 3D skeletons."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .features import GeometryFeatures


MULTISENSE_JOINTS = (
    "Hips", "RightUpLeg", "RightLeg", "RightFoot", "LeftUpLeg",
    "LeftLeg", "LeftFoot", "Spine", "Spine1", "Spine2", "Spine3",
    "Neck", "Head", "RightShoulder", "RightArm", "RightForeArm",
    "RightHand", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
)
MULTISENSE_JOINT_INDEX = {name: index for index, name in enumerate(MULTISENSE_JOINTS)}


def extract_multisense_geometry_features(
    positions: NDArray[np.floating], timestamps: NDArray[np.floating], *,
    handedness: str = "right",
) -> GeometryFeatures:
    """Extract view-independent features from flattened or shaped XYZ data."""
    if handedness not in {"left", "right"}:
        raise ValueError("handedness must be left or right")
    xyz = np.asarray(positions, dtype=np.float32)
    if xyz.ndim == 2 and xyz.shape[1] == 63:
        xyz = xyz.reshape(-1, 21, 3)
    if xyz.ndim != 3 or xyz.shape[1:] != (21, 3):
        raise ValueError(f"Expected positions with shape (T, 63) or (T, 21, 3), got {xyz.shape}")
    time_s = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if len(time_s) != len(xyz):
        raise ValueError("timestamps and positions must have the same number of frames")
    if len(time_s) and (not np.isfinite(time_s).all() or np.any(np.diff(time_s) <= 0)):
        raise ValueError("timestamps must be finite and strictly increasing")

    joint = MULTISENSE_JOINT_INDEX
    side = "Right" if handedness == "right" else "Left"
    shoulder, elbow, wrist = joint[f"{side}Arm"], joint[f"{side}ForeArm"], joint[f"{side}Hand"]
    hip, knee, ankle = joint[f"{side}UpLeg"], joint[f"{side}Leg"], joint[f"{side}Foot"]
    hips, upper_torso = joint["Hips"], joint["Spine3"]
    angle_specs = (
        ("elbow_angle", shoulder, elbow, wrist),
        ("shoulder_angle", joint[f"{side}Shoulder"], shoulder, elbow),
        ("hip_angle", hips, hip, knee),
        ("knee_angle", hip, knee, ankle),
    )
    names, columns = [], []
    for name, first, center, last in angle_specs:
        names.append(name)
        columns.append(_joint_angle(xyz[:, first], xyz[:, center], xyz[:, last]))

    left_shoulder, right_shoulder = joint["LeftArm"], joint["RightArm"]
    scale = np.linalg.norm(xyz[:, left_shoulder] - xyz[:, right_shoulder], axis=1)
    torso = xyz[:, upper_torso] - xyz[:, hips]
    torso_norm = np.linalg.norm(torso, axis=1)
    torso_lean = np.degrees(np.arccos(np.clip(np.abs(torso[:, 1]) / np.maximum(torso_norm, 1e-9), 0, 1)))
    names.append("torso_lean")
    columns.append(torso_lean.astype(np.float32))

    measures = (
        ("wrist_shoulder_distance", np.linalg.norm(xyz[:, wrist] - xyz[:, shoulder], axis=1)),
        ("wrist_height", xyz[:, wrist, 1] - xyz[:, shoulder, 1]),
        ("elbow_height", xyz[:, elbow, 1] - xyz[:, shoulder, 1]),
        ("stance_width", np.linalg.norm(xyz[:, joint["LeftFoot"]] - xyz[:, joint["RightFoot"]], axis=1)),
        (
            "elbow_torso_distance",
            _point_to_line_distance(xyz[:, elbow], xyz[:, upper_torso], xyz[:, hips]),
        ),
    )
    for name, values in measures:
        names.append(name)
        columns.append(_normalize(values, scale))

    motion = (
        ("wrist_speed", _point_speed(xyz[:, wrist], scale, time_s)),
        ("elbow_angular_speed", _angular_speed(columns[0], time_s)),
        ("shoulder_angular_speed", _angular_speed(columns[1], time_s)),
        ("hip_angular_speed", _angular_speed(columns[2], time_s)),
        ("knee_angular_speed", _angular_speed(columns[3], time_s)),
    )
    for name, values in motion:
        names.append(name)
        columns.append(values)
    values = np.column_stack(columns).astype(np.float32)
    confidence = np.where(np.isfinite(values), 1.0, 0.0).astype(np.float32)
    relative_time = (time_s - time_s[0]).astype(np.float32) if len(time_s) else time_s.astype(np.float32)
    return GeometryFeatures(tuple(names), values, confidence, relative_time)


def _joint_angle(first, center, last):
    a, b = first - center, last - center
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    result = np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1) / np.maximum(denominator, 1e-9), -1, 1))).astype(np.float32)
    result[denominator <= 1e-9] = np.nan
    return result


def _normalize(values, scale):
    result = np.full(len(values), np.nan, dtype=np.float32)
    valid = np.isfinite(values) & np.isfinite(scale) & (scale > 1e-6)
    result[valid] = np.asarray(values, dtype=np.float32)[valid] / scale[valid]
    return result


def _point_to_line_distance(points, line_start, line_end):
    axis = line_end - line_start
    offset = points - line_start
    denominator = np.linalg.norm(axis, axis=1)
    result = np.full(len(points), np.nan, dtype=np.float32)
    valid = denominator > 1e-9
    result[valid] = (
        np.linalg.norm(np.cross(offset[valid], axis[valid]), axis=1)
        / denominator[valid]
    )
    return result


def _point_speed(points, scale, time_s):
    result = np.full(len(points), np.nan, dtype=np.float32)
    if len(points) < 2:
        return result
    delta_t = np.diff(time_s)
    pair_scale = (scale[1:] + scale[:-1]) / 2
    valid = (delta_t > 0) & np.isfinite(pair_scale) & (pair_scale > 1e-6)
    values = np.linalg.norm(np.diff(points, axis=0), axis=1) / np.maximum(delta_t, 1e-9) / pair_scale
    result[1:][valid] = values[valid]
    return _median_filter(result)


def _angular_speed(angles, time_s):
    result = np.full(len(angles), np.nan, dtype=np.float32)
    if len(angles) < 2:
        return result
    smoothed = _median_filter(angles)
    delta_t = np.diff(time_s)
    valid = np.isfinite(smoothed[1:]) & np.isfinite(smoothed[:-1]) & (delta_t > 0)
    result[1:][valid] = (np.abs(np.diff(smoothed)) / np.maximum(delta_t, 1e-9))[valid]
    return result


def _median_filter(values, window=5):
    result = np.full(np.asarray(values).shape, np.nan, dtype=np.float32)
    radius = window // 2
    for index in range(len(values)):
        chunk = np.asarray(values)[max(0, index-radius):min(len(values), index+radius+1)]
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            result[index] = np.median(finite)
    return result
