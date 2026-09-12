"""Conservative phone-video observations not presented as validated coaching rules."""

from __future__ import annotations

import math

import numpy as np

from ai_classifier.pose import PoseSequence


LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


def _median_point(
    keypoints: np.ndarray, frame_range: tuple[int, int], joints: tuple[int, ...],
    *, min_confidence: float = 0.3,
) -> np.ndarray | None:
    start, end = frame_range
    points = keypoints[start:end, joints]
    valid = points[..., 2] >= min_confidence
    coordinates = []
    for frame_points, frame_valid in zip(points, valid):
        if frame_valid.any():
            coordinates.append(np.mean(frame_points[frame_valid, :2], axis=0))
    return np.median(coordinates, axis=0) if coordinates else None


def _torso_scale(keypoints: np.ndarray) -> float:
    confidence = np.min(
        keypoints[:, [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP], 2],
        axis=1,
    )
    shoulder = np.mean(keypoints[:, [LEFT_SHOULDER, RIGHT_SHOULDER], :2], axis=1)
    hip = np.mean(keypoints[:, [LEFT_HIP, RIGHT_HIP], :2], axis=1)
    values = np.linalg.norm(shoulder - hip, axis=1)
    valid = (confidence >= 0.3) & np.isfinite(values) & (values > 1e-6)
    return float(np.median(values[valid])) if valid.any() else math.nan


def _angle(first: np.ndarray, center: np.ndarray, last: np.ndarray) -> float:
    vector_a, vector_b = first - center, last - center
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator <= 1e-9:
        return math.nan
    cosine = np.dot(vector_a, vector_b) / denominator
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _phase_knee_angle(keypoints: np.ndarray, frame_range: tuple[int, int]) -> float:
    values = []
    start, end = frame_range
    for frame in range(start, end):
        per_side = []
        for hip, knee, ankle in (
            (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
            (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
        ):
            if np.min(keypoints[frame, [hip, knee, ankle], 2]) >= 0.3:
                per_side.append(_angle(
                    keypoints[frame, hip, :2], keypoints[frame, knee, :2],
                    keypoints[frame, ankle, :2],
                ))
        finite = [value for value in per_side if math.isfinite(value)]
        if finite:
            values.append(min(finite))
    return float(np.median(values)) if values else math.nan


def _item(
    name: str, label: str, observed: float | None, status: str,
    message: str, *, threshold: object = None, unit: str = "body_scale_ratio",
) -> dict:
    return {
        "criterion": name,
        "label": label,
        "observed": observed,
        "unit": unit,
        "status": status,
        "threshold": threshold,
        "evidence_level": "experimental_heuristic",
        "message": message,
    }


def evaluate_observable_criteria(
    sequence: PoseSequence,
    phases: object,
    technique: str,
    view: str,
    *,
    handedness: str = "right",
) -> list[dict]:
    """Describe visible technique cues without claiming expert validation."""
    if technique not in {"forehand_clear", "backhand_drive"} or not phases.valid:
        return []
    keypoints = np.asarray(sequence.keypoints, dtype=np.float32)
    scale = _torso_scale(keypoints)
    if not math.isfinite(scale) or scale <= 0:
        return []
    racket_shoulder = RIGHT_SHOULDER if handedness == "right" else LEFT_SHOULDER
    racket_wrist = RIGHT_WRIST if handedness == "right" else LEFT_WRIST
    other_shoulder = LEFT_SHOULDER if handedness == "right" else RIGHT_SHOULDER
    other_wrist = LEFT_WRIST if handedness == "right" else RIGHT_WRIST
    contact = phases.contact_estimated
    preparation = phases.preparation
    follow = phases.follow_through
    wrist_contact = _median_point(keypoints, contact, (racket_wrist,))
    shoulder_contact = _median_point(
        keypoints, contact, (LEFT_SHOULDER, RIGHT_SHOULDER)
    )
    hip_contact = _median_point(keypoints, contact, (LEFT_HIP, RIGHT_HIP))
    head_contact = _median_point(keypoints, contact, (0, 1, 2, 3, 4))
    output = []

    if technique == "forehand_clear":
        if wrist_contact is not None and head_contact is not None:
            height = float((head_contact[1] - wrist_contact[1]) / scale)
            status = "observed" if height >= 0 else "needs_review"
            output.append(_item(
                "contact_above_head", "Điểm đánh phía trên đầu", height, status,
                "Cổ tay ở trên vùng đầu tại contact." if status == "observed"
                else "Cổ tay chưa thể hiện rõ vị trí ở trên đầu tại contact.",
                threshold={"minimum": 0.0},
            ))
    else:
        if wrist_contact is not None and shoulder_contact is not None and hip_contact is not None:
            torso_height = max(abs(hip_contact[1] - shoulder_contact[1]), 1e-6)
            drive_height = float((hip_contact[1] - wrist_contact[1]) / torso_height)
            status = "observed" if 0.15 <= drive_height <= 1.30 else "needs_review"
            output.append(_item(
                "contact_drive_height", "Độ cao contact của drive", drive_height, status,
                "Contact nằm trong vùng từ hông đến trên vai phù hợp cú drive."
                if status == "observed" else "Độ cao contact nằm ngoài vùng drive tham khảo.",
                threshold={"minimum": 0.15, "maximum": 1.30}, unit="torso_height_ratio",
            ))

    nose = _median_point(keypoints, contact, (0,))
    if view == "side" and nose is not None and shoulder_contact is not None and wrist_contact is not None:
        facing = float(nose[0] - shoulder_contact[0])
        if abs(facing) >= 0.04 * scale:
            ahead = float(np.sign(facing) * (wrist_contact[0] - shoulder_contact[0]) / scale)
            status = "observed" if ahead >= -0.05 else "needs_review"
            output.append(_item(
                "contact_ahead_of_body", "Contact ở phía trước cơ thể", ahead, status,
                "Tay đánh nằm về phía trước hướng nhìn tại contact."
                if status == "observed" else "Contact có dấu hiệu nằm lùi sau vai.",
                threshold={"minimum": -0.05},
            ))
        else:
            output.append(_item(
                "contact_ahead_of_body", "Contact ở phía trước cơ thể", None,
                "unavailable", "Không xác định chắc chắn hướng nhìn từ camera này.",
            ))

    other_wrist_preparation = _median_point(keypoints, preparation, (other_wrist,))
    other_shoulder_preparation = _median_point(keypoints, preparation, (other_shoulder,))
    other_wrist_contact = _median_point(keypoints, contact, (other_wrist,))
    if (
        other_wrist_preparation is not None and other_shoulder_preparation is not None
        and other_wrist_contact is not None
    ):
        elevation = float(
            (other_shoulder_preparation[1] - other_wrist_preparation[1]) / scale
        )
        drop = float((other_wrist_contact[1] - other_wrist_preparation[1]) / scale)
        if technique == "forehand_clear":
            status = "observed" if elevation >= -0.10 and drop >= 0.05 else "needs_review"
            message = (
                "Tay không thuận có nâng định hướng rồi hạ khi tăng tốc."
                if status == "observed" else
                "Chưa thấy rõ chuỗi nâng–hạ của tay không thuận."
            )
        else:
            distance = float(np.linalg.norm(
                other_wrist_contact - other_shoulder_preparation
            ) / scale)
            status = "observed" if distance >= 0.25 else "needs_review"
            message = (
                "Tay không thuận tạo đối trọng quan sát được."
                if status == "observed" else "Tay không thuận ít tạo đối trọng."
            )
        output.append(_item(
            "non_racket_arm_coordination", "Phối hợp tay không thuận",
            drop if technique == "forehand_clear" else distance,
            status, message,
            threshold={"minimum": 0.05 if technique == "forehand_clear" else 0.25},
        ))

    knee_angle = _phase_knee_angle(keypoints, preparation)
    if math.isfinite(knee_angle):
        threshold = 165.0 if technique == "forehand_clear" else 160.0
        status = "observed" if knee_angle <= threshold else "needs_review"
        output.append(_item(
            "leg_loading", "Sử dụng chân khi chuẩn bị", knee_angle, status,
            "Có độ chùng gối quan sát được." if status == "observed"
            else "Hai gối khá thẳng; có thể tăng độ chùng khi chuẩn bị.",
            threshold={"maximum_degrees": threshold}, unit="degree",
        ))

    hip_preparation = _median_point(keypoints, preparation, (LEFT_HIP, RIGHT_HIP))
    if hip_preparation is not None and hip_contact is not None:
        transfer = float(np.linalg.norm(hip_contact - hip_preparation) / scale)
        status = "observed" if transfer >= 0.08 else "needs_review"
        output.append(_item(
            "body_transfer", "Dịch chuyển thân từ chuẩn bị đến contact", transfer,
            status, "Có chuyển động thân tham gia vào cú đánh."
            if status == "observed" else "Chuyển động thân quan sát được còn ít.",
            threshold={"minimum": 0.08},
        ))

    wrist_follow = _median_point(keypoints, follow, (racket_wrist,))
    if wrist_contact is not None and wrist_follow is not None:
        follow_distance = float(np.linalg.norm(wrist_follow - wrist_contact) / scale)
        status = "observed" if follow_distance >= 0.25 else "needs_review"
        output.append(_item(
            "followthrough_completion", "Biên độ follow-through", follow_distance,
            status, "Tay vợt tiếp tục di chuyển rõ sau contact."
            if status == "observed" else "Follow-through quan sát được còn ngắn.",
            threshold={"minimum": 0.25},
        ))
    return output
