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
    # Loading is a deliberate low-angle moment, not the median posture of a
    # long setup. A lower quartile is robust to one noisy frame while still
    # capturing a sustained knee bend.
    return float(np.percentile(values, 25)) if values else math.nan


def _joint_elevation_values(
    keypoints: np.ndarray,
    frame_range: tuple[int, int],
    wrist: int,
    shoulder: int,
    scale: float,
) -> np.ndarray:
    start, end = frame_range
    confidence = np.min(keypoints[start:end, [wrist, shoulder], 2], axis=1)
    elevation = (
        keypoints[start:end, shoulder, 1] - keypoints[start:end, wrist, 1]
    ) / scale
    return elevation[(confidence >= 0.3) & np.isfinite(elevation)]


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
    camera_view_reliable: bool = True,
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
    contact_confidence = float(getattr(phases, "contact_confidence", 0.0))
    contact_reliable = contact_confidence >= 0.80

    if technique == "forehand_clear":
        if wrist_contact is not None and head_contact is not None:
            height = float((head_contact[1] - wrist_contact[1]) / scale)
            status = (
                "unavailable" if not contact_reliable or -0.15 < height < 0.08
                else "observed" if height >= 0.08
                else "needs_review"
            )
            output.append(_item(
                "contact_above_head", "Điểm chạm cầu phía trên đầu", height, status,
                "Cổ tay ở trên vùng đầu tại thời điểm chạm cầu." if status == "observed"
                else "Chưa thấy rõ cổ tay ở trên đầu tại thời điểm chạm cầu."
                if status == "needs_review"
                else "Không xác định đủ chắc chắn thời điểm tiếp xúc cầu từ pose 2D.",
                threshold={"observed_minimum": 0.08, "review_maximum": -0.15},
            ))
    else:
        if wrist_contact is not None and shoulder_contact is not None and hip_contact is not None:
            torso_height = max(abs(hip_contact[1] - shoulder_contact[1]), 1e-6)
            drive_height = float((hip_contact[1] - wrist_contact[1]) / torso_height)
            status = "observed" if 0.15 <= drive_height <= 1.30 else "needs_review"
            output.append(_item(
                "contact_drive_height", "Độ cao điểm chạm cầu của cú drive", drive_height, status,
                "Điểm chạm cầu nằm trong vùng từ hông đến trên vai."
                if status == "observed" else "Điểm chạm cầu nằm ngoài vùng drive tham khảo.",
                threshold={"minimum": 0.15, "maximum": 1.30}, unit="torso_height_ratio",
            ))

    nose = _median_point(keypoints, contact, (0,))
    if (
        view == "side" and camera_view_reliable and contact_reliable
        and nose is not None and shoulder_contact is not None and wrist_contact is not None
    ):
        facing = float(nose[0] - shoulder_contact[0])
        if abs(facing) >= 0.04 * scale:
            ahead = float(np.sign(facing) * (wrist_contact[0] - shoulder_contact[0]) / scale)
            status = "observed" if ahead >= -0.05 else "needs_review"
            output.append(_item(
                "contact_ahead_of_body", "Điểm chạm cầu ở phía trước cơ thể", ahead, status,
                "Tay đánh nằm về phía trước hướng nhìn khi chạm cầu."
                if status == "observed" else "Điểm chạm cầu có dấu hiệu nằm lùi sau vai.",
                threshold={"minimum": -0.05},
            ))
        else:
            output.append(_item(
                "contact_ahead_of_body", "Điểm chạm cầu ở phía trước cơ thể", None,
                "unavailable", "Không xác định chắc chắn hướng nhìn từ camera này.",
            ))

    precontact = (preparation[0], contact[0])
    other_elevation = _joint_elevation_values(
        keypoints, precontact, other_wrist, other_shoulder, scale
    )
    other_contact_elevation = _joint_elevation_values(
        keypoints, contact, other_wrist, other_shoulder, scale
    )
    if other_elevation.size and other_contact_elevation.size:
        peak_elevation = float(np.percentile(other_elevation, 90))
        contact_elevation = float(np.median(other_contact_elevation))
        drop = peak_elevation - contact_elevation
        if technique == "forehand_clear":
            status = (
                "observed"
                if peak_elevation >= 0.15 and drop >= 0.15
                else "needs_review"
            )
            message = (
                "Tay không thuận có nâng định hướng rồi hạ khi tăng tốc."
                if status == "observed" else
                "Chưa thấy rõ chuỗi nâng–hạ của tay không thuận."
            )
        else:
            other_wrist_contact = _median_point(
                keypoints, contact, (other_wrist,)
            )
            other_shoulder_contact = _median_point(
                keypoints, contact, (other_shoulder,)
            )
            distance = float(np.linalg.norm(
                other_wrist_contact - other_shoulder_contact
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

    knee_angle = _phase_knee_angle(keypoints, precontact)
    if math.isfinite(knee_angle):
        threshold = 165.0 if technique == "forehand_clear" else 160.0
        status = "observed" if knee_angle <= threshold else "needs_review"
        output.append(_item(
            "leg_loading", "Sử dụng chân khi chuẩn bị", knee_angle, status,
            "Có độ chùng gối quan sát được." if status == "observed"
            else "Hai gối khá thẳng; có thể tăng độ chùng khi chuẩn bị.",
            threshold={"maximum_degrees": threshold}, unit="degree",
        ))

    racket_wrist_preparation = _median_point(keypoints, preparation, (racket_wrist,))
    racket_shoulder_preparation = _median_point(
        keypoints, preparation, (racket_shoulder,)
    )
    if racket_wrist_preparation is not None and racket_shoulder_preparation is not None:
        preparation_height = float(
            (racket_shoulder_preparation[1] - racket_wrist_preparation[1]) / scale
        )
        minimum_height = -0.20 if technique == "forehand_clear" else -0.45
        status = "observed" if preparation_height >= minimum_height else "needs_review"
        output.append(_item(
            "racket_arm_preparation", "Vị trí tay vợt khi chuẩn bị",
            preparation_height, status,
            "Tay vợt được giữ ở vị trí sẵn sàng."
            if status == "observed" else "Tay vợt ở khá thấp khi bắt đầu động tác.",
            threshold={"minimum": minimum_height},
        ))

    elbow_values, reach_values = [], []
    for frame in range(len(keypoints)):
        racket_elbow = RIGHT_ELBOW if handedness == "right" else LEFT_ELBOW
        if np.min(keypoints[frame, [racket_shoulder, racket_elbow, racket_wrist], 2]) < 0.3:
            continue
        elbow_values.append(_angle(
            keypoints[frame, racket_shoulder, :2],
            keypoints[frame, racket_elbow, :2],
            keypoints[frame, racket_wrist, :2],
        ))
        reach_values.append(float(np.linalg.norm(
            keypoints[frame, racket_wrist, :2]
            - keypoints[frame, racket_shoulder, :2]
        ) / scale))
    finite_elbow = np.asarray([value for value in elbow_values if math.isfinite(value)])
    if finite_elbow.size >= 10:
        elbow_excursion = float(np.percentile(finite_elbow, 90) - np.percentile(finite_elbow, 10))
        status = "observed" if elbow_excursion >= 25.0 else "needs_review"
        output.append(_item(
            "arm_extension_excursion", "Biên độ gập–duỗi tay đánh",
            elbow_excursion, status,
            "Tay đánh có thay đổi gập–duỗi rõ trong toàn động tác."
            if status == "observed" else "Biên độ gập–duỗi tay đánh còn nhỏ.",
            threshold={"minimum_degrees": 25.0}, unit="degree",
        ))
    finite_reach = np.asarray([value for value in reach_values if math.isfinite(value)])
    if finite_reach.size >= 10:
        reach_excursion = float(np.percentile(finite_reach, 90) - np.percentile(finite_reach, 10))
        status = "observed" if reach_excursion >= 0.20 else "needs_review"
        output.append(_item(
            "arm_reach_excursion", "Biên độ vươn của tay đánh",
            reach_excursion, status,
            "Tay đánh có biên độ thu–vươn rõ."
            if status == "observed" else "Biên độ vươn tay quan sát được còn nhỏ.",
            threshold={"minimum": 0.20},
        ))

    hip_preparation = _median_point(keypoints, preparation, (LEFT_HIP, RIGHT_HIP))
    if hip_preparation is not None and hip_contact is not None:
        transfer = float(np.linalg.norm(hip_contact - hip_preparation) / scale)
        status = "observed" if transfer >= 0.08 else "needs_review"
        output.append(_item(
            "body_transfer", "Dịch chuyển thân từ chuẩn bị đến chạm cầu", transfer,
            "informational", "Có chuyển động thân tham gia vào cú đánh."
            if status == "observed" else "Chuyển động thân quan sát được còn ít.",
            threshold={"minimum": 0.08},
        ))

    wrist_follow = _median_point(keypoints, follow, (racket_wrist,))
    if wrist_contact is not None and wrist_follow is not None:
        follow_distance = float(np.linalg.norm(wrist_follow - wrist_contact) / scale)
        status = "observed" if follow_distance >= 0.25 else "needs_review"
        output.append(_item(
            "followthrough_completion", "Biên độ vung theo đà", follow_distance,
            status, "Tay vợt tiếp tục di chuyển rõ sau khi chạm cầu."
            if status == "observed" else "Follow-through quan sát được còn ngắn.",
            threshold={"minimum": 0.25},
        ))

    hip_follow = _median_point(keypoints, follow, (LEFT_HIP, RIGHT_HIP))
    left_ankle_follow = _median_point(keypoints, follow, (LEFT_ANKLE,))
    right_ankle_follow = _median_point(keypoints, follow, (RIGHT_ANKLE,))
    if hip_follow is not None and left_ankle_follow is not None and right_ankle_follow is not None:
        low = min(left_ankle_follow[0], right_ankle_follow[0]) - 0.15 * scale
        high = max(left_ankle_follow[0], right_ankle_follow[0]) + 0.15 * scale
        outside = max(low - hip_follow[0], hip_follow[0] - high, 0.0) / scale
        status = "observed" if outside <= 0 else "needs_review"
        output.append(_item(
            "recovery_balance", "Thăng bằng sau khi đánh", float(outside), status,
            "Trọng tâm thân nằm trong vùng hỗ trợ của hai chân."
            if status == "observed" else "Trọng tâm thân lệch khỏi vùng hỗ trợ của hai chân.",
            threshold={"maximum": 0.0},
        ))
        if output[-1]["status"] == "observed":
            # Hip-between-ankles is only a static observation; it cannot prove
            # dynamic recovery quality after a stroke.
            output[-1]["status"] = "informational"
    return output
