"""Dynamic hitter localization and spatial cropping using YOLO-Pose."""

from __future__ import annotations

import numpy as np


def select_hitter_box(
    result,
    player_side: str,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Select the active hitter bounding box from YOLO detection result based on court side.
    
    Args:
        result: Ultralytics YOLO Results object.
        player_side: "top" or "bottom".
        width: Frame width.
        height: Frame height.
        
    Returns:
        Bounding box array [x1, y1, x2, y2] or None.
    """
    if result.boxes is None or result.keypoints is None:
        return None
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()
    keypoints = result.keypoints.data.detach().cpu().numpy()

    candidates = []
    for box, class_id, pose in zip(boxes, classes, keypoints):
        if int(class_id) != 0:
            continue
        visible = pose[:, 2] >= 0.25
        center = (
            pose[visible, :2].mean(axis=0)
            if visible.any()
            else np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        )
        x, y = center
        min_x, max_x = (0.32, 0.68) if player_side == "top" else (0.22, 0.78)
        min_y = 0.25 if player_side == "top" else 0.42
        if not (min_x * width <= x <= max_x * width and min_y * height <= y <= 0.92 * height):
            continue
        if player_side == "top" and y >= 0.58 * height:
            continue
        candidates.append((float(y), box))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1] if player_side == "top" else candidates[-1][1]


def padded_square(
    box: np.ndarray,
    width: int,
    height: int,
    padding: float = 0.30,
) -> tuple[int, int, int, int]:
    """Expand a bounding box into a padded square centered on the player.
    
    Returns:
        (x_min, y_min, x_max, y_max)
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(max(x2 - x1, y2 - y1) * (1.0 + 2.0 * padding), min(width, height) * 0.22)
    return (
        max(0, int(round(cx - side / 2))),
        max(0, int(round(cy - side * 0.58))),
        min(width, int(round(cx + side / 2))),
        min(height, int(round(cy + side * 0.42))),
    )

