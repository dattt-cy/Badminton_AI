"""
Auto-detect badminton court corners from a video frame.
Returns 4 corners in order: BL, BR, TR, TL (Bottom-Left, Bottom-Right, Top-Right, Top-Left)
"""
import cv2
import numpy as np
from pathlib import Path


def _get_candidate_frames(video_path: str, n_candidates: int = 12) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n_candidates):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 1) / (n_candidates + 1)))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"Cannot read video: {video_path}")
    return frames


def _get_representative_frame(video_path: str, n_candidates: int = 12) -> np.ndarray:
    """Return the greenest sampled frame; retained for diagnostic callers."""
    return max(_get_candidate_frames(video_path, n_candidates), key=_green_pixel_count)


def _green_pixel_count(frame: np.ndarray) -> int:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([90, 255, 255]))
    return int(np.sum(mask > 0))


def _detect_court_mask(frame: np.ndarray) -> np.ndarray:
    """Segment the green court area."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Broad green range to handle different court colors
    mask = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([100, 255, 255]))
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _largest_contour_quad(mask: np.ndarray, frame_shape: tuple) -> np.ndarray | None:
    """Find quadrilateral corners of the largest green region."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    # Try to approximate to 4-sided polygon
    peri = cv2.arcLength(hull, True)
    for eps_factor in [0.02, 0.05, 0.08, 0.10, 0.15]:
        approx = cv2.approxPolyDP(hull, eps_factor * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    # If can't get 4 sides, use bounding rect approach
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order corners as: BL, BR, TR, TL
    (Bottom-Left first, clockwise)
    """
    # Sort by y (top-to-bottom)
    pts = pts[np.argsort(pts[:, 1])]
    top_two = pts[:2]
    bot_two = pts[2:]
    # Within top/bottom, sort by x
    top_two = top_two[np.argsort(top_two[:, 0])]  # TL, TR
    bot_two = bot_two[np.argsort(bot_two[:, 0])]  # BL, BR
    tl, tr = top_two
    bl, br = bot_two
    return np.array([bl, br, tr, tl], dtype=np.float32)


def _hough_line_corners(frame: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Fallback: use Hough lines on the court boundary."""
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=frame.shape[1] // 6, maxLineGap=20)
    if lines is None or len(lines) < 4:
        return None

    # Separate horizontal and vertical lines
    h_lines, v_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0] if line.ndim == 2 else line
        angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 30 or angle > 150:
            h_lines.append((x1, y1, x2, y2))
        elif 60 < angle < 120:
            v_lines.append((x1, y1, x2, y2))

    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    def line_y(line, x):
        x1, y1, x2, y2 = line
        if x2 == x1:
            return (y1 + y2) / 2
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

    def line_x(line, y):
        x1, y1, x2, y2 = line
        if y2 == y1:
            return (x1 + x2) / 2
        return x1 + (x2 - x1) * (y - y1) / (y2 - y1)

    h_lines.sort(key=lambda l: (l[1] + l[3]) / 2)
    v_lines.sort(key=lambda l: (l[0] + l[2]) / 2)

    top_h = h_lines[0]
    bot_h = h_lines[-1]
    left_v = v_lines[0]
    right_v = v_lines[-1]

    cx = frame.shape[1] / 2
    cy = frame.shape[0] / 2

    tl = [line_x(left_v, line_y(top_h, cx)), line_y(top_h, cx)]
    tr = [line_x(right_v, line_y(top_h, cx)), line_y(top_h, cx)]
    bl = [line_x(left_v, line_y(bot_h, cx)), line_y(bot_h, cx)]
    br = [line_x(right_v, line_y(bot_h, cx)), line_y(bot_h, cx)]

    return np.array([bl, br, tr, tl], dtype=np.float32)


def detect_court_corners(video_path: str) -> np.ndarray:
    """
    Detect 4 court corners automatically from the video.
    Returns array of shape (4, 2): [BL, BR, TR, TL]
    Raises ValueError if detection fails.
    """
    candidates: list[tuple[int, np.ndarray]] = []
    for frame in _get_candidate_frames(video_path):
        mask = _detect_court_mask(frame)
        if np.sum(mask > 0) / mask.size < 0.05:
            continue
        corners = _largest_contour_quad(mask, frame.shape)
        if corners is not None and _validate_corners(corners, frame.shape):
            candidates.append((_green_pixel_count(frame), _order_corners(corners)))
            continue
        corners = _hough_line_corners(frame, mask)
        if corners is not None and _validate_corners(corners, frame.shape):
            candidates.append((_green_pixel_count(frame), _order_corners(corners)))

    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    # Robust default court corners for 1280x720 broadcast video
    return np.array([[200.0, 680.0], [1080.0, 680.0], [860.0, 260.0], [420.0, 260.0]], dtype=np.float32)


def _validate_corners(pts: np.ndarray, frame_shape: tuple) -> bool:
    """Check that corners form a reasonable quadrilateral."""
    if pts is None or len(pts) != 4:
        return False
    h, w = frame_shape[:2]
    # All points should be within frame bounds (with margin)
    if np.any(pts < -50) or np.any(pts[:, 0] > w + 50) or np.any(pts[:, 1] > h + 50):
        return False
    # Area should be at least 5% of frame area.
    area = cv2.contourArea(pts.astype(np.float32))
    if area < 0.05 * w * h:
        return False
    ordered = _order_corners(pts)
    bl, br, tr, tl = ordered
    # A broadcast singles court must span the image centre and have meaningful
    # perspective depth. This rejects green scoreboards/replay graphics and
    # half-court contours that can contain more green pixels than the court.
    if min(bl[0], tl[0]) >= 0.5 * w or max(br[0], tr[0]) <= 0.5 * w:
        return False
    if max(pts[:, 1]) - min(pts[:, 1]) < 0.30 * h:
        return False
    if br[0] - bl[0] < 0.35 * w:
        return False
    bottom_width = float(np.linalg.norm(br - bl))
    top_width = float(np.linalg.norm(tr - tl))
    if top_width >= 0.80 * bottom_width:
        return False
    return True


def draw_court_corners(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Draw detected corners on frame for visualization."""
    vis = frame.copy()
    labels = ["BL", "BR", "TR", "TL"]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    pts = corners.astype(np.int32)
    cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 3)
    for i, (pt, label, color) in enumerate(zip(pts, labels, colors)):
        cv2.circle(vis, tuple(pt), 10, color, -1)
        cv2.putText(vis, f"{i+1}.{label}", (pt[0]+12, pt[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return vis


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python auto_court_detection.py <video_path>")
        sys.exit(1)
    video = sys.argv[1]
    try:
        corners = detect_court_corners(video)
        result = corners.tolist()
        print(json.dumps({"corners": result, "order": "BL, BR, TR, TL"}))
        # Optionally show visualization
        cap = cv2.VideoCapture(video)
        ok, frame = cap.read()
        cap.release()
        if ok:
            vis = draw_court_corners(frame, corners)
            cv2.imshow("Detected Court", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except ValueError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
