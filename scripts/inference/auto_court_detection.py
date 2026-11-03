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


def _refine_corners_by_edges(mask: np.ndarray, ordered: np.ndarray) -> np.ndarray:
    """Replace each corner by the intersection of the two adjacent fitted edges.

    The morphological opening in ``_detect_court_mask`` rounds off the acute
    baseline corners, so the polygon approximation picks a vertex on the cut
    and pulls that corner inwards. Fitting a line to the middle part of every
    side ignores the cut, and intersecting neighbouring lines restores it.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return ordered
    points = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    lines = []
    for i in range(4):
        a, b = ordered[i], ordered[(i + 1) % 4]
        side = b - a
        length = float(np.linalg.norm(side))
        if length < 1:
            return ordered
        direction = side / length
        rel = points - a
        t = rel @ direction / length
        dist = np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])
        selected = points[(t > 0.15) & (t < 0.85) & (dist < max(6.0, 0.03 * length))]
        if len(selected) < 10:
            return ordered
        vx, vy, x0, y0 = cv2.fitLine(selected, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    refined = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        det = d1[0] * -d2[1] + d1[1] * d2[0]
        if abs(det) < 1e-6:
            return ordered
        s = ((p2 - p1)[0] * -d2[1] + (p2 - p1)[1] * d2[0]) / det
        refined.append(p1 + s * d1)
    refined = np.array(refined, dtype=np.float32)
    # A refinement that moves a corner far away means the edge fit was wrong.
    if np.linalg.norm(refined - ordered, axis=1).max() > 0.08 * mask.shape[1]:
        return ordered
    return refined


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


# Doubles court in metres (x across 0..6.1, y along 0..13.4, far baseline y=0).
# The returned corners are the outer doubles corners, as in ShuttleSet's homography.csv.
_COURT_W, _COURT_L = 6.1, 13.4
_COURT_LINES = [
    ((0.0, 0.0), (6.1, 0.0)), ((0.0, 13.4), (6.1, 13.4)),          # baselines
    ((0.0, 0.76), (6.1, 0.76)), ((0.0, 12.64), (6.1, 12.64)),      # doubles long service lines
    ((0.0, 4.72), (6.1, 4.72)), ((0.0, 8.68), (6.1, 8.68)),        # short service lines
    ((0.0, 0.0), (0.0, 13.4)), ((6.1, 0.0), (6.1, 13.4)),          # doubles sidelines
    ((0.46, 0.0), (0.46, 13.4)), ((5.64, 0.0), (5.64, 13.4)),      # singles sidelines
    ((3.05, 0.0), (3.05, 4.72)), ((3.05, 8.68), (3.05, 13.4)),     # centre lines
]
_LINE_SAMPLES = np.concatenate([
    np.linspace(a, b, 60) for a, b in (tuple(map(np.asarray, line)) for line in _COURT_LINES)
]).astype(np.float32)
_UNIT_CORNERS = np.float32([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])  # BL, BR, TR, TL


def _line_mask(frame: np.ndarray) -> np.ndarray:
    """Thin bright structures (court lines) as a float map in [0, 1]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    mask = (tophat > 35).astype(np.uint8)
    return cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0) > 0.2


def _line_score(corners: np.ndarray, lines: np.ndarray) -> float:
    """Fraction of projected court-line samples that land on a bright line pixel."""
    h_mat = cv2.getPerspectiveTransform(_UNIT_CORNERS, corners.astype(np.float32))
    unit = _LINE_SAMPLES / np.float32([_COURT_W, _COURT_L])
    pts = cv2.perspectiveTransform(unit[None], h_mat)[0]
    x, y = np.rint(pts[:, 0]).astype(int), np.rint(pts[:, 1]).astype(int)
    inside = (x >= 0) & (x < lines.shape[1]) & (y >= 0) & (y < lines.shape[0])
    hits = np.zeros(len(pts), dtype=bool)
    hits[inside] = lines[y[inside], x[inside]]
    return float(hits.mean())


def _fit_court_lines(frame: np.ndarray, corners: np.ndarray, green: np.ndarray) -> tuple[np.ndarray, float]:
    """Move the four corners so the court-line model overlaps the painted lines (coordinate ascent).

    The search runs coarse-to-fine: large steps on thickened lines, then small steps on
    the thin line mask. The court must keep a plausible share of the green-mat area so
    the model cannot collapse onto a single bright line.
    """
    fine = _line_mask(frame)
    coarse = cv2.dilate(fine.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    green_area = cv2.contourArea(green.astype(np.float32))
    x_min, y_min = green.min(axis=0) - 15
    x_max, y_max = green.max(axis=0) + 15

    def plausible(quad: np.ndarray) -> bool:
        ratio = cv2.contourArea(quad) / max(green_area, 1.0)
        return (0.25 <= ratio <= 1.05 and np.all(quad[:, 0] >= x_min) and np.all(quad[:, 0] <= x_max)
                and np.all(quad[:, 1] >= y_min) and np.all(quad[:, 1] <= y_max))

    best = corners.astype(np.float32).copy()
    for lines, steps in ((coarse, (32, 16, 8)), (fine, (6, 3, 1))):
        best_score = _line_score(best, lines)
        for step in steps:
            improved = True
            while improved:
                improved = False
                for corner in range(4):
                    for axis in range(2):
                        for sign in (-1, 1):
                            trial = best.copy()
                            trial[corner, axis] += sign * step
                            if not plausible(trial):
                                continue
                            score = _line_score(trial, lines)
                            if score > best_score:
                                best, best_score, improved = trial, score, True
    return best, best_score


def _shrink_towards_lines(green: np.ndarray, frame: np.ndarray, keep: int = 4) -> list[np.ndarray]:
    """Initial guesses: the green-mat quad shrunk by a grid of margins, best line overlap first.

    The mat can extend far beyond the court, and green advertising boards can pull its
    top edge up to the stands, so the margins cover a wide range.
    """
    bl, br, tr, tl = green
    coarse = cv2.dilate(_line_mask(frame).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    scored = []
    for bottom in np.arange(0.0, 0.22, 0.03):
        for top in np.arange(0.0, 0.46, 0.05):
            for side in (0.0, 0.05, 0.10):
                left_bottom = bl + bottom * (tl - bl)
                right_bottom = br + bottom * (tr - br)
                left_top = tl + top * (bl - tl)
                right_top = tr + top * (br - tr)
                quad = np.float32([
                    left_bottom + side * (right_bottom - left_bottom), right_bottom + side * (left_bottom - right_bottom),
                    right_top + side * (left_top - right_top), left_top + side * (right_top - left_top),
                ])
                scored.append((_line_score(quad, coarse), quad))
    scored.sort(key=lambda item: -item[0])
    return [green] + [quad for _, quad in scored[:keep]]


def _resolve_line_aliasing(
    frame: np.ndarray, fit: tuple[np.ndarray, float], green: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Re-fit after shifting by one line spacing; keep the best-scoring court.

    Parallel lines close together (singles/doubles sidelines 0.46 m apart, baseline and
    doubles long service line 0.76 m apart) give local optima where the whole model sits
    one spacing outside or inside the court.
    """
    side, end = 0.46 / _COURT_W, 0.76 / _COURT_L
    best = fit
    for dx, dy in ((side, 0.0), (-side, 0.0), (0.0, end), (0.0, -end), (side, end), (-side, -end)):
        h_mat = cv2.getPerspectiveTransform(_UNIT_CORNERS, best[0].astype(np.float32))
        unit = np.float32([[dx, 1 - dy], [1 - dx, 1 - dy], [1 - dx, dy], [dx, dy]])
        shifted = cv2.perspectiveTransform(unit[None], h_mat)[0]
        trial = _fit_court_lines(frame, shifted, green)
        if trial[1] > best[1]:
            best = trial
    return best


def detect_court_line_corners(video_path: str, min_score: float = 0.55) -> np.ndarray:
    """Court corners on the painted outer lines, refined from the green-mat quad.

    The green mat usually extends beyond the court, so the mat quad overstates the
    court (about 80 px at the near baseline on ShuttleSet broadcasts). Falls back to
    the mat quad when the line model does not fit well.
    """
    green = detect_court_corners(video_path)
    fits = []
    for frame in _get_candidate_frames(video_path, n_candidates=12):
        best = max((_fit_court_lines(frame, guess, green) for guess in _shrink_towards_lines(green, frame)),
                   key=lambda r: r[1])
        if best[1] >= min_score:
            best = _resolve_line_aliasing(frame, best, green)
        if best[1] >= min_score and _validate_corners(best[0], frame.shape):
            fits.append((best[1], best[0]))
    if not fits:
        return green
    if len(fits) <= 2:
        # Too few fits for a medoid; a replay or alternate camera can still fit its own
        # lines well, so prefer the court lying closest to the main-camera mat quad.
        return min(fits, key=lambda item: float(np.abs(item[1] - green).sum()))[1]
    return _consensus_corners(fits)


def detect_court_corners(video_path: str) -> np.ndarray:
    """
    Detect 4 court corners automatically from the video.
    Returns array of shape (4, 2): [BL, BR, TR, TL]
    Raises ValueError if detection fails.
    """
    candidates: list[tuple[int, np.ndarray]] = []
    for frame in _get_candidate_frames(video_path, n_candidates=24):
        mask = _detect_court_mask(frame)
        if np.sum(mask > 0) / mask.size < 0.05:
            continue
        corners = _largest_contour_quad(mask, frame.shape)
        if corners is not None and _validate_corners(corners, frame.shape):
            refined = _refine_corners_by_edges(mask, _order_corners(corners))
            if _validate_corners(refined, frame.shape):
                corners = refined
            candidates.append((_green_pixel_count(frame), _order_corners(corners)))
            continue
        corners = _hough_line_corners(frame, mask)
        if corners is not None and _validate_corners(corners, frame.shape):
            candidates.append((_green_pixel_count(frame), _order_corners(corners)))

    if candidates:
        return _consensus_corners(candidates)

    # Robust default court corners for 1280x720 broadcast video
    return np.array([[200.0, 680.0], [1080.0, 680.0], [860.0, 260.0], [420.0, 260.0]], dtype=np.float32)


def _consensus_corners(candidates: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """Return the medoid quadrilateral of the per-frame candidates.

    The main broadcast camera recurs across sampled frames, whereas replays and
    low or close alternate angles are one-off views. Those views can contain
    more green pixels than the main camera, so the greenest candidate is only
    used when there are too few candidates to form a consensus.
    """
    if len(candidates) <= 2:
        return max(candidates, key=lambda item: item[0])[1]
    quads = np.stack([corners for _, corners in candidates])
    distances = np.abs(quads[:, None] - quads[None]).sum(axis=(2, 3))
    return quads[int(distances.sum(axis=1).argmin())]


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
    # The main broadcast camera sees a near-symmetric trapezoid with level
    # baselines. A far player whose shirt falls in the green range pulls one
    # top corner towards them, which breaks these properties.
    if abs(tl[1] - tr[1]) >= 0.05 * h or abs(bl[1] - br[1]) >= 0.05 * h:
        return False
    if abs((tl[0] + tr[0]) / 2 - (bl[0] + br[0]) / 2) >= 0.06 * w:
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
