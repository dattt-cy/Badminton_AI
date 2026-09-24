"""Render Professional AI Broadcast Visual Demo for CR-Ensemble.

Overlays:
  - Multi-person Pose Skeleton (YOLO/COCO format 17 keypoints with colored bones)
  - Shuttlecock Trajectory Trail with glowing head
  - Court Boundary Polygon
  - Broadcast HUD Banner with predicted stroke, confidence bar, and hit alert
"""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_5MIN = REPO_ROOT / "work_dirs" / "bfmd_indonesia2025_5min.mp4"
RESULTS_JSON = REPO_ROOT / "work_dirs" / "bfmd_full_pipeline_cr_results.json"
ANNOT_ROOT = Path(r"C:\Users\ADMIN\Downloads\BFMD_data-20260921T175743Z-1-001\BFMD_data\annotations")
VIDEO_PREFIX = "KAPAL-API-Indonesia-Open-2025-Anders-Antonsen-DEN-3-vs.-Chou-Tien-Chen-TPE-6-F"
OUTPUT_VIDEO = REPO_ROOT / "work_dirs" / "bfmd_cr_ensemble_broadcast_demo.mp4"

CLIP_OFFSET = 18_000

# COCO 17 Keypoints connections
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),               # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # Legs
]

COLOR_PALETTE = {
    "net_shot": (0, 215, 255),    # Gold / Amber
    "lift": (255, 180, 0),        # Sky Blue
    "smash": (0, 0, 255),         # Bright Red
    "drop": (255, 100, 255),      # Magenta
    "clear": (255, 255, 0),       # Cyan
    "drive": (50, 205, 50),       # Lime Green
    "serve": (0, 255, 128),       # Spring Green
    "net_attack": (0, 69, 255),   # Orange-Red
}


def load_data():
    with open(ANNOT_ROOT / "pose" / f"{VIDEO_PREFIX}.json") as f:
        pose_raw = json.load(f)
    pose_dict = {}
    for vid_data in pose_raw["videos"].values():
        for fid_str, persons in vid_data["frames"].items():
            pose_dict[int(fid_str)] = persons

    with open(ANNOT_ROOT / "shuttle" / f"{VIDEO_PREFIX}.json") as f:
        shuttle_raw = json.load(f)
    shuttle_dict = {}
    for item in shuttle_raw["predictions"][0]["result"][0]["value"]["sequence"]:
        if item.get("enabled", True):
            # Percentage to pixel (1280x720)
            shuttle_dict[int(item["frame"])] = (
                int(item["x"] * 12.80), int(item["y"] * 7.20)
            )

    with open(RESULTS_JSON) as f:
        res_data = json.load(f)["results"]
    events_by_frame = {}
    for r in res_data:
        events_by_frame[r["rel_frame"]] = r

    return pose_dict, shuttle_dict, events_by_frame


def draw_skeleton(frame: np.ndarray, persons: list):
    for p in persons:
        kps = p.get("keypoints", [])
        if len(kps) < 17:
            continue
        pts = [(int(kp[0] * 1280), int(kp[1] * 720)) for kp in kps]

        # Draw bones
        for u, v in SKELETON_EDGES:
            pt1, pt2 = pts[u], pts[v]
            if pt1[0] > 0 and pt2[0] > 0:
                cv2.line(frame, pt1, pt2, (0, 255, 200), 2, cv2.LINE_AA)

        # Draw joints
        for pt in pts:
            if pt[0] > 0:
                cv2.circle(frame, pt, 3, (0, 165, 255), -1, cv2.LINE_AA)


def draw_shuttle_trail(frame: np.ndarray, shuttle_dict: dict, current_abs: int, trail_len: int = 12):
    pts = []
    for d in range(trail_len, -1, -1):
        f = current_abs - d
        if f in shuttle_dict:
            pts.append(shuttle_dict[f])

    if len(pts) >= 2:
        for i in range(len(pts) - 1):
            alpha = (i + 1) / len(pts)
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, pts[i], pts[i + 1], (0, 255, 255), thickness, cv2.LINE_AA)

    if pts:
        # Glow head
        cv2.circle(frame, pts[-1], 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, pts[-1], 8, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, active_event: dict | None, event_age: int, max_age: int = 25):
    h, w = frame.shape[:2]

    # 1. Top Brand Header
    header_bar = frame[15:55, 30:370].copy()
    overlay = frame.copy()
    cv2.rectangle(overlay, (30, 15), (370, 55), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (30, 15), (370, 55), (0, 215, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "AI BADMINTON ANALYZER", (42, 35), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "CR-Gated Fusion | PBL6", (42, 49), cv2.FONT_HERSHEY_DUPLEX, 0.38, (0, 215, 255), 1, cv2.LINE_AA)

    # 2. Match Info Bottom-Left
    cv2.rectangle(frame, (30, h - 50), (450, h - 20), (15, 15, 20), -1)
    cv2.putText(frame, "INDONESIA OPEN 2025 | FINALS (ANTONSEN vs CHOU)", (40, h - 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    # 3. Action Card HUD (Top-Right)
    if active_event is not None and event_age < max_age:
        card_w, card_h = 320, 95
        card_x = w - card_w - 30
        card_y = 15

        pred = active_event.get("pred", "unknown").upper().replace("_", " ")
        conf = active_event.get("conf", 0.0)
        gt = (active_event.get("gt_label") or "").upper().replace("_", " ")
        pred_key = active_event.get("pred", "net_shot")
        accent_color = COLOR_PALETTE.get(pred_key, (0, 255, 255))

        # Background card with alpha
        card_bg = frame.copy()
        cv2.rectangle(card_bg, (card_x, card_y), (card_x + card_w, card_y + card_h), (18, 18, 22), -1)
        fade_alpha = max(0.2, 1.0 - (event_age / max_age) ** 2)
        cv2.addWeighted(card_bg, 0.88 * fade_alpha, frame, 1.0 - 0.88 * fade_alpha, 0, frame)

        # Border
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), accent_color, 2, cv2.LINE_AA)

        # Title: STROKE CLASSIFIED
        cv2.putText(frame, "HIT DETECTED - STROKE TYPE", (card_x + 15, card_y + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

        # Action Name
        cv2.putText(frame, pred, (card_x + 15, card_y + 56),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

        # Confidence Bar
        bar_x = card_x + 15
        bar_y = card_y + 70
        bar_w = int(210 * conf)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 210, bar_y + 10), (50, 50, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), accent_color, -1)
        cv2.putText(frame, f"{conf:.0%}", (card_x + 235, bar_y + 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, accent_color, 1, cv2.LINE_AA)


def main():
    print("=" * 72)
    print("RENDERING BROADCAST AI DEMO VIDEO")
    print(f"Source: {VIDEO_5MIN}")
    print(f"Output: {OUTPUT_VIDEO}")
    print("=" * 72)

    pose_dict, shuttle_dict, events_by_frame = load_data()

    cap = cv2.VideoCapture(str(VIDEO_5MIN))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # We select 2 highlight rally clips:
    # Segment 1: Rally 1 (Giao cầu + Clear + Net Attack) frames 1160 to 1360 (~6.6s)
    # Segment 2: Rally 8 (Đôi công lưới đỉnh cao: Net Shot + Lift + Drop) frames 8200 to 8680 (~16.0s)
    segments = [
        (1160, 1360, "Rally 1 - Opening Exchange"),
        (8200, 8680, "Rally 8 - Intense 20-Shot Net Battle"),
    ]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (w, h))

    total_rendered = 0
    for seg_idx, (s_start, s_end, seg_name) in enumerate(segments):
        print(f"\n[*] Rendering {seg_name} (Frames {s_start} -> {s_end})...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, s_start)

        active_event = None
        event_age = 999

        for f_idx in range(s_start, s_end + 1):
            ok, frame = cap.read()
            if not ok:
                break

            abs_f = f_idx + CLIP_OFFSET

            # Check if hit event happened at this frame or nearby
            for check_f in range(f_idx - 2, f_idx + 1):
                if check_f in events_by_frame:
                    active_event = events_by_frame[check_f]
                    event_age = 0
                    break

            # 1. Draw Skeleton
            # Find nearest pose
            persons = None
            for d in range(4):
                for sgn in [0, 1, -1]:
                    cand = abs_f + sgn * d
                    if cand in pose_dict:
                        persons = pose_dict[cand]
                        break
                if persons:
                    break
            if persons:
                draw_skeleton(frame, persons)

            # 2. Draw Shuttlecock trail
            draw_shuttle_trail(frame, shuttle_dict, abs_f, trail_len=14)

            # 3. Draw Broadcast HUD
            draw_hud(frame, active_event, event_age)
            event_age += 1

            out.write(frame)
            total_rendered += 1

    cap.release()
    out.release()

    size_mb = OUTPUT_VIDEO.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 72)
    print(f"DEMO VIDEO RENDERED SUCCESSFULLY!")
    print(f"Path: {OUTPUT_VIDEO}")
    print(f"Size: {size_mb:.2f} MB | Frames: {total_rendered} ({total_rendered/fps:.1f} seconds)")
    print("=" * 72)


if __name__ == "__main__":
    main()

