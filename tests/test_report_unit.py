"""Unit tests UT-01..UT-08 from the PBL6 Test Report (sheet UnitTesting).

Function A (UT-01..UT-04): court corner detection on synthetic broadcast frames
with exact ground truth, plus a sanity check on a real broadcast clip.
Function B (UT-05..UT-08): hit spotting with the production BadmintonHitNet
checkpoint against ShuttleSet labels, and NMS / rescue post-processing.
"""

import csv
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.inference.auto_court_detection as court
from ai_classifier.localization import (
    HitEvent,
    rescue_alternating_bridges,
    rescue_pre_rally_serves,
    two_stage_temporal_nms,
)
from ai_classifier.localization.hit_spotting import temporal_nms

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAME_H, FRAME_W = 720, 1280
# BL, BR, TR, TL of the synthetic broadcast court.
GT_CORNERS = np.array([[250, 650], [1030, 650], [830, 250], [450, 250]], dtype=np.float32)
FALLBACK_CORNERS = np.array(
    [[200.0, 680.0], [1080.0, 680.0], [860.0, 260.0], [420.0, 260.0]], dtype=np.float32
)
BACKGROUND_BGR = (120, 45, 45)
COURT_BGR = (60, 140, 40)

HIT_CHECKPOINT = REPO_ROOT / "src" / "ai_classifier" / "models" / "checkpoints" / "hit_detector.pth"
CLIP_VIDEO = REPO_ROOT / "work_dirs" / "eval_clips" / "clip1_match44_10min.mp4"
MANIFEST = REPO_ROOT / "data" / "manifests" / "shuttleset_rgb.csv"
CLIP_MATCH_ID, CLIP_RAW_START = 44, 45000
# The whole 10-minute clip; UT_SCAN_FRAMES scans only its start (e.g. 1800 = 1 minute) for a quick demo.
SCAN_FRAMES = int(os.environ.get("UT_SCAN_FRAMES", 18000))
TOLERANCE = 12


def court_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.full((FRAME_H, FRAME_W, 3), BACKGROUND_BGR, dtype=np.uint8)
    cv2.fillPoly(frame, [GT_CORNERS.astype(np.int32)], COURT_BGR)
    cv2.polylines(frame, [GT_CORNERS.astype(np.int32)], True, (235, 235, 235), 3)
    cv2.line(frame, (360, 450), (920, 450), (235, 235, 235), 3)  # net line
    noise = rng.integers(-6, 7, frame.shape, dtype=np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def crowd_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    palette = np.array([[40, 40, 160], [150, 60, 60], [90, 90, 90], [30, 120, 200], [200, 200, 200]])
    blocks = palette[rng.integers(0, len(palette), (FRAME_H // 40, FRAME_W // 40))]
    return np.kron(blocks, np.ones((40, 40, 1))).astype(np.uint8)


def billboard_frame() -> np.ndarray:
    frame = crowd_frame(seed=7)
    frame[20:110, :] = COURT_BGR  # wide green advertising board, ~12% of the frame
    return frame


def write_video(path: Path, frames: list[np.ndarray]) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (FRAME_W, FRAME_H))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def max_corner_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=1).max())


def assert_broadcast_order(corners: np.ndarray) -> None:
    bl, br, tr, tl = corners
    assert bl[1] > tl[1] and br[1] > tr[1]
    assert bl[0] < br[0] and tl[0] < tr[0]


# --- Function A: detect_court_corners ---------------------------------------


def test_ut01_detects_four_corners_on_standard_video(tmp_path: Path) -> None:
    video = write_video(tmp_path / "court_std.mp4", [court_frame(i) for i in range(60)])

    corners = court.detect_court_corners(str(video))

    assert corners.shape == (4, 2)
    assert_broadcast_order(corners)
    assert max_corner_error(corners, GT_CORNERS) <= 15
    assert np.all((corners[:, 0] >= 0) & (corners[:, 0] < FRAME_W))
    assert np.all((corners[:, 1] >= 0) & (corners[:, 1] < FRAME_H))
    assert cv2.contourArea(corners) >= 0.05 * FRAME_W * FRAME_H


@pytest.mark.skipif(not CLIP_VIDEO.exists(), reason="broadcast eval clip not available")
def test_ut01_real_broadcast_clip_is_detected_without_fallback() -> None:
    corners = court.detect_court_corners(str(CLIP_VIDEO))

    assert corners.shape == (4, 2)
    assert court._validate_corners(corners, (FRAME_H, FRAME_W))
    assert not np.allclose(corners, FALLBACK_CORNERS)


def test_ut02_skips_crowd_frames_before_the_court(monkeypatch) -> None:
    court_only = [court_frame(i) for i in range(12)]
    mixed = [crowd_frame(i) for i in range(12)] + court_only
    monkeypatch.setattr(court, "_get_candidate_frames", lambda *_a, **_k: court_only)
    reference = court.detect_court_corners("court_only.mp4")

    quad_calls = []
    original_quad = court._largest_contour_quad

    def counting_quad(mask, shape):
        quad_calls.append(1)
        return original_quad(mask, shape)

    monkeypatch.setattr(court, "_get_candidate_frames", lambda *_a, **_k: mixed)
    monkeypatch.setattr(court, "_largest_contour_quad", counting_quad)
    corners = court.detect_court_corners("with_intro.mp4")

    assert len(quad_calls) == 12  # crowd frames rejected by the 5% green gate
    assert_broadcast_order(corners)
    assert max_corner_error(corners, reference) <= 15


def _lighting_variant(frame: np.ndarray, name: str) -> np.ndarray:
    f = frame.astype(np.float32)
    if name == "darker_30":
        f *= 0.7
    elif name == "brighter_30":
        f *= 1.3
    elif name == "warm":
        f += np.array([-20, 30, 40])
    elif name == "cool":
        f += np.array([40, 10, -15])
    return np.clip(f, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("variant", ["original", "darker_30", "brighter_30", "warm", "cool"])
def test_ut03_detects_court_under_lighting_changes(monkeypatch, variant: str) -> None:
    base = [court_frame(i) for i in range(24)]
    monkeypatch.setattr(court, "_get_candidate_frames", lambda *_a, **_k: base)
    reference = court.detect_court_corners("original.mp4")

    frames = [_lighting_variant(f, variant) for f in base]
    monkeypatch.setattr(court, "_get_candidate_frames", lambda *_a, **_k: frames)
    corners = court.detect_court_corners(f"{variant}.mp4")

    assert corners.shape == (4, 2)
    assert_broadcast_order(corners)
    assert max_corner_error(corners, reference) <= 20
    assert cv2.contourArea(corners) >= 0.05 * FRAME_W * FRAME_H


@pytest.mark.parametrize("kind", ["crowd", "green_billboard"])
def test_ut04_returns_default_corners_without_court(tmp_path: Path, kind: str) -> None:
    frames = [crowd_frame(i) for i in range(60)] if kind == "crowd" else [billboard_frame()] * 60
    video = write_video(tmp_path / f"{kind}.mp4", frames)

    corners = court.detect_court_corners(str(video))

    np.testing.assert_allclose(corners, FALLBACK_CORNERS)
    assert_broadcast_order(corners)


# --- Function B: hit spotting -----------------------------------------------


def _load_clip_ground_truth() -> list[dict]:
    end = CLIP_RAW_START + SCAN_FRAMES
    gt = []
    with open(MANIFEST, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["match_id"] != str(CLIP_MATCH_ID):
                continue
            frame = int(float(row["hit_frame"]))
            if CLIP_RAW_START <= frame < end:
                side = {"top": "upper", "bottom": "lower"}.get(row["player_side"], "unknown")
                gt.append({"frame": frame - CLIP_RAW_START, "side": side})
    return sorted(gt, key=lambda item: item["frame"])


def _match_one_to_one(preds: list[HitEvent], gt: list[dict]) -> list[tuple[HitEvent, dict]]:
    pairs = sorted(
        (abs(p.frame - g["frame"]), i, j)
        for i, p in enumerate(preds)
        for j, g in enumerate(gt)
        if abs(p.frame - g["frame"]) <= TOLERANCE
    )
    used_p, used_g, matched = set(), set(), []
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            matched.append((preds[i], gt[j]))
    return matched


@pytest.fixture(scope="module")
def scanned_clip():
    if not (HIT_CHECKPOINT.exists() and CLIP_VIDEO.exists() and MANIFEST.exists()):
        pytest.skip("checkpoint, eval clip or ShuttleSet manifest not available")
    torch = pytest.importorskip("torch")
    from scripts.inference.scan_video_hit_rgb import load_model, scan_video

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(HIT_CHECKPOINT)
    model.to(device)
    raw = scan_video(
        CLIP_VIDEO, model, int(meta.get("frames", 16)),
        stride=4, batch_size=32, device=device,
        use_amp=device.type == "cuda", max_frames=SCAN_FRAMES,
    )
    events = two_stage_temporal_nms(raw, threshold=0.75, radius=8, same_side_radius=15)
    gt = _load_clip_ground_truth()
    return events, gt, _match_one_to_one(events, gt)


def test_ut05_detects_hit_moments_on_match_video(scanned_clip) -> None:
    events, gt, matched = scanned_clip
    precision = len(matched) / len(events)
    recall = len(matched) / len(gt)
    print(f"\nUT-05 gt={len(gt)} pred={len(events)} matched={len(matched)} "
          f"precision={precision:.3f} recall={recall:.3f}")

    assert all(isinstance(e.frame, int) and e.side in {"upper", "lower"} for e in events)
    assert all(0 <= e.frame < SCAN_FRAMES and 0.75 <= e.score <= 1.0 + 1e-6 for e in events)
    assert len({e.frame for e in events}) == len(events)
    assert all(abs(p.frame - g["frame"]) <= TOLERANCE for p, g in matched)
    assert precision >= 0.85
    assert recall >= 0.85


def test_ut06_assigns_hitter_side(scanned_clip) -> None:
    events, _gt, matched = scanned_clip
    labelled = [(p, g) for p, g in matched if g["side"] != "unknown"]
    accuracy = sum(p.side == g["side"] for p, g in labelled) / len(labelled)
    print(f"\nUT-06 labelled={len(labelled)} side_accuracy={accuracy:.3f}")

    assert all(e.side in {"upper", "lower"} for e in events)
    assert accuracy >= 0.90


def test_ut07_two_stage_nms_removes_duplicates() -> None:
    raw = [
        HitEvent(100, "upper", 0.80), HitEvent(102, "upper", 0.95),
        HitEvent(104, "upper", 0.85), HitEvent(106, "upper", 0.78),
        HitEvent(113, "upper", 0.82), HitEvent(125, "lower", 0.90),
        HitEvent(200, "lower", 0.88), HitEvent(203, "lower", 0.91),
        HitEvent(207, "lower", 0.76),
    ]

    stage1 = temporal_nms(raw, threshold=0.75, radius=8)
    final = two_stage_temporal_nms(raw, threshold=0.75, radius=8, same_side_radius=15)

    assert [e.frame for e in stage1] == [102, 113, 125, 203]
    assert [e.frame for e in final] == [102, 125, 203]
    assert {e.frame: e.score for e in final}[102] == 0.95
    assert {e.frame: e.score for e in final}[203] == 0.91


def test_ut08_rescues_serve_and_missed_alternating_hit() -> None:
    serve_base = [HitEvent(120, "lower", 0.88), HitEvent(340, "lower", 0.90)]
    serve_raw = serve_base + [HitEvent(300, "upper", 0.55)]
    bridge_base = [HitEvent(500, "upper", 0.92), HitEvent(560, "upper", 0.89)]
    bridge_raw = bridge_base + [
        HitEvent(505, "lower", 0.70),
        HitEvent(530, "lower", 0.60),
        HitEvent(545, "upper", 0.70),
    ]

    out_serve = rescue_pre_rally_serves(serve_base, serve_raw)
    out_bridge = rescue_alternating_bridges(bridge_base, bridge_raw)

    assert [(e.frame, e.side) for e in out_serve] == [(120, "lower"), (300, "upper"), (340, "lower")]
    assert [(e.frame, e.side) for e in out_bridge] == [(500, "upper"), (530, "lower"), (560, "upper")]
    rescued = [e for e in out_serve + out_bridge if e.frame in (300, 530)]
    assert all(0.40 <= e.score < 0.75 for e in rescued)
