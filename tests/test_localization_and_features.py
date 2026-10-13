"""Consolidated tests for court detection, hit spotting NMS, contact refinement, and cache."""

import numpy as np
import pytest

from ai_classifier.features import modality_quality_vector
from ai_classifier.localization import (
    HitEvent,
    rescue_alternating_bridges,
    rescue_pre_rally_serves,
    two_stage_temporal_nms,
)
from ai_classifier.localization.contact_refinement import refine_contact_index
from scripts.evaluation.tune_hit_postprocessing import Case, aggregate
from scripts.inference.auto_court_detection import _validate_corners
from scripts.inference.robust_aligned_video_pipeline import AlignedVideoPipeline

FRAME_SHAPE = (720, 1280, 3)


def test_accepts_full_centered_broadcast_court() -> None:
    corners = np.float32([[222, 694], [1061, 709], [873, 262], [412, 260]])
    assert _validate_corners(corners, FRAME_SHAPE)


def test_rejects_one_sided_or_shallow_green_regions() -> None:
    right_half = np.float32([[858, 719], [1279, 719], [1279, 364], [794, 364]])
    shallow = np.float32([[0, 719], [877, 719], [877, 584], [0, 584]])
    non_perspective = np.float32([[0, 719], [1279, 719], [1177, 360], [0, 388]])
    assert not _validate_corners(right_half, FRAME_SHAPE)
    assert not _validate_corners(shallow, FRAME_SHAPE)
    assert not _validate_corners(non_perspective, FRAME_SHAPE)


def test_refines_to_plausible_nearby_wrist_contact() -> None:
    shuttle = np.full((9, 2), [0.8, 0.8], dtype=np.float32)
    wrists = np.full((9, 2), [0.1, 0.1], dtype=np.float32)
    shuttle[6] = [0.42, 0.40]
    wrists[6] = [0.40, 0.40]
    assert refine_contact_index(shuttle, wrists, center=4, radius=3) == 6


def test_keeps_center_when_tracks_are_missing_or_implausible() -> None:
    missing = np.zeros((7, 2), dtype=np.float32)
    assert refine_contact_index(missing, missing, center=3, radius=3) == 3

    shuttle = np.full((7, 2), [0.9, 0.9], dtype=np.float32)
    wrists = np.full((7, 2), [0.1, 0.1], dtype=np.float32)
    assert refine_contact_index(shuttle, wrists, center=3, radius=3) == 3


def test_zero_radius_is_disabled_and_invalid_center_is_rejected() -> None:
    track = np.ones((5, 2), dtype=np.float32)
    assert refine_contact_index(track, track, center=2, radius=0) == 2
    with pytest.raises(ValueError):
        refine_contact_index(track, track, center=5, radius=2)


def test_second_stage_suppresses_same_side_but_keeps_fast_exchange() -> None:
    events = [
        HitEvent(100, "upper", 0.90),
        HitEvent(116, "lower", 0.85),
        HitEvent(132, "lower", 0.70),
    ]
    kept = two_stage_temporal_nms(
        events, threshold=0.55, radius=15, same_side_radius=18
    )
    assert [(event.frame, event.side) for event in kept] == [
        (100, "upper"), (116, "lower")
    ]


def test_aggregate_uses_same_side_deduplication() -> None:
    case = Case(
        "demo",
        [HitEvent(100, "upper", 0.9), HitEvent(116, "upper", 0.8)],
        [HitEvent(100, "upper", 1.0)],
    )
    metrics = aggregate(
        [case], threshold=0.5, radius=8, same_side_radius=18, tolerance=12
    )
    assert metrics["predicted"] == 1
    assert metrics["precision"] == 1.0


def test_event_windows_are_merged_without_duplicate_frames() -> None:
    ranges = AlignedVideoPipeline._merge_event_ranges(
        [100, 120, 300], total_frames=400, before=21, after=36
    )
    assert ranges == [(79, 156), (279, 336)]


def test_event_windows_are_clipped_to_video_bounds() -> None:
    ranges = AlignedVideoPipeline._merge_event_ranges(
        [5, 395], total_frames=400, before=21, after=36
    )
    assert ranges == [(0, 41), (374, 399)]


def test_modality_quality_vector_preserves_training_scale() -> None:
    pose = np.array(
        [
            [[0.0, 0.0], [0.2, 0.4]],
            [[0.1, 0.2], [0.4, 0.8]],
            [[0.2, 0.4], [0.6, 1.2]],
        ],
        dtype=np.float32,
    )
    shuttle = np.array(
        [[0.0, 0.0], [0.2, 0.1], [0.7, 0.5]],
        dtype=np.float32,
    )

    quality = modality_quality_vector(pose, shuttle)
    expected_speed_std = np.linalg.norm(np.diff(shuttle, axis=0), axis=-1).std()
    expected_pose_motion = pose.reshape(len(pose), -1).std(axis=0).mean()

    assert quality.shape == (4,)
    assert quality.dtype == np.float32
    assert quality[0] == np.float32(1 / 3)
    assert quality[1] == np.float32(expected_speed_std)
    assert quality[2] == np.float32(1 / 6)
    assert quality[3] == np.float32(expected_pose_motion)


def test_rescues_opposite_side_candidate_bridging_same_side_hits() -> None:
    base = [HitEvent(100, "upper", 0.9), HitEvent(160, "upper", 0.9)]
    raw = base + [HitEvent(130, "lower", 0.61)]
    rescued = rescue_alternating_bridges(base, raw)
    assert [(event.frame, event.side) for event in rescued] == [
        (100, "upper"), (130, "lower"), (160, "upper")
    ]


def test_bridge_rescue_rejects_wrong_side_and_edge_echoes() -> None:
    base = [HitEvent(100, "upper", 0.9), HitEvent(160, "upper", 0.9)]
    raw = base + [
        HitEvent(130, "upper", 0.8),
        HitEvent(153, "lower", 0.8),
    ]
    assert rescue_alternating_bridges(base, raw) == base


def test_serve_rescue_accepts_guarded_low_score_serve() -> None:
    base = [HitEvent(100, "upper", 0.9), HitEvent(300, "upper", 0.9)]
    raw = base + [HitEvent(250, "lower", 0.41)]
    rescued = rescue_pre_rally_serves(base, raw)
    assert any(event.frame == 250 for event in rescued)


def test_adaptive_tta_skips_extra_windows_for_confident_prediction(monkeypatch) -> None:
    import torch

    pipeline = object.__new__(AlignedVideoPipeline)
    calls = []

    def extract(_cap, frame, _total, **_kwargs):
        calls.append(frame)
        return {"frame": frame}

    def predict(_feature, classifier=None):
        del classifier
        return {
            "stroke": "smash",
            "confidence": 0.9,
            "top2": ["smash", "clear"],
            "logits": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0]),
        }

    monkeypatch.setattr(pipeline, "extract_features", extract)
    monkeypatch.setattr(pipeline, "predict_detailed", predict)
    _feature, detail = pipeline.predict_detailed_adaptive_tta(
        None, 100, 200, "upper"
    )
    assert calls == [100]
    assert detail["tta_used"] is False
    assert detail["tta_windows"] == 1

