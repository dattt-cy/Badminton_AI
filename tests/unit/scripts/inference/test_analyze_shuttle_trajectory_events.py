from types import SimpleNamespace

import numpy as np
import torch

from scripts.inference.analyze_shuttle_trajectory_events import assign_player


def fake_result() -> SimpleNamespace:
    boxes = torch.tensor([
        [400.0, 100.0, 600.0, 300.0],
        [420.0, 650.0, 620.0, 900.0],
    ])
    poses = torch.zeros((2, 17, 3), dtype=torch.float32)
    poses[0, 9] = torch.tensor([500.0, 180.0, 0.9])
    poses[0, 10] = torch.tensor([510.0, 190.0, 0.9])
    poses[1, 9] = torch.tensor([500.0, 700.0, 0.9])
    poses[1, 10] = torch.tensor([510.0, 710.0, 0.9])
    return SimpleNamespace(
        boxes=SimpleNamespace(xyxy=boxes, cls=torch.tensor([0.0, 0.0])),
        keypoints=SimpleNamespace(data=poses),
    )


def test_assign_player_uses_shuttle_to_wrist_distance_for_upper_player():
    side, distance = assign_player(fake_result(), np.array([505.0, 185.0]), 1000, 1000)
    assert side == "upper"
    assert distance < 10.0


def test_assign_player_uses_shuttle_to_wrist_distance_for_lower_player():
    side, distance = assign_player(fake_result(), np.array([505.0, 705.0]), 1000, 1000)
    assert side == "lower"
    assert distance < 10.0
