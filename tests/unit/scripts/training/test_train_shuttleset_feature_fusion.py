import numpy as np

from pathlib import Path

from scripts.training.train_shuttleset_feature_fusion import (
    resample, select_rows, target_aligned_feature_arrays, target_indices,
)


def test_resample_preserves_endpoints_and_shape() -> None:
    values = np.asarray([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    output = resample(values, 5)
    assert output.shape == (5, 2)
    np.testing.assert_allclose(output[0], values[0])
    np.testing.assert_allclose(output[-1], values[-1])


def test_full_data_selection_returns_all_cached_valid_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,valid_coarse_label,split,coarse_label\n"
        "a,True,train,serve\n"
        "b,False,train,serve\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "a.npy", np.zeros(1))
    np.save(cache / "b.npy", np.zeros(1))

    rows = select_rows(manifest, cache, 1, 1, 1, full_data=True)

    assert [row["sample_id"] for row in rows] == ["a"]


def test_target_alignment_pads_and_marks_target() -> None:
    joints = np.ones((4, 2, 17, 2), dtype=np.float32)
    position = np.ones((4, 2, 2), dtype=np.float32)
    shuttle = np.ones((4, 2), dtype=np.float32)
    feature = target_aligned_feature_arrays(joints, position, shuttle, 2, length=46)
    assert feature.shape[0] == 46 * 75 + 74 * 2


def test_target_index_uses_previous_hit_and_limit() -> None:
    rows = [
        {"sample_id": "a", "match_id": "1", "set_id": "1", "rally": "2", "ball_round": "1", "hit_frame": "100"},
        {"sample_id": "b", "match_id": "1", "set_id": "1", "rally": "2", "ball_round": "2", "hit_frame": "180"},
    ]
    assert target_indices(rows) == {"a": 15, "b": 45}
