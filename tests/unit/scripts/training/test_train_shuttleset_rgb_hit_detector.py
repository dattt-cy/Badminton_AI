from pathlib import Path

from scripts.training.train_shuttleset_rgb_hit_detector import build_samples


def test_build_samples_keeps_negatives_away_from_hits() -> None:
    matches = {"1": {"video_path": Path("match.mp4"), "hits": [(100, 1), (200, 2), (300, 1)]}}

    samples = build_samples(matches, 3, 2.0, 12, 0, 7)

    positives = [sample for sample in samples if sample.label]
    negatives = [sample for sample in samples if not sample.label]
    assert len(positives) == 3
    assert len(negatives) == 6
    assert all(min(abs(sample.center - hit) for hit in (100, 200, 300)) > 12 for sample in negatives)
