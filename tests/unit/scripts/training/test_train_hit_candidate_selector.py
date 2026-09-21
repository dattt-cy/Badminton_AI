import numpy as np

from scripts.training.train_hit_candidate_selector import select_f1_threshold


def test_select_f1_threshold_uses_out_of_fold_f1():
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.3, 0.35, 0.9], dtype=np.float64)
    result = select_f1_threshold(labels, probabilities)
    predictions = probabilities >= result["threshold"]
    assert predictions.tolist() == [False, False, True, True]
    assert result["f1"] == 1.0
