import numpy as np

from scripts.training.train_hit_candidate_selector import select_f1_threshold, selector_probability


def test_select_f1_threshold_uses_out_of_fold_f1():
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.3, 0.35, 0.9], dtype=np.float64)
    result = select_f1_threshold(labels, probabilities)
    predictions = probabilities >= result["threshold"]
    assert predictions.tolist() == [False, False, True, True]
    assert result["f1"] == 1.0


def test_serialized_random_forest_probability():
    selector = {
        "model": "random_forest",
        "trees": [{
            "children_left": [1, -1, -1],
            "children_right": [2, -1, -1],
            "feature": [0, -2, -2],
            "threshold": [0.5, -2.0, -2.0],
            "positive_probability": [0.5, 0.2, 0.9],
        }],
    }
    assert selector_probability([0.4], selector) == 0.2
    assert selector_probability([0.8], selector) == 0.9
