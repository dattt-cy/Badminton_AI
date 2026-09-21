from scripts.training.train_fine_badminton_rgb import confusion_metrics

import pytest
import torch


def test_confusion_metrics_reports_macro_values():
    confusion = torch.tensor([[2, 0], [1, 1]], dtype=torch.int64)

    metrics = confusion_metrics(confusion)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert metrics["support"] == [2, 2]
