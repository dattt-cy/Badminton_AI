import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "paper"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "training"))

from ai_classifier.models import KAPCrossTransV2Model, OwnRGBEncoder
from ai_classifier.models.own_rgb_encoder import ENCODER_MEAN, ENCODER_STD, PIPELINE_MEAN, PIPELINE_STD
from train_kap_ablation import KAPAblationModel
from train_kap_rgb_phases import SLOTS, KAPRgbPhases


def _stats(values):
    return torch.tensor(values).view(1, 3, 1, 1, 1)


def test_extract_features_undoes_pipeline_normalisation():
    torch.manual_seed(0)
    encoder = OwnRGBEncoder().eval()
    raw = torch.rand(2, 3, 16, 112, 112)
    pipeline_clip = (raw - _stats(PIPELINE_MEAN)) / _stats(PIPELINE_STD)
    with torch.inference_mode():
        embedding = encoder.extract_features(pipeline_clip)
        encoder.net((raw - _stats(ENCODER_MEAN)) / _stats(ENCODER_STD))
    assert embedding.shape == (2, 224)
    assert torch.allclose(embedding, encoder._context, atol=1e-4)


def test_phase_model_matches_the_model_it_was_trained_as():
    torch.manual_seed(0)
    trained = KAPRgbPhases(SLOTS["phases"]).eval()
    deployed = KAPCrossTransV2Model(rgb_dim=224, gate_free=True, rgb_phases=3).eval()
    deployed.load_state_dict(trained.state_dict())
    rgb = torch.randn(3, 4, 224)
    rest = (torch.randn(3, 32, 68), torch.randn(3, 32, 4), torch.randn(3, 32, 2), torch.rand(3, 32, 1), torch.rand(3, 4))
    with torch.inference_mode():
        stroke_a, side_a, _ = trained(rgb, *rest)
        stroke_b, side_b, _ = deployed(rgb[:, SLOTS["phases"]], *rest)
    assert torch.allclose(stroke_a, stroke_b, atol=1e-5)
    assert torch.allclose(side_a, side_b, atol=1e-5)


def test_phase_encoder_returns_one_vector_per_phase():
    encoder = OwnRGBEncoder(phases=True).eval()
    with torch.inference_mode():
        assert encoder.extract_features(torch.rand(2, 3, 16, 112, 112)).shape == (2, 3, 224)


def test_gate_free_model_matches_the_model_it_was_trained_as():
    torch.manual_seed(0)
    trained = KAPAblationModel(gate="none", rgb_dim=224).eval()
    deployed = KAPCrossTransV2Model(rgb_dim=224, gate_free=True).eval()
    deployed.load_state_dict(trained.state_dict())
    inputs = (torch.randn(3, 224), torch.randn(3, 32, 68), torch.randn(3, 32, 4), torch.randn(3, 32, 2),
              torch.rand(3, 32, 1), torch.rand(3, 4))
    with torch.inference_mode():
        stroke_a, side_a, _ = trained(*inputs)
        stroke_b, side_b, _ = deployed(*inputs)
    assert torch.allclose(stroke_a, stroke_b, atol=1e-5)
    assert torch.allclose(side_a, side_b, atol=1e-5)
