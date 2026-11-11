"""The team's RGB stroke encoder, a drop-in for the pipeline's R(2+1)D embedding."""

from __future__ import annotations

import torch
from torch import nn

from .badminton_hitnet import BadmintonHitNetV2

# Normalisation the pipeline applies to every RGB clip.
PIPELINE_MEAN = (0.43216, 0.394666, 0.37645)
PIPELINE_STD = (0.22803, 0.22145, 0.216989)
# Normalisation the encoder was trained with.
ENCODER_MEAN = (0.43, 0.43, 0.40)
ENCODER_STD = (0.23, 0.23, 0.23)
# Steps of the 16 crops spanning hit-30..hit+30 that precede, surround and follow contact.
CONTACT_PHASES = (slice(0, 6), slice(6, 10), slice(10, 16))


def _channel_stats(values: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(values).view(1, 3, 1, 1, 1)


class OwnRGBEncoder(nn.Module):
    """BadmintonHitNetV2 with an 8-way stroke head; the 224-d context vector is the embedding.

    With ``phases`` the embedding is instead one 224-d vector per contact phase, (B, 3, 224).
    """

    def __init__(self, phases: bool = False) -> None:
        super().__init__()
        self.net = BadmintonHitNetV2(num_classes=8)
        self.phases = phases
        self._context: torch.Tensor | None = None
        self._steps: torch.Tensor | None = None
        self.net.classifier.register_forward_hook(lambda _m, inputs, _o: setattr(self, "_context", inputs[0]))
        self.net.temporal_tcn2.register_forward_hook(lambda _m, _i, output: setattr(self, "_steps", output))

    def extract_features(self, clip: torch.Tensor) -> torch.Tensor:
        """Embed a clip (B, C, T, H, W) that carries the pipeline's normalisation."""
        raw = clip * _channel_stats(PIPELINE_STD).to(clip) + _channel_stats(PIPELINE_MEAN).to(clip)
        own = (raw.clamp(0, 1) - _channel_stats(ENCODER_MEAN).to(clip)) / _channel_stats(ENCODER_STD).to(clip)
        with torch.autocast(clip.device.type, dtype=torch.float16, enabled=clip.device.type == "cuda"):
            self.net(own)
            if self.phases:
                return torch.stack([self._steps[:, :, phase].mean(dim=-1) for phase in CONTACT_PHASES], dim=1).float()
        return self._context.float()
