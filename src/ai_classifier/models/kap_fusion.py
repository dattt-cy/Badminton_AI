"""Scratch-trained KAP fusion and conservative Top-2 reranking models.

External visual/pose/tracking networks are treated as frozen feature extractors.
The modules in this file never load checkpoints in their constructors, which
makes random initialization and staged training explicit and auditable.
"""

from __future__ import annotations

import torch
from torch import nn

from .taxonomy import SIDE_CLASSES, STROKE_CLASSES


def _masked_pool(
    sequence: torch.Tensor,
    mask: torch.Tensor,
    *,
    maximum: bool = False,
) -> torch.Tensor:
    """Pool ``(B,T,H)`` features with a per-sample temporal mask."""
    mask = mask.unsqueeze(-1)
    if maximum:
        masked = sequence.masked_fill(~mask, torch.finfo(sequence.dtype).min)
        pooled = masked.max(dim=1).values
        valid = mask.any(dim=1)
        return torch.where(valid, pooled, sequence.mean(dim=1))
    count = mask.sum(dim=1).clamp_min(1)
    return (sequence * mask).sum(dim=1) / count


class ContactAwareTemporalEncoder(nn.Module):
    """Residual TCN with frame-level contact encoding and phase pooling."""

    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.contact_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh())
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
        )
        self.phase_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, sequence: torch.Tensor, contact_distance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Contact information is injected before temporal convolutions, so its
        # signed value remains associated with each individual frame.
        projected = self.input_proj(sequence) + self.contact_proj(contact_distance)
        temporal = self.temporal(projected.transpose(1, 2)).transpose(1, 2)
        encoded = torch.nn.functional.gelu(projected + temporal)

        distance = contact_distance.squeeze(-1)
        approach = _masked_pool(encoded, distance < -0.20)
        impact = _masked_pool(encoded, distance.abs() <= 0.20, maximum=True)
        flight = _masked_pool(encoded, distance > 0.20)
        pooled = self.phase_proj(torch.cat([approach, impact, flight], dim=-1))
        return encoded, pooled


class KAPScratchModel(nn.Module):
    """Kinematic and phase-aware multimodal fusion trained from random weights."""

    def __init__(
        self,
        rgb_dim: int = 512,
        pose_dim: int = 68,
        court_dim: int = 4,
        shuttle_dim: int = 2,
        quality_dim: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.register_buffer("pose_mean", torch.zeros(pose_dim))
        self.register_buffer("pose_std", torch.ones(pose_dim))
        self.register_buffer("court_mean", torch.zeros(court_dim))
        self.register_buffer("court_std", torch.ones(court_dim))
        self.register_buffer("shuttle_mean", torch.zeros(shuttle_dim))
        self.register_buffer("shuttle_std", torch.ones(shuttle_dim))
        self.register_buffer("quality_mean", torch.zeros(quality_dim))
        self.register_buffer("quality_std", torch.ones(quality_dim))

        self.rgb_encoder = nn.Sequential(
            nn.Linear(rgb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pose_encoder = ContactAwareTemporalEncoder(pose_dim, hidden_dim, dropout)
        self.court_encoder = ContactAwareTemporalEncoder(court_dim, hidden_dim, dropout)
        self.shuttle_encoder = ContactAwareTemporalEncoder(shuttle_dim, hidden_dim, dropout)

        # A sigmoid gate does not force all modalities to compete for a unit
        # softmax mass. It starts close to identity and learns attenuation only
        # when quality evidence supports it.
        num_modalities = 4
        self.quality_gate = nn.Sequential(
            nn.Linear(quality_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.quality_gate[2].weight)
        nn.init.constant_(self.quality_gate[2].bias, 2.0)

        self.fusion_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.modality_embedding = nn.Parameter(
            torch.randn(1, num_modalities + 1, hidden_dim) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=2)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.stroke_head = nn.Linear(hidden_dim, len(STROKE_CLASSES))
        self.side_head = nn.Linear(hidden_dim, len(SIDE_CLASSES))

    @torch.no_grad()
    def set_normalization(
        self,
        pose_mean: torch.Tensor,
        pose_std: torch.Tensor,
        court_mean: torch.Tensor,
        court_std: torch.Tensor,
        shuttle_mean: torch.Tensor,
        shuttle_std: torch.Tensor,
        quality_mean: torch.Tensor,
        quality_std: torch.Tensor,
    ) -> None:
        for name, value in locals().copy().items():
            if name == "self":
                continue
            getattr(self, name).copy_(value.to(getattr(self, name)))

    @staticmethod
    def _normalize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (value - mean) / std.clamp_min(1e-5)

    def forward(
        self,
        rgb: torch.Tensor,
        pose: torch.Tensor,
        court: torch.Tensor,
        shuttle: torch.Tensor,
        contact_distance: torch.Tensor,
        quality: torch.Tensor,
        *,
        return_features: bool = False,
    ):
        pose_n = self._normalize(pose, self.pose_mean, self.pose_std)
        court_n = self._normalize(court, self.court_mean, self.court_std)
        shuttle_n = self._normalize(shuttle, self.shuttle_mean, self.shuttle_std)
        quality_n = self._normalize(quality, self.quality_mean, self.quality_std)

        _, pose_token = self.pose_encoder(pose_n, contact_distance)
        _, court_token = self.court_encoder(court_n, contact_distance)
        _, shuttle_token = self.shuttle_encoder(shuttle_n, contact_distance)
        tokens = torch.stack(
            [self.rgb_encoder(rgb), pose_token, court_token, shuttle_token], dim=1
        )
        gates = self.quality_gate(quality_n)
        tokens = tokens * gates.unsqueeze(-1)

        batch = rgb.shape[0]
        fusion_token = self.fusion_token.expand(batch, -1, -1)
        fused_sequence = torch.cat([fusion_token, tokens], dim=1) + self.modality_embedding
        fused = self.output_norm(self.fusion(fused_sequence)[:, 0])
        stroke = self.stroke_head(fused)
        side = self.side_head(fused)
        if return_features:
            return stroke, side, gates, fused
        return stroke, side, gates


def physics_kinematics(
    shuttle: torch.Tensor,
    court: torch.Tensor,
    contact_distance: torch.Tensor,
) -> torch.Tensor:
    """Extract differentiable post-contact kinematics without fixed frame indices."""
    # Derive the index from the actual cache metadata. This keeps phase pooling
    # and physics extraction on the same definition of contact.
    contact = int(contact_distance[0, :, 0].abs().argmin())
    early = min(contact + max(1, shuttle.shape[1] // 6), shuttle.shape[1] - 1)
    middle = min(contact + max(2, shuttle.shape[1] // 3), shuttle.shape[1] - 1)
    late = min(contact + max(3, shuttle.shape[1] // 2), shuttle.shape[1] - 1)

    p_hit, p_early, p_mid, p_late = (
        shuttle[:, contact], shuttle[:, early], shuttle[:, middle], shuttle[:, late]
    )
    v_early = p_early - p_hit
    v_late = p_late - p_mid
    speed_early = v_early.norm(dim=-1, keepdim=True)
    speed_late = v_late.norm(dim=-1, keepdim=True)
    deceleration = speed_early - speed_late
    ratio = speed_late / speed_early.clamp_min(1e-4)
    displacement = (p_late - p_hit).norm(dim=-1, keepdim=True)
    court_y = court[:, contact, 1:2]
    forward_sign = torch.sign(court_y - 0.5 + 1e-5)
    forward_early = -(p_early[:, 1:2] - p_hit[:, 1:2]) * forward_sign
    forward_total = -(p_late[:, 1:2] - p_hit[:, 1:2]) * forward_sign
    lateral = (p_late[:, 0:1] - p_hit[:, 0:1]).abs()
    curvature = ((p_late[:, 1:2] - p_mid[:, 1:2]) - (p_mid[:, 1:2] - p_hit[:, 1:2])) * forward_sign
    features = torch.cat([
        v_early, speed_early, v_late, speed_late, deceleration, ratio,
        displacement, forward_early, forward_total, lateral, curvature, court_y,
    ], dim=-1)
    return torch.nan_to_num(features)


class Top2Reranker(nn.Module):
    """Predict whether KAP rank 1 should be replaced with rank 2."""

    def __init__(self, input_dim: int = 178) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


class PairSpecialist(nn.Module):
    """Binary classifier for one known confusing pair of stroke classes."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 96),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(96, 32),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def kap_reranker_features(
    model: KAPScratchModel,
    rgb: torch.Tensor,
    pose: torch.Tensor,
    court: torch.Tensor,
    shuttle: torch.Tensor,
    contact: torch.Tensor,
    quality: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the same 178-D representation directly from KAP outputs."""
    logits, _side, gates, fused = model(
        rgb, pose, court, shuttle, contact, quality, return_features=True
    )
    probabilities = logits.softmax(dim=1)
    top2_probability, top2_indices = probabilities.topk(2, dim=1)
    pair_one_hot = torch.nn.functional.one_hot(
        top2_indices, num_classes=len(STROKE_CLASSES)
    ).float().flatten(1)
    physics = physics_kinematics(shuttle, court, contact)
    normalized_quality = model._normalize(
        quality, model.quality_mean, model.quality_std
    )
    margin = top2_probability[:, :1] - top2_probability[:, 1:2]
    entropy = -(
        probabilities * probabilities.clamp_min(1e-8).log()
    ).sum(dim=1, keepdim=True)
    features = torch.cat([
        probabilities, top2_probability, margin, entropy, pair_one_hot,
        fused, physics, normalized_quality, gates,
    ], dim=1)
    return features, logits, top2_indices
