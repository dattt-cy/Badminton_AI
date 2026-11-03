"""Scratch-trained KAP fusion and conservative Top-2 reranking models.

External visual/pose/tracking networks are treated as frozen feature extractors.
The modules in this file never load checkpoints in their constructors, which
makes random initialization and staged training explicit and auditable.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

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
        return stroke, side, gates


class KAPCrossTransModel(nn.Module):
    """Experimental KAP variant with pose-to-shuttle cross-attention.

    Unlike :class:`KAPScratchModel`, this model preserves the encoded pose and
    shuttle sequences until they have interacted.  Pose features provide the
    queries and shuttle features provide keys/values, following the central
    interaction idea evaluated by BST.  The original KAP tokens, phase pooling,
    quality gates and fusion transformer remain present so the experiment tests
    the incremental value of cross-attention instead of replacing the model.
    """

    def __init__(
        self,
        rgb_dim: int = 512,
        pose_dim: int = 68,
        court_dim: int = 4,
        shuttle_dim: int = 2,
        quality_dim: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.20,
        cross_heads: int = 4,
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

        self.cross_query_norm = nn.LayerNorm(hidden_dim)
        self.cross_key_value_norm = nn.LayerNorm(hidden_dim)
        self.pose_to_shuttle = nn.MultiheadAttention(
            hidden_dim, cross_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention_norm = nn.LayerNorm(hidden_dim)
        self.cross_feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.cross_output_norm = nn.LayerNorm(hidden_dim)
        self.cross_phase_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Gates remain independently learned for the four measured modalities.
        # The derived cross-token gate is conservative: it is high only when
        # both pose and shuttle are trusted.
        self.quality_gate = nn.Sequential(
            nn.Linear(quality_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.quality_gate[2].weight)
        nn.init.constant_(self.quality_gate[2].bias, 2.0)

        token_count = 5  # RGB, pose, court, shuttle, pose-to-shuttle cross token.
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.modality_embedding = nn.Parameter(
            torch.randn(1, token_count + 1, hidden_dim) * 0.02
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

    def _phase_pool(
        self, sequence: torch.Tensor, contact_distance: torch.Tensor
    ) -> torch.Tensor:
        distance = contact_distance.squeeze(-1)
        approach = _masked_pool(sequence, distance < -0.20)
        impact = _masked_pool(sequence, distance.abs() <= 0.20, maximum=True)
        flight = _masked_pool(sequence, distance > 0.20)
        return self.cross_phase_proj(torch.cat([approach, impact, flight], dim=-1))

    def forward(
        self,
        rgb: torch.Tensor,
        pose: torch.Tensor,
        court: torch.Tensor,
        shuttle: torch.Tensor,
        contact_distance: torch.Tensor,
        quality: torch.Tensor,
    ):
        pose_n = self._normalize(pose, self.pose_mean, self.pose_std)
        court_n = self._normalize(court, self.court_mean, self.court_std)
        shuttle_n = self._normalize(shuttle, self.shuttle_mean, self.shuttle_std)
        quality_n = self._normalize(quality, self.quality_mean, self.quality_std)

        pose_sequence, pose_token = self.pose_encoder(pose_n, contact_distance)
        _, court_token = self.court_encoder(court_n, contact_distance)
        shuttle_sequence, shuttle_token = self.shuttle_encoder(shuttle_n, contact_distance)

        cross_attention, _ = self.pose_to_shuttle(
            self.cross_query_norm(pose_sequence),
            self.cross_key_value_norm(shuttle_sequence),
            self.cross_key_value_norm(shuttle_sequence),
            need_weights=False,
        )
        cross_sequence = self.cross_attention_norm(pose_sequence + cross_attention)
        cross_sequence = self.cross_output_norm(
            cross_sequence + self.cross_feed_forward(cross_sequence)
        )
        cross_token = self._phase_pool(cross_sequence, contact_distance)

        gates = self.quality_gate(quality_n)
        cross_gate = torch.sqrt((gates[:, 1] * gates[:, 3]).clamp_min(1e-8))
        token_gates = torch.cat([gates, cross_gate.unsqueeze(1)], dim=1)
        tokens = torch.stack(
            [
                self.rgb_encoder(rgb),
                pose_token,
                court_token,
                shuttle_token,
                cross_token,
            ],
            dim=1,
        )
        tokens = tokens * token_gates.unsqueeze(-1)

        batch = rgb.shape[0]
        fusion_token = self.fusion_token.expand(batch, -1, -1)
        fused_sequence = torch.cat([fusion_token, tokens], dim=1) + self.modality_embedding
        fused = self.output_norm(self.fusion(fused_sequence)[:, 0])
        return self.stroke_head(fused), self.side_head(fused), gates


class UnitGate(nn.Module):
    """Quality gate of checkpoints trained without one: every modality keeps full weight."""

    def forward(self, quality: torch.Tensor) -> torch.Tensor:
        return torch.ones(quality.shape[0], 4, device=quality.device, dtype=quality.dtype)


class KAPCrossTransV2Model(nn.Module):
    """BST-Disentangled Dual-Branch CrossTrans with Quality-Gated Residual.

    Key improvements over v1:
    1. Slices 68-d pose into Upper Player (34) and Lower Player (34).
    2. Runs separate cross-attentions between each player and shuttle trajectory.
    3. Hitter Disambiguation module computes attention weights to isolate the active hitter.
    4. Quality-Gated Residual prevents TrackNet noise from corrupting the pose representation.
    """

    def __init__(
        self,
        rgb_dim: int = 512,
        pose_dim: int = 68,
        court_dim: int = 4,
        shuttle_dim: int = 2,
        quality_dim: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.20,
        cross_heads: int = 4,
        gate_free: bool = False,
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
        # Shared 34-dim pose encoder for both upper and lower players
        self.pose_encoder = ContactAwareTemporalEncoder(34, hidden_dim, dropout)
        self.court_encoder = ContactAwareTemporalEncoder(court_dim, hidden_dim, dropout)
        self.shuttle_encoder = ContactAwareTemporalEncoder(shuttle_dim, hidden_dim, dropout)

        # Dual-branch Cross Attention
        self.cross_query_norm1 = nn.LayerNorm(hidden_dim)
        self.cross_query_norm2 = nn.LayerNorm(hidden_dim)
        self.cross_key_value_norm = nn.LayerNorm(hidden_dim)

        self.p1_to_shuttle = nn.MultiheadAttention(
            hidden_dim, cross_heads, dropout=dropout, batch_first=True
        )
        self.p2_to_shuttle = nn.MultiheadAttention(
            hidden_dim, cross_heads, dropout=dropout, batch_first=True
        )

        # Hitter Disambiguation Head
        self.hitter_disambiguation = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

        self.cross_attention_norm = nn.LayerNorm(hidden_dim)
        self.cross_feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.cross_output_norm = nn.LayerNorm(hidden_dim)
        self.cross_phase_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if gate_free:
            self.quality_gate = UnitGate()
        else:
            self.quality_gate = nn.Sequential(
                nn.Linear(quality_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 4),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.quality_gate[2].weight)
            nn.init.constant_(self.quality_gate[2].bias, 2.0)

        token_count = 5
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.modality_embedding = nn.Parameter(
            torch.randn(1, token_count + 1, hidden_dim) * 0.02
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

    def _phase_pool(
        self, sequence: torch.Tensor, contact_distance: torch.Tensor
    ) -> torch.Tensor:
        distance = contact_distance.squeeze(-1)
        approach = _masked_pool(sequence, distance < -0.20)
        impact = _masked_pool(sequence, distance.abs() <= 0.20, maximum=True)
        flight = _masked_pool(sequence, distance > 0.20)
        return self.cross_phase_proj(torch.cat([approach, impact, flight], dim=-1))

    def forward(
        self,
        rgb: torch.Tensor,
        pose: torch.Tensor,
        court: torch.Tensor,
        shuttle: torch.Tensor,
        contact_distance: torch.Tensor,
        quality: torch.Tensor,
    ):
        pose_n = self._normalize(pose, self.pose_mean, self.pose_std)
        court_n = self._normalize(court, self.court_mean, self.court_std)
        shuttle_n = self._normalize(shuttle, self.shuttle_mean, self.shuttle_std)
        quality_n = self._normalize(quality, self.quality_mean, self.quality_std)

        # 1. Disentangle Upper Player (0..34) and Lower Player (34..68)
        p1 = pose_n[:, :, :34]
        p2 = pose_n[:, :, 34:]

        seq_p1, tok_p1 = self.pose_encoder(p1, contact_distance)
        seq_p2, tok_p2 = self.pose_encoder(p2, contact_distance)
        _, court_token = self.court_encoder(court_n, contact_distance)
        shuttle_sequence, shuttle_token = self.shuttle_encoder(shuttle_n, contact_distance)

        # 2. Dual Cross-Attentions
        shuttle_kv = self.cross_key_value_norm(shuttle_sequence)
        cross_p1, _ = self.p1_to_shuttle(
            self.cross_query_norm1(seq_p1), shuttle_kv, shuttle_kv, need_weights=False
        )
        cross_p2, _ = self.p2_to_shuttle(
            self.cross_query_norm2(seq_p2), shuttle_kv, shuttle_kv, need_weights=False
        )

        # 3. Hitter Disambiguation
        hitter_logits = self.hitter_disambiguation(
            torch.cat([tok_p1, tok_p2, shuttle_token], dim=-1)
        )
        hitter_weights = F.softmax(hitter_logits, dim=-1)
        w1 = hitter_weights[:, 0:1].unsqueeze(1)
        w2 = hitter_weights[:, 1:2].unsqueeze(1)

        active_pose = w1 * seq_p1 + w2 * seq_p2
        active_cross = w1 * cross_p1 + w2 * cross_p2
        active_pose_token = hitter_weights[:, 0:1] * tok_p1 + hitter_weights[:, 1:2] * tok_p2

        # 4. Quality-Gated Residual (Shuttle gate regulates cross-attention addition)
        gates = self.quality_gate(quality_n)
        shuttle_gate = gates[:, 3:4].unsqueeze(1)

        cross_sequence = self.cross_attention_norm(active_pose + shuttle_gate * active_cross)
        cross_sequence = self.cross_output_norm(
            cross_sequence + self.cross_feed_forward(cross_sequence)
        )
        cross_token = self._phase_pool(cross_sequence, contact_distance)

        # 5. Token Gating & Late Fusion
        cross_gate = torch.sqrt((gates[:, 1] * gates[:, 3]).clamp_min(1e-8))
        token_gates = torch.cat([gates, cross_gate.unsqueeze(1)], dim=1)
        tokens = torch.stack(
            [
                self.rgb_encoder(rgb),
                active_pose_token,
                court_token,
                shuttle_token,
                cross_token,
            ],
            dim=1,
        )
        tokens = tokens * token_gates.unsqueeze(-1)

        batch = rgb.shape[0]
        fusion_token = self.fusion_token.expand(batch, -1, -1)
        fused_sequence = torch.cat([fusion_token, tokens], dim=1) + self.modality_embedding
        fused = self.output_norm(self.fusion(fused_sequence)[:, 0])
        return self.stroke_head(fused), self.side_head(fused), gates

