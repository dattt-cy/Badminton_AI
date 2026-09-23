"""Contact-Aware Reliability-Gated Multimodal Fusion Model (CR-Gated Fusion).

A novel architecture for badminton stroke classification:
1. Contact-Aware Alignment: embeds distance to contact point d_t = t - t_hit.
2. Temporal Modality Encoders: 1D Temporal Convolutions (TCN) preserving motion dynamics.
3. Reliability Gating: adaptive weighting g_m = softmax(MLP([z_m, q_m])) handling missing/noisy signals.
4. Cross-Modal Fusion: Transformer-based cross-modal attention between modality tokens.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TemporalConvEncoder(nn.Module):
    """1D Temporal Convolutional Network (TCN) for sequence modality encoding."""

    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, in_features)
        Returns:
            seq_out: (B, T, hidden_dim)
            pooled: (B, hidden_dim)
        """
        h = self.input_proj(x)  # (B, T, H)
        res = h
        h = h.transpose(1, 2)   # (B, H, T)
        h = self.relu(self.bn1(self.conv1(h)))
        h = self.dropout(h)
        h = self.bn2(self.conv2(h))
        h = h.transpose(1, 2)   # (B, T, H)
        seq_out = self.relu(h + res)
        pooled = self.pooling(seq_out.transpose(1, 2)).squeeze(-1)  # (B, H)
        return seq_out, pooled


class ReliabilityGate(nn.Module):
    """Computes dynamic modality reliability weights g_m based on representations and quality metrics."""

    def __init__(self, num_modalities: int = 4, hidden_dim: int = 128, quality_dim: int = 4):
        super().__init__()
        in_dim = num_modalities * hidden_dim + quality_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_modalities),
        )

    def forward(self, tokens: list[torch.Tensor], quality: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: list of (B, hidden_dim) tensors for [rgb, pose, court, shuttle]
            quality: (B, quality_dim) quality metric vector
        Returns:
            gated_tokens: (B, num_modalities, hidden_dim)
            weights: (B, num_modalities) softmax attention weights
        """
        concat_feat = torch.cat(tokens + [quality], dim=-1)
        logits = self.mlp(concat_feat)  # (B, num_modalities)
        weights = F.softmax(logits, dim=-1)  # (B, num_modalities)
        stacked = torch.stack(tokens, dim=1)  # (B, num_modalities, hidden_dim)
        gated = stacked * weights.unsqueeze(-1)
        return gated, weights


class CRGatedFusionModel(nn.Module):
    """Contact-Aware Reliability-Gated Multimodal Fusion Architecture."""

    def __init__(
        self,
        rgb_dim: int = 512,
        pose_dim: int = 68,
        court_dim: int = 4,
        shuttle_dim: int = 2,
        quality_dim: int = 4,
        hidden_dim: int = 128,
        num_stroke_classes: int = 8,
        num_side_classes: int = 3,
        dropout: float = 0.2,
        use_temporal: bool = True,
        use_contact: bool = True,
        use_gate: bool = True,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_contact = use_contact
        self.use_gate = use_gate
        self.use_cross_attention = use_cross_attention
        self.hidden_dim = hidden_dim

        # 1. Contact-Aware Embedding
        if use_contact:
            self.contact_embed = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),
            )

        # 2. Modality Encoders
        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if use_temporal:
            self.pose_encoder = TemporalConvEncoder(pose_dim, hidden_dim, dropout)
            self.court_encoder = TemporalConvEncoder(court_dim, hidden_dim, dropout)
            self.shuttle_encoder = TemporalConvEncoder(shuttle_dim, hidden_dim, dropout)
        else:
            # Flatten-MLP fallback (for ablation comparison with baseline)
            self.pose_encoder = nn.Linear(32 * pose_dim, hidden_dim)
            self.court_encoder = nn.Linear(32 * court_dim, hidden_dim)
            self.shuttle_encoder = nn.Linear(32 * shuttle_dim, hidden_dim)

        # 3. Reliability Gate
        if use_gate:
            self.gate = ReliabilityGate(num_modalities=4, hidden_dim=hidden_dim, quality_dim=quality_dim)

        # 4. Cross-Modal Fusion
        if use_cross_attention:
            self.fusion_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.modality_pos_embed = nn.Parameter(torch.randn(1, 5, hidden_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=256,
                dropout=dropout,
                activation="relu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.post_norm = nn.LayerNorm(hidden_dim)
        else:
            # Direct weighted sum + projection
            self.fusion_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        # 5. Multi-Task Classification Heads
        self.stroke_head = nn.Linear(hidden_dim, num_stroke_classes)
        self.side_head = nn.Linear(hidden_dim, num_side_classes)

    def forward(
        self,
        rgb: torch.Tensor,
        pose: torch.Tensor,
        court: torch.Tensor,
        shuttle: torch.Tensor,
        contact_dist: torch.Tensor,
        quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            stroke_logits: (B, num_stroke_classes)
            side_logits: (B, num_side_classes)
            gate_weights: (B, 4) or None
        """
        B = rgb.shape[0]

        # 1. Contact embedding addition
        if self.use_contact:
            c_emb = self.contact_embed(contact_dist)  # (B, T, hidden_dim)
        else:
            c_emb = 0.0

        # 2. Encode RGB
        z_rgb = self.rgb_proj(rgb)  # (B, hidden_dim)

        # 3. Encode Structured Sequences
        if self.use_temporal:
            _, z_pose = self.pose_encoder(pose)
            _, z_court = self.court_encoder(court)
            _, z_shuttle = self.shuttle_encoder(shuttle)
            if self.use_contact:
                # Add contact context
                z_pose = z_pose + c_emb.mean(dim=1)
                z_court = z_court + c_emb.mean(dim=1)
                z_shuttle = z_shuttle + c_emb.mean(dim=1)
        else:
            z_pose = F.relu(self.pose_encoder(pose.reshape(B, -1)))
            z_court = F.relu(self.court_encoder(court.reshape(B, -1)))
            z_shuttle = F.relu(self.shuttle_encoder(shuttle.reshape(B, -1)))

        tokens = [z_rgb, z_pose, z_court, z_shuttle]

        # 4. Reliability Gating
        gate_weights = None
        if self.use_gate:
            gated_tokens, gate_weights = self.gate(tokens, quality)
        else:
            gated_tokens = torch.stack(tokens, dim=1)

        # 5. Cross-Modal Fusion
        if self.use_cross_attention:
            # Prepend [FUSION] token
            f_token = self.fusion_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
            x_seq = torch.cat([f_token, gated_tokens], dim=1)  # (B, 5, hidden_dim)
            x_seq = x_seq + self.modality_pos_embed
            x_out = self.transformer(x_seq)
            fused = self.post_norm(x_out[:, 0])  # take [FUSION] output token (B, hidden_dim)
        else:
            fused = self.fusion_mlp(gated_tokens.reshape(B, -1))

        # 6. Multi-Task Heads
        stroke_logits = self.stroke_head(fused)
        side_logits = self.side_head(fused)

        return stroke_logits, side_logits, gate_weights
