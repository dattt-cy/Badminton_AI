"""Neural Disambiguation Head (NDH) for Multimodal Badminton Stroke Classification.

Self-contained module housing:
- augment_shuttle_kinematics
- PhaseAwareTemporalEncoder
- ReliabilityGate
- KAPFusionModel
- extract_exit_kinematics
- NeuralDisambiguationHead
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ai_classifier.models import SIDE_CLASSES, STROKE_CLASSES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def augment_shuttle_kinematics(shuttle: torch.Tensor) -> torch.Tensor:
    """Augment 2D shuttle trajectory with 6 kinematic features -> 8D."""
    diff = torch.zeros_like(shuttle)
    diff[:, 1:] = shuttle[:, 1:] - shuttle[:, :-1]
    speed = torch.norm(diff, dim=-1, keepdim=True)
    acc = torch.zeros_like(speed)
    acc[:, 1:] = speed[:, 1:] - speed[:, :-1]
    denom = speed + 1e-6
    sin_th = diff[:, :, 1:2] / denom
    cos_th = diff[:, :, 0:1] / denom
    return torch.cat([shuttle, diff, speed, acc, sin_th, cos_th], dim=-1)


class PhaseAwareTemporalEncoder(nn.Module):
    """1D TCN with Phase-Aware Temporal Pooling (Approach, Impact, Flight)."""

    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.phase_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        res = h
        h = h.transpose(1, 2)
        h = self.relu(self.bn1(self.conv1(h)))
        h = self.dropout(h)
        h = self.bn2(self.conv2(h))
        h = h.transpose(1, 2)
        seq_out = self.relu(h + res)

        # 3 badminton stroke phases
        p_approach = seq_out[:, :14].mean(dim=1)
        p_impact = seq_out[:, 14:18].max(dim=1).values
        p_flight = seq_out[:, 18:].mean(dim=1)
        phase_cat = torch.cat([p_approach, p_impact, p_flight], dim=-1)
        pooled = self.phase_proj(phase_cat)
        return seq_out, pooled


class ReliabilityGate(nn.Module):
    """Computes dynamic modality reliability weights based on tokens and quality metrics."""

    def __init__(self, num_modalities: int = 4, hidden_dim: int = 128, quality_dim: int = 4):
        super().__init__()
        in_dim = num_modalities * hidden_dim + quality_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_modalities),
        )

    def forward(self, tokens: list[torch.Tensor], quality: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        concat_feat = torch.cat(tokens + [quality], dim=-1)
        logits = self.mlp(concat_feat)
        weights = F.softmax(logits, dim=-1)
        stacked = torch.stack(tokens, dim=1)
        gated = stacked * weights.unsqueeze(-1)
        return gated, weights


class KAPFusionModel(nn.Module):
    """Kinematic & Phase-Aware Multimodal Fusion Model."""

    def __init__(
        self,
        rgb_dim: int = 512,
        pose_dim: int = 68,
        court_dim: int = 4,
        shuttle_dim: int = 8,
        quality_dim: int = 4,
        hidden_dim: int = 128,
        num_stroke_classes: int = 8,
        num_side_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.contact_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.pose_encoder = PhaseAwareTemporalEncoder(pose_dim, hidden_dim, dropout)
        self.court_encoder = PhaseAwareTemporalEncoder(court_dim, hidden_dim, dropout)
        self.shuttle_encoder = PhaseAwareTemporalEncoder(shuttle_dim, hidden_dim, dropout)
        self.gate = ReliabilityGate(num_modalities=4, hidden_dim=hidden_dim, quality_dim=quality_dim)
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
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
        self.stroke_head = nn.Linear(hidden_dim, num_stroke_classes)
        self.side_head = nn.Linear(hidden_dim, num_side_classes)

    def forward(self, rgb, pose, court, shuttle, contact, quality, return_features=False):
        B = rgb.size(0)
        if shuttle.size(-1) == 2:
            shuttle = augment_shuttle_kinematics(shuttle)
        c_emb = self.contact_embed(contact)
        c_pool = c_emb.mean(dim=1)
        z_rgb = self.rgb_proj(rgb)
        _, z_pose = self.pose_encoder(pose)
        _, z_court = self.court_encoder(court)
        _, z_shuttle = self.shuttle_encoder(shuttle)
        z_pose = z_pose + c_pool
        z_shuttle = z_shuttle + c_pool
        tokens = [z_rgb, z_pose, z_court, z_shuttle]
        gated_tokens, gate_weights = self.gate(tokens, quality)
        cls_token = self.fusion_token.expand(B, -1, -1)
        seq = torch.cat([cls_token, gated_tokens], dim=1) + self.modality_pos_embed
        out = self.transformer(seq)
        fused = self.post_norm(out[:, 0])
        stroke_logits = self.stroke_head(fused)
        side_logits = self.side_head(fused)
        if return_features:
            return stroke_logits, side_logits, gate_weights, fused
        return stroke_logits, side_logits, gate_weights


def extract_exit_kinematics(shuttle: torch.Tensor, court: torch.Tensor) -> torch.Tensor:
    """Extracts post-impact directional exit kinematics (5-D)."""
    vy_post = shuttle[:, 16, 1:2] - shuttle[:, 11, 1:2]
    vx_post = torch.abs(shuttle[:, 16, 0:1] - shuttle[:, 11, 0:1])
    diff = shuttle[:, 16] - shuttle[:, 11]
    speed_post = torch.norm(diff, dim=-1, keepdim=True) * 20.0
    court_y = court[:, 10, 1:2]
    fwd_sign = torch.sign(court_y - 0.5 + 1e-5)
    vy_fwd = vy_post * fwd_sign
    kin = torch.cat([vy_post, vx_post, speed_post, court_y, vy_fwd], dim=-1)
    return torch.nan_to_num(kin, nan=0.0, posinf=0.0, neginf=0.0)


class NeuralDisambiguationHead(nn.Module):
    """Residual Neural Disambiguation Head (NDH v1)."""

    def __init__(self, base_kap_path: str = "work_dirs/cr_gated_fusion/cr_gated_kap/best.pth"):
        super().__init__()
        self.base_model = KAPFusionModel().to(DEVICE)
        ckpt = torch.load(base_kap_path, map_location=DEVICE, weights_only=False)
        self.base_model.load_state_dict(ckpt["model"])
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = False

        self.kin_proj = nn.Sequential(
            nn.Linear(5, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
        )

        self.residual_mlp = nn.Sequential(
            nn.Linear(8 + 128 + 64, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 8),
        )

        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, rgb, pose, court, shuttle, contact, quality):
        with torch.no_grad():
            s_logits, side_logits, _, fused = self.base_model(
                rgb, pose, court, shuttle, contact, quality, return_features=True
            )

        raw_shuttle = shuttle[..., :2] if shuttle.size(-1) > 2 else shuttle
        raw_court = court[..., :4] if court.size(-1) > 4 else court
        kin = extract_exit_kinematics(raw_shuttle, raw_court)
        z_kin = self.kin_proj(kin)

        combined = torch.cat([s_logits, fused, z_kin], dim=-1)
        delta = self.residual_mlp(combined)

        refined_logits = s_logits + self.gamma * delta
        return refined_logits, side_logits


def extract_post_flight_kinematics_12d(shuttle: torch.Tensor, court: torch.Tensor) -> torch.Tensor:
    """Extracts 12-D post-impact flight kinematics (aligned with contact frame index 10)."""
    p_hit = shuttle[:, 10]
    p_early = shuttle[:, 16]
    p_mid = shuttle[:, 22]
    p_late = shuttle[:, 28]

    v_early = (p_early - p_hit) * 10.0
    speed_early = torch.norm(v_early, dim=-1, keepdim=True)

    v_late = (p_late - p_mid) * 10.0
    speed_late = torch.norm(v_late, dim=-1, keepdim=True)

    decel = speed_early - speed_late
    disp_norm = torch.norm(p_late - p_hit, dim=-1, keepdim=True)

    court_y = court[:, 10, 1:2]
    fwd_sign = torch.sign(court_y - 0.5 + 1e-5)
    fwd_dy = -(p_late[:, 1:2] - p_hit[:, 1:2]) * fwd_sign
    fwd_dy_early = -(p_early[:, 1:2] - p_hit[:, 1:2]) * fwd_sign
    abs_dx = torch.abs(p_late[:, 0:1] - p_hit[:, 0:1])

    kin = torch.cat(
        [v_early, speed_early, v_late, speed_late, decel, disp_norm, fwd_dy, fwd_dy_early, abs_dx, court_y],
        dim=-1,
    )
    return torch.nan_to_num(kin, nan=0.0, posinf=0.0, neginf=0.0)


class TwinDisambiguationHead(nn.Module):
    """NDH v2 with 12-D Flight Kinematics and Twin-Disambiguation Residual MLP."""

    def __init__(self, base_kap_path: str = "work_dirs/cr_gated_fusion/cr_gated_kap/best.pth"):
        super().__init__()
        self.base_model = KAPFusionModel().to(DEVICE)
        ckpt = torch.load(base_kap_path, map_location=DEVICE, weights_only=False)
        self.base_model.load_state_dict(ckpt["model"])
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = False

        self.kin_proj = nn.Sequential(
            nn.Linear(12, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
        )

        self.residual_mlp = nn.Sequential(
            nn.Linear(8 + 128 + 64, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 8),
        )

        self.gamma = nn.Parameter(torch.tensor(0.6))

    def forward(self, rgb, pose, court, shuttle, contact, quality):
        with torch.no_grad():
            s_logits, side_logits, _, fused = self.base_model(
                rgb, pose, court, shuttle, contact, quality, return_features=True
            )

        raw_shuttle = shuttle[..., :2] if shuttle.size(-1) > 2 else shuttle
        raw_court = court[..., :4] if court.size(-1) > 4 else court
        kin = extract_post_flight_kinematics_12d(raw_shuttle, raw_court)
        z_kin = self.kin_proj(kin)

        combined = torch.cat([s_logits, fused, z_kin], dim=-1)
        delta = self.residual_mlp(combined)

        refined_logits = s_logits + self.gamma * delta
        return refined_logits, side_logits
