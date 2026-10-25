"""BadmintonHitNet & BadmintonHitNetV2: Spatio-Temporal Hit Spotting from Scratch.

Theoretical & Architectural Foundations:
- TTNet (CVPRW 2020): Frame differencing + lightweight spatio-temporal separable convolutions
  for high-speed (>100 FPS) racket sports event spotting from scratch.
- E2E-Spot (ECCV 2022): Elimination of temporal pooling smearing; multi-scale dilated temporal
  convolutions (MS-TCN) capturing preparation, impact, and follow-through.
- MobileNetV3 (ICCV 2019): Squeeze-and-Excitation (SE) channel attention for high-velocity
  micro-objects (shuttlecock & racket head) against complex background court textures.

Key Features:
1. Dual-stream motion input: RGB + Frame Difference ΔI_t = I_t - I_{t-1} (6 channels).
2. Squeeze-and-Excitation (SE) inverted residual spatial feature encoder.
3. 4-scale Dilated Temporal Convolutional Network (MS-TCN) spanning dilations d ∈ {1, 2, 4, 8}.
4. Dual-task prediction heads:
   - Hit Spotting (3 classes: no_hit, upper_hit, lower_hit) with Focal Loss.
   - Temporal Offset Regression (Δt ∈ [-2, +2] frames) with Smooth L1 Loss.
5. Fully compatible drop-in interface: forward(clips) -> logits of shape (B, 3).
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation (SE) Block for Channel Attention
# ---------------------------------------------------------------------------

class SEBlock2D(nn.Module):
    """Squeeze-and-Excitation channel attention for micro-object highlighting."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


# ---------------------------------------------------------------------------
# Inverted Residual Blocks (V1 and V2 with SE)
# ---------------------------------------------------------------------------

class InvertedResidual2D(nn.Module):
    """Mobile inverted bottleneck block with depthwise-separable convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 2,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.use_residual = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish(inplace=True),
            ])
        layers.extend([
            # Depthwise convolution
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.Hardswish(inplace=True),
        ])
        if use_se:
            layers.append(SEBlock2D(hidden_dim, reduction=4))
        layers.extend([
            # Pointwise linear projection
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Multi-Scale Dilated TCN (V1 with 3 branches, V2 with 4 branches)
# ---------------------------------------------------------------------------

class MultiScaleDilatedTCN(nn.Module):
    """Multi-Scale Dilated 1D Temporal Convolutional Block (MS-TCN) - 3 Scales."""

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        branch_channels = channels // 4

        self.branch1 = nn.Sequential(
            nn.Conv1d(channels, branch_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(branch_channels),
            nn.Hardswish(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(channels, branch_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(branch_channels),
            nn.Hardswish(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(channels, branch_channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(branch_channels),
            nn.Hardswish(inplace=True),
        )
        self.branch_pw = nn.Sequential(
            nn.Conv1d(channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.Hardswish(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv1d(branch_channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        bpw = self.branch_pw(x)
        merged = torch.cat([b1, b2, b3, bpw], dim=1)
        out = self.proj(merged)
        return residual + self.dropout(out)


class MultiScaleDilatedTCNV2(nn.Module):
    """Upgraded 4-Scale Dilated TCN (d ∈ {1, 2, 4, 8}) spanning full 16-frame span."""

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        branch_channels = channels // 4

        self.branch1 = nn.Sequential(
            nn.Conv1d(channels, branch_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(branch_channels),
            nn.Hardswish(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(channels, branch_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(branch_channels),
            nn.Hardswish(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(channels, branch_channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(branch_channels),
            nn.Hardswish(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv1d(channels, branch_channels, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(branch_channels),
            nn.Hardswish(inplace=True),
        )
        self.branch_pw = nn.Sequential(
            nn.Conv1d(channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.Hardswish(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv1d(branch_channels * 5, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        bpw = self.branch_pw(x)
        merged = torch.cat([b1, b2, b3, b4, bpw], dim=1)
        out = self.proj(merged)
        return residual + self.dropout(out)


# ---------------------------------------------------------------------------
# BadmintonHitNet (Version 1 - Champion 0.55M params)
# ---------------------------------------------------------------------------

class BadmintonHitNet(nn.Module):
    """BadmintonHitNet V1: Custom Spatio-Temporal Hit Detector (Train-From-Scratch)."""

    def __init__(
        self,
        num_classes: int = 3,
        in_frames: int = 16,
        dropout: float = 0.2,
        spatial_dim: int = 144,
        temporal_dim: int = 192,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_frames = in_frames
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim

        self.spatial_stem = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 112 -> 56
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True),
        )
        self.spatial_stage1 = nn.Sequential(
            InvertedResidual2D(32, 48, stride=2, expand_ratio=2),             # 56 -> 28
            InvertedResidual2D(48, 48, stride=1, expand_ratio=2),
        )
        self.spatial_stage2 = nn.Sequential(
            InvertedResidual2D(48, 96, stride=2, expand_ratio=3),             # 28 -> 14
            InvertedResidual2D(96, 96, stride=1, expand_ratio=2),
        )
        self.spatial_stage3 = nn.Sequential(
            InvertedResidual2D(96, spatial_dim, stride=2, expand_ratio=3),    # 14 -> 7
            InvertedResidual2D(spatial_dim, spatial_dim, stride=1, expand_ratio=2),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.temporal_proj = nn.Conv1d(spatial_dim, temporal_dim, kernel_size=1)
        self.temporal_tcn1 = MultiScaleDilatedTCN(temporal_dim, dropout=dropout * 0.5)
        self.temporal_tcn2 = MultiScaleDilatedTCN(temporal_dim, dropout=dropout * 0.5)

        self.temporal_attention = nn.Sequential(
            nn.Conv1d(temporal_dim, 1, kernel_size=1),
            nn.Softmax(dim=-1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim, 64),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self.offset_regressor = nn.Sequential(
            nn.Linear(temporal_dim, 32),
            nn.Hardswish(inplace=True),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_motion_channels(self, x: torch.Tensor) -> torch.Tensor:
        diff_0 = torch.zeros_like(x[:, :, :1, :, :])
        diff_rest = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        diff = torch.cat([diff_0, diff_rest], dim=2)
        return torch.cat([x, diff], dim=1)

    def forward(
        self,
        clips: torch.Tensor,
        return_offset: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, c, t, h, w = clips.shape
        motion_clips = self.extract_motion_channels(clips)

        spatial_in = motion_clips.permute(0, 2, 1, 3, 4).contiguous().view(b * t, 6, h, w)
        feat = self.spatial_stem(spatial_in)
        feat = self.spatial_stage1(feat)
        feat = self.spatial_stage2(feat)
        feat = self.spatial_stage3(feat)
        feat_pooled = self.spatial_pool(feat).view(b, t, self.spatial_dim)

        temp_in = feat_pooled.permute(0, 2, 1)
        temp_proj = self.temporal_proj(temp_in)
        tcn_out = self.temporal_tcn1(temp_proj)
        tcn_out = self.temporal_tcn2(tcn_out)

        attn_weights = self.temporal_attention(tcn_out)
        context_vector = torch.sum(tcn_out * attn_weights, dim=-1)

        logits = self.classifier(context_vector)

        if not return_offset:
            return logits

        offset = self.offset_regressor(context_vector) * 2.0
        return logits, offset


# ---------------------------------------------------------------------------
# BadmintonHitNetV2 (Upgraded SE-Attention & 4-Scale MS-TCN ~1.1M params)
# ---------------------------------------------------------------------------

class BadmintonHitNetV2(nn.Module):
    """BadmintonHitNetV2: Supercharged with Squeeze-and-Excitation & 4-Scale MS-TCN."""

    def __init__(
        self,
        num_classes: int = 3,
        in_frames: int = 16,
        dropout: float = 0.2,
        spatial_dim: int = 160,
        temporal_dim: int = 224,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_frames = in_frames
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim

        # 1. 2D Spatial Encoder with Squeeze-and-Excitation
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 112 -> 56
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True),
        )
        self.spatial_stage1 = nn.Sequential(
            InvertedResidual2D(32, 48, stride=2, expand_ratio=2, use_se=True),  # 56 -> 28
            InvertedResidual2D(48, 48, stride=1, expand_ratio=2, use_se=True),
        )
        self.spatial_stage2 = nn.Sequential(
            InvertedResidual2D(48, 96, stride=2, expand_ratio=3, use_se=True),  # 28 -> 14
            InvertedResidual2D(96, 96, stride=1, expand_ratio=2, use_se=True),
            InvertedResidual2D(96, 96, stride=1, expand_ratio=2, use_se=True),
        )
        self.spatial_stage3 = nn.Sequential(
            InvertedResidual2D(96, spatial_dim, stride=2, expand_ratio=3, use_se=True),  # 14 -> 7
            InvertedResidual2D(spatial_dim, spatial_dim, stride=1, expand_ratio=2, use_se=True),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 2. 4-Scale Temporal Modeling
        self.temporal_proj = nn.Conv1d(spatial_dim, temporal_dim, kernel_size=1)
        self.temporal_tcn1 = MultiScaleDilatedTCNV2(temporal_dim, dropout=dropout * 0.5)
        self.temporal_tcn2 = MultiScaleDilatedTCNV2(temporal_dim, dropout=dropout * 0.5)

        # Contact-aware temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(temporal_dim, 1, kernel_size=1),
            nn.Softmax(dim=-1),
        )

        # 3. Dual Prediction Heads
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim, 96),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )

        self.offset_regressor = nn.Sequential(
            nn.Linear(temporal_dim, 48),
            nn.Hardswish(inplace=True),
            nn.Linear(48, 1),
            nn.Tanh(),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_motion_channels(self, x: torch.Tensor) -> torch.Tensor:
        diff_0 = torch.zeros_like(x[:, :, :1, :, :])
        diff_rest = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        diff = torch.cat([diff_0, diff_rest], dim=2)
        return torch.cat([x, diff], dim=1)

    def forward(
        self,
        clips: torch.Tensor,
        return_offset: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, c, t, h, w = clips.shape
        motion_clips = self.extract_motion_channels(clips)

        spatial_in = motion_clips.permute(0, 2, 1, 3, 4).contiguous().view(b * t, 6, h, w)
        feat = self.spatial_stem(spatial_in)
        feat = self.spatial_stage1(feat)
        feat = self.spatial_stage2(feat)
        feat = self.spatial_stage3(feat)
        feat_pooled = self.spatial_pool(feat).view(b, t, self.spatial_dim)

        temp_in = feat_pooled.permute(0, 2, 1)
        temp_proj = self.temporal_proj(temp_in)
        tcn_out = self.temporal_tcn1(temp_proj)
        tcn_out = self.temporal_tcn2(tcn_out)

        attn_weights = self.temporal_attention(tcn_out)
        context_vector = torch.sum(tcn_out * attn_weights, dim=-1)

        logits = self.classifier(context_vector)

        if not return_offset:
            return logits

        offset = self.offset_regressor(context_vector) * 2.0
        return logits, offset
