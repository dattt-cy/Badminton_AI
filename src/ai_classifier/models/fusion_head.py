"""Multimodal Fusion Head for Badminton Stroke Recognition.

Fuses:
1. RGB spatial-temporal feature embedding from R(2+1)D-18 (512-dim)
2. Structured kinematics feature vector from Pose and Shuttlecock trajectory (256-dim projection)

Outputs:
1. Stroke classification logits (8 classes: Clear, Smash, Drop, Net Shot, Lift, Drive, Net Attack, Serve)
2. Stroke side classification logits (3 classes: Forehand, Backhand, Aroundhead)
"""

from __future__ import annotations

import torch
from torch import nn

STROKE_CLASSES = [
    "serve", "clear", "smash", "drop", "net_shot", "lift", "drive", "net_attack"
]
SIDE_CLASSES = ["forehand", "backhand", "aroundhead"]


class FusionHead(nn.Module):
    """Multimodal Fusion Neural Network.
    
    Combines visual motion representation with kinematic trajectories
    and joint dynamics, outputting multi-task predictions.
    """
    def __init__(
        self,
        rgb_dim: int = 512,
        structured_dim: int = 2176,
        hidden_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.structured = nn.Sequential(
            nn.LayerNorm(structured_dim),
            nn.Linear(structured_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(rgb_dim + hidden_dim),
            nn.Linear(rgb_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stroke = nn.Linear(hidden_dim, len(STROKE_CLASSES))
        self.side = nn.Linear(hidden_dim, len(SIDE_CLASSES))

    def forward(
        self, rgb: torch.Tensor, structured: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for multimodal fusion.
        
        Args:
            rgb: Tensor of shape (B, rgb_dim) from visual backbone.
            structured: Tensor of shape (B, structured_dim) from kinematics.
            
        Returns:
            Tuple of (stroke_logits, side_logits).
        """
        struct_feat = self.structured(structured)
        fused = torch.cat([rgb, struct_feat], dim=1)
        hidden = self.fusion(fused)
        return self.stroke(hidden), self.side(hidden)

