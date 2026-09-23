"""Multi-Task R(2+1)D-18 Backbone for Badminton Video Action Recognition.

Adapts torchvision's r2plus1d_18 backbone by replacing the single fc head with
two multi-task classification heads:
1. stroke_head: 8 classes (Clear, Smash, Drop, Net Shot, Lift, Drive, Net Attack, Serve)
2. side_head: 3 classes (Forehand, Backhand, Aroundhead)
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models.video import r2plus1d_18

from .fusion_head import SIDE_CLASSES, STROKE_CLASSES


class MultiTaskR2Plus1D(nn.Module):
    """Multi-Task (2+1)D ResNet Backbone for Badminton Video Analysis.
    
    Processes 16-frame cropped hitter video clips and extracts 512-dim visual features
    or produces multi-task classification logits.
    """
    def __init__(self, weights=None) -> None:
        super().__init__()
        self.backbone = r2plus1d_18(weights=weights)
        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.stroke_head = nn.Linear(features, len(STROKE_CLASSES))
        self.side_head = nn.Linear(features, len(SIDE_CLASSES))

    def forward(self, clips: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            clips: Tensor of shape (B, 3, T, H, W) normalized video clips.
            
        Returns:
            Tuple of (stroke_logits, side_logits).
        """
        features = self.backbone(clips)
        return self.stroke_head(features), self.side_head(features)

    def extract_features(self, clips: torch.Tensor) -> torch.Tensor:
        """Extract visual representation embedding (512-dim)."""
        return self.backbone(clips)

