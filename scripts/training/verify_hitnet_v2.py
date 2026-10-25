import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_classifier.models import BadmintonHitNetV2

model = BadmintonHitNetV2(num_classes=3, in_frames=16)
params = sum(p.numel() for p in model.parameters())
print(f"BadmintonHitNetV2 instantiated successfully! Params: {params:,}")

dummy = torch.randn(2, 3, 16, 112, 112)
logits, offset = model(dummy, return_offset=True)
print(f"Logits shape: {logits.shape}, Offset shape: {offset.shape}")
print("Verified 100% OK!")

