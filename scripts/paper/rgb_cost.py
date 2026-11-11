"""Latency (batch 1, GPU) and parameters of the RGB branch versus the KAP fusion model.

Times R(2+1)D-18 feature extraction on one 16x112x112 hitter clip, and the KAP
classifier with and without the RGB token, as deployed (fp32, inference mode).

Example:
    python scripts/paper/rgb_cost.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "training", REPO_ROOT / "scripts" / "paper"):
    sys.path.insert(0, str(path))

from ai_classifier.models.multi_task_r2plus1d import MultiTaskR2Plus1D  # noqa: E402
from train_kap_ablation import KAPAblationModel  # noqa: E402


def timed(fn, repeats: int = 200) -> float:
    with torch.inference_mode():
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1000


def main() -> None:
    device = torch.device("cuda")
    rgb = MultiTaskR2Plus1D().to(device).eval()
    clip = torch.randn(1, 3, 16, 112, 112, device=device)
    inputs = [torch.randn(1, 512), torch.randn(1, 32, 68), torch.rand(1, 32, 4), torch.rand(1, 32, 2),
              torch.rand(1, 32, 1), torch.rand(1, 4)]
    inputs = [x.to(device) for x in inputs]
    out = {"rgbParamsM": sum(p.numel() for p in rgb.parameters()) / 1e6,
           "msRgb": timed(lambda: rgb.extract_features(clip))}
    for name, use_rgb in (("msKap", True), ("msKapNoRgb", False)):
        model = KAPAblationModel(gate="none", pooling="three", use_cross=True, use_hitter=True, use_rgb=use_rgb).to(device).eval()
        out[name] = timed(lambda: model(*inputs))
    out["msTotal"] = out["msRgb"] + out["msKap"]
    out["gpu"] = torch.cuda.get_device_name()
    path = REPO_ROOT / "work_dirs/paper_lan4/cost.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
