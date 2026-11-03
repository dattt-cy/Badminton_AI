"""Export own-RGB features per contact phase for every KAP event (resumable).

Clips hold 16 crops spanning hit-30..hit+30, so steps 0-5 precede contact, 6-9 surround it
and 10-15 follow it. Each event gets [context, before, contact, after], each 224-d; finished
chunks are skipped on restart.

    python scripts/paper/export_rgb_phases.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts/training"), str(REPO_ROOT / "scripts/paper")]

from train_rgb_scratch import MEAN, STD, RGBEncoder  # noqa: E402

CHUNK = 3200
PHASES = (slice(0, 6), slice(6, 10), slice(10, 16))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, default=REPO_ROOT / "work_dirs/paper_ownrgb_ss22/x13/rgb_encoder_best.pth")
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "work_dirs/paper_ownrgb_ss22/x13/features_w15_30_ownrgb.npz")
    parser.add_argument("--cache", type=Path, default=Path(r"C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_v3_full\frames_16"))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "work_dirs/paper_rgb_phases/features_rgb_phases.npz")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chunks = args.output.parent / "chunks"
    chunks.mkdir(exist_ok=True)
    raw = np.load(args.source, allow_pickle=True)
    ids = [str(x) for x in raw["sample_id"]]
    device = torch.device("cuda")
    model = RGBEncoder().to(device)
    model.load_state_dict(torch.load(args.encoder, weights_only=False)["model"])
    model.eval()
    steps: list[torch.Tensor] = []
    model.net.temporal_tcn2.register_forward_hook(lambda _m, _i, out: steps.append(out))

    for start in range(0, len(ids), CHUNK):
        path = chunks / f"{start:06d}.npy"
        if path.exists():
            continue
        rows = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for first in range(start, min(start + CHUNK, len(ids)), 64):
                batch = []
                for sample in ids[first:first + 64]:
                    file = args.cache / f"{sample}.npy"
                    clip = torch.from_numpy(np.load(file)).float().div_(255.0) if file.exists() else torch.zeros(16, 3, 112, 112)
                    batch.append((clip.permute(1, 0, 2, 3).clamp(0, 1) - MEAN) / STD)
                steps.clear()
                context = model.embed(torch.stack(batch).to(device))
                sequence = steps[0]
                pooled = [sequence[:, :, phase].mean(dim=-1) for phase in PHASES]
                rows.append(torch.stack([context, *pooled], dim=1).float().cpu())
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, torch.cat(rows).numpy().astype(np.float32))
        temporary.replace(path)
        print(f"{min(start + CHUNK, len(ids))}/{len(ids)}", flush=True)

    rgb = np.concatenate([np.load(chunks / f"{start:06d}.npy") for start in range(0, len(ids), CHUNK)])
    difference = np.abs(rgb[:, 0] - raw["rgb"]).max()
    print(f"context vs stored embedding, max difference {difference:.4f}", flush=True)
    np.savez(args.output, **{**{key: raw[key] for key in raw.files}, "rgb": rgb})
    shutil.rmtree(chunks)
    print(f"[export] {rgb.shape}", flush=True)


if __name__ == "__main__":
    main()
