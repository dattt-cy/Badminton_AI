"""Cut 16-frame hitter crops for ShuttleSet22 strokes into the frames_16 cache format (resumable).

Crop boxes come from the pipeline records of embed_pipeline_ownrgb.py.  Windows are centred on
the refined hit frame when the records carry one (otherwise the labeled frame) and span hit-30..hit+30.  Each clip is a
uint8 array (16, 3, 112, 112) written atomically, and existing clips are skipped on restart.

Example:
    python scripts/paper/cut_ss22_crops.py --ss22-matches 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as VF

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts/training"), str(REPO_ROOT / "scripts/paper")]

from embed_pipeline_ownrgb import OWN_RECORDS, SS22, balanced_train  # noqa: E402
from scripts.data.extract_shuttleset22_records import ForwardCapture  # noqa: E402


def clip_id(match: str, record: dict) -> str:
    return f"{re.sub(r'[^A-Za-z0-9]+', '_', match)}__{record['hit_frame']}_{record['side']}"


def cut(match: str, video: Path, output: Path, source: Path) -> int:
    payload = torch.load(source / f"{re.sub(r'\s+', '_', match)}.pt", map_location="cpu", weights_only=False)
    pending = [r for r in payload["records"] if not (output / f"{clip_id(match, r)}.npy").exists()]
    cap = ForwardCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for record in sorted(pending, key=lambda r: r["hit_frame"]):
        hit, (x1, y1, x2, y2) = record.get("refined_hit_frame", record["hit_frame"]), record["crop_box"]
        wanted = np.rint(np.linspace(max(0, hit - 30), min(total - 1, hit + 30), 16)).astype(int)
        crops = []
        for frame_index in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                break
            crops.append(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        if len(crops) != 16:
            continue
        clip = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().div_(255.0)
        clip = VF.resize(clip, [112, 112], antialias=False).clamp(0, 1).mul_(255.0).round_().byte().numpy()
        target = output / f"{clip_id(match, record)}.npy"
        temporary = target.with_suffix(".tmp.npy")
        np.save(temporary, clip)
        os.replace(temporary, target)
    cap.release()
    return len(pending)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ss22-matches", type=int, default=5)
    parser.add_argument("--source", type=Path, default=OWN_RECORDS)
    parser.add_argument("--output", type=Path, default=Path(r"C:\Users\ADMIN\Downloads\AI_Classifier_cache\ss22_frames_16"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    splits = json.loads((REPO_ROOT / "data/manifests/shuttleset22_splits.json").read_text(encoding="utf-8"))
    ids = {item["video"]: item["id"] for item in json.loads((SS22 / "download_list.json").read_text(encoding="utf-8"))}
    for match in [*splits["val"], *balanced_train(splits)[:args.ss22_matches]]:
        count = cut(match, SS22 / "videos" / f"{ids[match]}.mp4", args.output, args.source)
        print(f"[+] {match}: cut {count} clips", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
