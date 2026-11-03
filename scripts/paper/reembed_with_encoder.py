"""Re-embed pipeline records with a new RGB encoder, without re-running pose or shuttle tracking.

The refined hit frame was not stored, so it is recovered: candidates around the labeled frame go
through the encoder that produced the stored vector, and the frame reproducing it is kept.  The new
encoder then embeds that frame window with the stored crop box.  Files finish atomically and are
skipped on restart.  Records with no exact match keep the closest frame and are counted.

Example:
    python scripts/paper/reembed_with_encoder.py --new-encoder work_dirs/paper_ownrgb_ss22/rgb_encoder_best.pth
"""

from __future__ import annotations

import argparse
import csv
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

from embed_pipeline_ownrgb import CLIP_OFFSETS, MANIFEST, OLD_RECORDS, OWN_RECORDS, SS22, balanced_train  # noqa: E402
from scripts.data.extract_shuttleset22_records import ForwardCapture  # noqa: E402
from train_rgb_scratch import MEAN, STD, RGBEncoder  # noqa: E402

SEARCH_RADIUS = 8
MATCH_TOLERANCE = 0.02


def load_encoder(path: Path, device: torch.device) -> RGBEncoder:
    encoder = RGBEncoder().to(device)
    encoder.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return encoder.eval()


def window(centre: int, total: int) -> np.ndarray:
    return np.rint(np.linspace(max(0, centre - 30), min(total - 1, centre + 30), 16)).astype(int)


def clip_tensor(frames: dict[int, np.ndarray], wanted: np.ndarray, device: torch.device) -> torch.Tensor:
    clip = torch.from_numpy(np.stack([frames[int(f)] for f in wanted])).permute(0, 3, 1, 2).float().div_(255.0)
    clip = VF.resize(clip, [112, 112], antialias=False).clamp(0, 1).permute(1, 0, 2, 3)
    return (clip - MEAN) / STD


@torch.inference_mode()
def embed(encoder: RGBEncoder, clips: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        return encoder.embed(clips.to(device)).float().cpu()


def process(path: Path, video: Path, offset: int, old: RGBEncoder, new: RGBEncoder, output: Path,
            device: torch.device, limit: int | None) -> None:
    target = output / path.name
    if target.exists():
        print(f"[skip] {path.stem}", flush=True)
        return
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = sorted(payload["records"], key=lambda row: row["hit_frame"])[:limit]
    cap = ForwardCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    updated, misses, shifts = [], 0, []
    for record in records:
        hit = record["hit_frame"] + offset
        x1, y1, x2, y2 = record["crop_box"]
        first = max(0, hit - SEARCH_RADIUS - 30)
        last = min(total - 1, hit + SEARCH_RADIUS + 30)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        frames = {}
        for index in range(first, last + 1):
            ok, frame = cap.read()
            if not ok:
                break
            frames[index] = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        centres = [c for c in range(hit - SEARCH_RADIUS, hit + SEARCH_RADIUS + 1)
                   if all(int(f) in frames for f in window(c, total))]
        stored = record.get("refined_hit_frame")
        if stored is not None and stored + offset in centres:
            # The frame is already known, so only confirm it with one forward pass.
            direct = torch.stack([clip_tensor(frames, window(stored + offset, total), device)])
            if float((embed(old, direct, device) - record["emb"].float()).abs().amax()) <= MATCH_TOLERANCE:
                centres = [stored + offset]
        candidates = torch.stack([clip_tensor(frames, window(c, total), device) for c in centres])
        distance = (embed(old, candidates, device) - record["emb"].float()).abs().amax(dim=1)
        best = int(distance.argmin())
        misses += float(distance[best]) > MATCH_TOLERANCE
        shifts.append(centres[best] - hit)
        updated.append({**record, "emb": embed(new, candidates[best:best + 1], device),
                        "refined_hit_frame": centres[best] - offset, "rgb_match_distance": float(distance[best])})
    cap.release()
    temporary = target.with_suffix(".tmp")
    torch.save({**{k: v for k, v in payload.items() if k != "records"}, "records": updated}, temporary)
    os.replace(temporary, target)
    print(f"[+] {path.stem}: {len(updated)} records, no exact match {misses}, "
          f"mean |shift| {np.abs(shifts).mean():.2f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-encoder", type=Path, default=REPO_ROOT / "work_dirs/paper_ownrgb/rgb_encoder_best.pth")
    parser.add_argument("--new-encoder", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=OWN_RECORDS,
                        help="Records whose emb was produced by --old-encoder.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "work_dirs/paper_ownrgb_ss22/pipeline_records")
    parser.add_argument("--ss22-matches", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", default=None, help="Only files whose name contains this text.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N records of each file (smoke test).")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    old, new = load_encoder(args.old_encoder, device), load_encoder(args.new_encoder, device)
    splits = json.loads((REPO_ROOT / "data/manifests/shuttleset22_splits.json").read_text(encoding="utf-8"))
    ids = {item["video"]: item["id"] for item in json.loads((SS22 / "download_list.json").read_text(encoding="utf-8"))}
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        old_videos = {row["match_id"]: Path(row["video_path"]) for row in csv.DictReader(stream)}

    jobs = [(args.source / Path(name).name, old_videos[match], CLIP_OFFSETS[match]) for match, name in OLD_RECORDS.items()]
    for match in [*splits["val"], *balanced_train(splits)[:args.ss22_matches]]:
        jobs.append((args.source / f"{re.sub(r'\s+', '_', match)}.pt", SS22 / "videos" / f"{ids[match]}.mp4", 0))
    for path, video, offset in jobs:
        if args.only is None or args.only in path.name:
            process(path, video, offset, old, new, args.output_dir, device, args.limit)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
