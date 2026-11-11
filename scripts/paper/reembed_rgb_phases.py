"""Give pipeline records the own-RGB features per contact phase, keeping pose, shuttle and court (resumable).

Each emb becomes [context, before, contact, after]. Clips come from the crop cache when present,
otherwise from the video using the stored crop box and refined frame. The context vector must
reproduce the stored emb, which checks that the clip is the one the record was built from.

    python scripts/paper/reembed_rgb_phases.py --source work_dirs/paper_ownrgb_ss22/x13/pipeline_records \
        --output-dir work_dirs/paper_ownrgb_ss22/x13/pipeline_records_phases
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

from embed_pipeline_ownrgb import CLIP_OFFSETS, MANIFEST, OLD_RECORDS, SS22, balanced_train  # noqa: E402
from export_rgb_phases import PHASES  # noqa: E402
from scripts.data.extract_shuttleset22_records import ForwardCapture  # noqa: E402
from train_rgb_scratch import MEAN, STD, RGBEncoder  # noqa: E402

CROPS = Path(r"C:\Users\ADMIN\Downloads\AI_Classifier_cache\ss22_frames_16")
TOLERANCE = 0.02
FRENCH = {"Viktor_AXELSEN_Kodai_NARAOKA_YONEX_French_Open_2022_Semifinals": "CpeAGY7xwtI",
          "Viktor_AXELSEN_Rasmus_GEMKE_French_Open_2022_Finals": "FYQCIkw32jw"}


def clip_from_video(cap, record: dict, offset: int, total: int) -> torch.Tensor | None:
    hit = record.get("refined_hit_frame", record["hit_frame"]) + offset
    x1, y1, x2, y2 = record["crop_box"]
    crops = []
    for frame_index in np.rint(np.linspace(max(0, hit - 30), min(total - 1, hit + 30), 16)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            return None
        crops.append(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    clip = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().div_(255.0)
    return VF.resize(clip, [112, 112], antialias=False).clamp(0, 1)


def process(source: Path, output: Path, encoder: RGBEncoder, steps: list, device, *, match: str | None, video: Path | None, offset: int) -> None:
    target = output / source.name
    if target.exists():
        print(f"[skip] {source.stem}", flush=True)
        return
    payload = torch.load(source, map_location="cpu", weights_only=False)
    cap, total = None, 0
    if video is not None:
        cap = ForwardCapture(video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    records, far, skipped, batch, kept = [], 0, 0, [], []

    def flush() -> None:
        nonlocal far
        if not batch:
            return
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16):
            steps.clear()
            context = encoder.embed(torch.stack(batch).to(device))
            pooled = [steps[0][:, :, phase].mean(dim=-1) for phase in PHASES]
            emb = torch.stack([context, *pooled], dim=1).float().cpu()
        for index, record in enumerate(kept):
            far += float((emb[index, 0] - record["emb"].float().view(-1)).abs().amax()) > TOLERANCE
            records.append({**record, "emb": emb[index:index + 1]})
        batch.clear(); kept.clear()

    for record in sorted(payload["records"], key=lambda row: row["hit_frame"]):
        clip = None
        if match is not None:
            name = f"{re.sub(r'[^A-Za-z0-9]+', '_', match)}__{record['hit_frame']}_{record['side']}.npy"
            if (CROPS / name).exists():
                clip = torch.from_numpy(np.load(CROPS / name)).float().div_(255.0)
        elif cap is not None:
            clip = clip_from_video(cap, record, offset, total)
        if clip is None:
            skipped += 1
            continue
        batch.append((clip.permute(1, 0, 2, 3).clamp(0, 1) - MEAN) / STD)
        kept.append(record)
        if len(batch) == 64:
            flush()
    flush()
    if cap is not None:
        cap.release()
    temporary = target.with_suffix(".tmp")
    torch.save({**{k: v for k, v in payload.items() if k != "records"}, "records": records}, temporary)
    os.replace(temporary, target)
    print(f"[+] {source.stem[:48]}: {len(records)} records, context differs from stored emb {far}, no clip {skipped}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, default=REPO_ROOT / "work_dirs/paper_ownrgb_ss22/x13/rgb_encoder_best.pth")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--french-source", type=Path, default=REPO_ROOT / "work_dirs/paper_ownrgb_ss22/x13/pipeline_records_french")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    encoder = RGBEncoder().to(device)
    encoder.load_state_dict(torch.load(args.encoder, map_location=device, weights_only=False)["model"])
    encoder.eval()
    steps: list[torch.Tensor] = []
    encoder.net.temporal_tcn2.register_forward_hook(lambda _m, _i, out: steps.append(out))
    splits = json.loads((REPO_ROOT / "data/manifests/shuttleset22_splits.json").read_text(encoding="utf-8"))
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        old_videos = {row["match_id"]: Path(row["video_path"]) for row in csv.DictReader(stream)}
    for match, name in OLD_RECORDS.items():
        process(args.source / Path(name).name, args.output_dir, encoder, steps, device, match=None, video=old_videos[match], offset=CLIP_OFFSETS[match])
    for match in [*splits["val"], *balanced_train(splits)]:
        process(args.source / f"{re.sub(r'\s+', '_', match)}.pt", args.output_dir, encoder, steps, device, match=match, video=None, offset=0)
    for stem, youtube_id in FRENCH.items():
        process(args.french_source / f"{stem}.pt", args.output_dir, encoder, steps, device, match=None, video=SS22 / "videos" / f"{youtube_id}.mp4", offset=0)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
