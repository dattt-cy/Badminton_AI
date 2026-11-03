"""Replace ``emb`` of the pipeline records with the team's own RGB encoder (resumable).

Pose, shuttle, court and contact streams of each record are kept.  Frames and crop
boxes come from AlignedVideoPipeline.extract_features, so the crops are exactly the
ones the pipeline produces.  Progress is written atomically every few rallies to
<output>/<name>.partial.pt, and finished files are skipped on restart.

Order: match 41 and the ShuttleSet22 validation matches first (an early sanity check),
then the old online matches, ShuttleSet22 train, and ShuttleSet22 test last.

Example:
    python scripts/paper/embed_pipeline_ownrgb.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from itertools import zip_longest
from pathlib import Path

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts/training"), str(REPO_ROOT / "scripts/paper")]

from finetune_kap_online_multi_matches import OFFSETS  # noqa: E402
from scripts.data.extract_shuttleset22_records import CKPTS, ForwardCapture  # noqa: E402
from scripts.inference.auto_court_detection import detect_court_line_corners  # noqa: E402
from scripts.inference.robust_aligned_video_pipeline import _MEAN, _STD, AlignedVideoPipeline  # noqa: E402
from train_rgb_scratch import MEAN, STD, RGBEncoder  # noqa: E402

WORK = REPO_ROOT / "work_dirs"
MANIFEST = REPO_ROOT / "data/manifests/shuttleset_rgb.csv"
# The match 41 validation clip starts 450 s into the source; checked against the manifest hit frames.
CLIP_OFFSETS = {**OFFSETS, "41": 13500}
SS22 = Path(r"C:\Users\ADMIN\Downloads\shuttleset22")
OLD_RECORDS = {"41": "match41_5min_aligned_features_lines.pt", "13": "match13_5min_features_lines.pt",
               **{str(m): f"online_train_match{m}_features_lines.pt" for m in range(14, 18)}}
OWN_RECORDS = WORK / "paper_ownrgb/pipeline_records"
SAVE_EVERY = 10
VERIFY_FIRST = 5


def balanced_train(splits: dict) -> list[str]:
    """ShuttleSet22 train matches alternating women and men, so the first N stay balanced."""
    women = {item["video"]: item["women"] for item in json.loads((SS22 / "download_list.json").read_text(encoding="utf-8"))}
    groups = [[m for m in splits["train"] if women[m]], [m for m in splits["train"] if not women[m]]]
    ordered = [m for pair in zip_longest(*groups) for m in pair if m is not None]
    return ordered


class OwnRGBAdapter:
    """Stands in for pipeline.rgb_model: undoes the pipeline normalisation and embeds with the own encoder."""

    def __init__(self, encoder: RGBEncoder, reference) -> None:
        self.encoder, self.reference = encoder, reference
        self.verify_left, self.stored = 0, None
        self.diffs: list[float] = []

    def extract_features(self, clip: torch.Tensor) -> torch.Tensor:
        if self.verify_left > 0 and self.stored is not None:
            self.diffs.append((self.reference.extract_features(clip).cpu() - self.stored).abs().max().item())
            self.verify_left -= 1
        raw = clip * _STD.view(1, 3, 1, 1, 1) + _MEAN.view(1, 3, 1, 1, 1)
        own = (raw.clamp(0, 1) - MEAN.view(1, 3, 1, 1, 1).to(raw)) / STD.view(1, 3, 1, 1, 1).to(raw)
        with torch.autocast("cuda", dtype=torch.float16):
            return self.encoder.embed(own).float()


def save_atomic(payload, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def embed_file(name: str, records: list[dict], video: Path, offset: int, encoder: RGBEncoder, output: Path,
               extra: dict) -> None:
    final, partial = output / f"{name}.pt", output / f"{name}.partial.pt"
    if final.exists():
        print(f"[skip] {name}", flush=True)
        return
    done: dict = torch.load(partial, weights_only=False) if partial.exists() else {}
    cap = ForwardCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pipeline = AlignedVideoPipeline(
        ckpt_path=WORK / "kap_cross_v2_liftboost_seed20260926/kap_best.pth", corners=detect_court_line_corners(str(video)),
        orig_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), orig_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        pose_model_path=CKPTS / "yolov8n-pose.pt",
    )
    adapter = OwnRGBAdapter(encoder, pipeline.rgb_model)
    pipeline.rgb_model = adapter
    groups: dict = {}
    for index, record in enumerate(records):
        key = (record.get("set"), record.get("rally")) if "rally" in record else index // SAVE_EVERY
        groups.setdefault(key, []).append(index)
    pending = sorted((g for g in groups.values() if any(i not in done for i in g)),
                     key=lambda g: records[g[0]]["hit_frame"])
    print(f"[start] {name}: {len(records)} records, {len(done)} already done", flush=True)
    for count, group in enumerate(pending, 1):
        if "rally" in records[group[0]]:
            pipeline.prepare_event_cache(cap, [records[i]["hit_frame"] + offset for i in group], total, contact_refine_radius=6)
        for index in group:
            record = records[index]
            adapter.stored, adapter.verify_left = record["emb"], int(len(adapter.diffs) < VERIFY_FIRST)
            feat = pipeline.extract_features(cap, record["hit_frame"] + offset, total, hitter_side=record["side"],
                                             contact_refine_radius=6)
            if feat is not None:
                done[index] = {"emb": feat["emb"].detach().cpu(), "crop_box": tuple(int(v) for v in feat["crop_box"]),
                               "refined_hit_frame": int(feat["hit_frame"]) - offset}
        if count % SAVE_EVERY == 0:
            save_atomic(done, partial)
            print(f"  {name}: {len(done)}/{len(records)}", flush=True)
    cap.release()
    kept = [{**records[i], **done[i]} for i in sorted(done)]
    save_atomic({**extra, "records": kept}, final)
    partial.unlink(missing_ok=True)
    largest = max(adapter.diffs, default=float("nan"))
    print(f"[+] {name}: {len(kept)}/{len(records)} records, R(2+1)D reproduction max diff {largest:.3f}", flush=True)
    del pipeline
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=WORK / "paper_ownrgb/pipeline_records")
    parser.add_argument("--encoder", type=Path, default=WORK / "paper_ownrgb/rgb_encoder_best.pth")
    parser.add_argument("--groups", nargs="+", default=["val", "old", "train", "test"],
                        choices=("val", "old", "train", "test"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder = RGBEncoder().cuda()
    encoder.load_state_dict(torch.load(args.encoder, weights_only=False)["model"])
    encoder.eval()
    splits = json.loads((REPO_ROOT / "data/manifests/shuttleset22_splits.json").read_text(encoding="utf-8"))
    ids = {item["video"]: item["id"] for item in json.loads((SS22 / "download_list.json").read_text(encoding="utf-8"))}

    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        old_videos = {row["match_id"]: Path(row["video_path"]) for row in csv.DictReader(stream)}

    def old(match: str) -> None:
        path = WORK / OLD_RECORDS[match]
        embed_file(path.stem, torch.load(path, map_location="cpu", weights_only=False), old_videos[match],
                   CLIP_OFFSETS[match], encoder, args.output_dir, {})

    def new(match: str) -> None:
        payload = torch.load(WORK / "shuttleset22_records" / f"{match}.pt", map_location="cpu", weights_only=False)
        embed_file(re.sub(r"\s+", "_", match), payload["records"], SS22 / "videos" / f"{ids[match]}.mp4", 0,
                   encoder, args.output_dir, {"court_corners": payload["court_corners"], "video": match})

    for group in args.groups:
        if group == "val":
            old("41")
            for match in splits["val"]:
                new(match)
        elif group == "old":
            for match in ("13", "14", "15", "16", "17"):
                old(match)
        else:
            for match in (balanced_train(splits) if group == "train" else splits[group]):
                new(match)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
