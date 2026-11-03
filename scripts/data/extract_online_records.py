"""Re-extract the online fine-tuning records with a given KAP checkpoint and court reference.

Each record keeps its hit frame, side and labels; every KAP stream is recomputed by the
pipeline the checkpoint selects (a context_window stored in the checkpoint widens the
window), with the other records of the same video as neighbouring hits.

    # long context window, green-mat court
    python scripts/data/extract_online_records.py \
        --checkpoint work_dirs/context_window_pilot/kap_w45_52/kap_best.pth --suffix _ctx45_52
    # standard window, court corners fitted to the painted lines
    python scripts/data/extract_online_records.py \
        --checkpoint work_dirs/kap_cross_v2_liftboost_seed20260926/kap_best.pth --court lines --suffix _lines
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

from scripts.data.reembed_rgb_window import ONLINE  # noqa: E402
from scripts.inference.auto_court_detection import detect_court_corners, detect_court_line_corners  # noqa: E402
from scripts.inference.robust_aligned_video_pipeline import AlignedVideoPipeline  # noqa: E402

WORK = REPO_ROOT / "work_dirs"
CKPTS = REPO_ROOT / "src/ai_classifier/models/checkpoints"
KEYS = ("emb", "s_j", "s_p", "s_s", "s_c", "s_q")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--court", choices=("mat", "lines"), default="mat",
                        help="mat: green-mat quad (the original records); lines: painted court lines.")
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--source-suffix", default="_rgbw30", help="Records whose frames and labels are reused.")
    parser.add_argument("--contact-refine-radius", type=int, default=6,
                        help="Radius used when the online records were extracted.")
    args = parser.parse_args()
    detect = detect_court_line_corners if args.court == "lines" else detect_court_corners
    for video, name in ONLINE:
        source = WORK / name.replace(".pt", f"{args.source_suffix}.pt")
        out = WORK / name.replace(".pt", f"{args.suffix}.pt")
        if out.exists():
            print(f"[skip] {out.name} exists")
            continue
        records = torch.load(source, map_location="cpu", weights_only=False)
        cap = cv2.VideoCapture(str(WORK / video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        corners = detect(str(WORK / video))
        print(f"[*] {video}: court corners {corners.round(1).tolist()}", flush=True)
        pipeline = AlignedVideoPipeline(
            ckpt_path=args.checkpoint, corners=corners,
            orig_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), orig_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            pose_model_path=CKPTS / "yolov8n-pose.pt",
        )
        frames = [int(record["hit_frame"]) for record in records]
        pipeline.set_event_frames(frames)
        pipeline.prepare_event_cache(cap, frames, total, contact_refine_radius=args.contact_refine_radius)
        kept = []
        for record in records:
            feat = pipeline.extract_features(cap, int(record["hit_frame"]), total, hitter_side=record["side"],
                                             contact_refine_radius=args.contact_refine_radius)
            if feat is None:
                continue
            kept.append({**record, **{key: feat[key].detach().cpu() for key in KEYS}})
        cap.release()
        torch.save(kept, out)
        print(f"[+] {out.name}: {len(kept)}/{len(records)} records", flush=True)


if __name__ == "__main__":
    main()
