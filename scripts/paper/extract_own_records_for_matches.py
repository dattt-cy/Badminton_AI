"""Extract labeled KAP records of extra ShuttleSet22 matches directly with the team's own RGB encoder (resumable).

Finished matches are skipped and progress is saved every few rallies, so a stopped run continues.

    python scripts/paper/extract_own_records_for_matches.py --encoder work_dirs/paper_ownrgb_ss22/x13/rgb_encoder_best.pth \
        --output-dir work_dirs/paper_ownrgb_ss22/x13/pipeline_records_french \
        --match CpeAGY7xwtI=Viktor_AXELSEN_Kodai_NARAOKA_YONEX_French_Open_2022_Semifinals
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts/training"), str(REPO_ROOT / "scripts/paper")]

from embed_pipeline_ownrgb import SAVE_EVERY, SS22, WORK, OwnRGBAdapter, save_atomic  # noqa: E402
from scripts.data.extract_shuttleset22_records import CKPTS, KEYS, ForwardCapture, court_corners, labeled_strokes  # noqa: E402
from scripts.inference.auto_court_detection import detect_court_line_corners  # noqa: E402
from scripts.inference.robust_aligned_video_pipeline import AlignedVideoPipeline  # noqa: E402
from train_rgb_scratch import RGBEncoder  # noqa: E402

CONTACT_RADIUS = 6


def extract_match(video: Path, folder: str, encoder: RGBEncoder, output: Path) -> None:
    name = re.sub(r"\s+", "_", folder)
    final, partial = output / f"{name}.pt", output / f"{name}.partial.pt"
    if final.exists():
        print(f"[skip] {name}", flush=True)
        return
    done: dict = torch.load(partial, weights_only=False) if partial.exists() else {}
    cap = ForwardCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    strokes = [s for s in labeled_strokes(SS22, folder, court_corners(SS22, folder)) if 45 <= s["hit_frame"] < total - 60]
    corners = detect_court_line_corners(str(video))
    pipeline = AlignedVideoPipeline(
        ckpt_path=WORK / "kap_cross_v2_liftboost_seed20260926/kap_best.pth", corners=corners,
        orig_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), orig_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        pose_model_path=CKPTS / "yolov8n-pose.pt",
    )
    pipeline.rgb_model = OwnRGBAdapter(encoder, pipeline.rgb_model)
    pipeline.set_event_frames([s["hit_frame"] for s in strokes])
    rallies: dict = {}
    for index, stroke in enumerate(strokes):
        rallies.setdefault((stroke["set"], stroke["rally"]), []).append(index)
    pending = sorted((g for g in rallies.values() if any(i not in done for i in g)), key=lambda g: strokes[g[0]]["hit_frame"])
    print(f"[start] {name}: {len(strokes)} strokes, {len(done)} already done", flush=True)
    for count, group in enumerate(pending, 1):
        pipeline.prepare_event_cache(cap, [strokes[i]["hit_frame"] for i in group], total, contact_refine_radius=CONTACT_RADIUS)
        for index in group:
            feat = pipeline.extract_features(cap, strokes[index]["hit_frame"], total, hitter_side=strokes[index]["side"],
                                             contact_refine_radius=CONTACT_RADIUS)
            if feat is not None:
                done[index] = {**{key: feat[key].detach().cpu() for key in KEYS},
                               "crop_box": tuple(int(v) for v in feat["crop_box"]), "refined_hit_frame": int(feat["hit_frame"])}
        if count % SAVE_EVERY == 0:
            save_atomic(done, partial)
            print(f"  {name}: {len(done)}/{len(strokes)}", flush=True)
    cap.release()
    kept = [{**strokes[i], "video": folder, **done[i]} for i in sorted(done)]
    save_atomic({"court_corners": corners.tolist(), "video": folder, "records": kept}, final)
    partial.unlink(missing_ok=True)
    print(f"[+] {name}: {len(kept)}/{len(strokes)} records", flush=True)
    del pipeline
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match", action="append", required=True, help="youtube_id=folder")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder = RGBEncoder().cuda()
    encoder.load_state_dict(torch.load(args.encoder, weights_only=False)["model"])
    encoder.eval()
    for spec in args.match:
        youtube_id, folder = spec.split("=", 1)
        extract_match(SS22 / "videos" / f"{youtube_id}.mp4", folder, encoder, args.output_dir)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
