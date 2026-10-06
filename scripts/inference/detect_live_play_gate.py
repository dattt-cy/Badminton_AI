"""Detect full-court LIVE rally segments using Rally-State Gate (R(2+1)D + Scene MLP).

Replaces the legacy color/two-player heuristic with the trained RallyGateMLP checkpoint
(work_dirs/rally_gate_best.pth), achieving >99% rally recall while rejecting non-play
intervals (walking, towel, picking shuttle, replay graphics).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18
from torchvision.transforms import functional as VF

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.localization.live_gate import GateSample, samples_to_segments
from scripts.training.train_rally_gate import RallyGateMLP
from scripts.data.extract_rally_gate_features import extract_scene_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MEAN = torch.tensor([0.43216, 0.394666, 0.37645], device=DEVICE).view(1, 3, 1, 1, 1)
_STD = torch.tensor([0.22803, 0.22145, 0.216989], device=DEVICE).view(1, 3, 1, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Broadcast match video or clip")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON with live segments")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Evaluation frequency in frames per second")
    parser.add_argument("--gate-checkpoint", type=Path, default=Path("work_dirs/rally_gate_best.pth"))
    parser.add_argument("--backbone-checkpoint", type=Path, default=Path("work_dirs/r2plus1d18_hit_full/best.pth"))
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (defaults to checkpoint value)")
    parser.add_argument("--batch-size", type=int, default=16, help="GPU forward batch size")
    return parser.parse_args()


def load_models(gate_ckpt_path: Path, backbone_ckpt_path: Path) -> tuple[nn.Module, nn.Module, float]:
    # 1. Visual Backbone
    backbone = r2plus1d_18(weights=None)
    backbone.fc = nn.Linear(512, 3)
    if backbone_ckpt_path.exists():
        ckpt_bb = torch.load(backbone_ckpt_path, map_location="cpu", weights_only=False)
        backbone.load_state_dict(ckpt_bb["model"])
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten()).to(DEVICE)
    feature_extractor.eval()

    # 2. Gate MLP
    ckpt_gate = torch.load(gate_ckpt_path, map_location="cpu", weights_only=False)
    in_dim = int(ckpt_gate.get("input_dim", 516))
    threshold = float(ckpt_gate.get("threshold", 0.60))

    gate_model = RallyGateMLP(in_features=in_dim).to(DEVICE)
    gate_model.load_state_dict(ckpt_gate["model"])
    gate_model.eval()

    return feature_extractor, gate_model, threshold


def main() -> None:
    args = parse_args()
    if args.sample_fps <= 0:
        raise ValueError("sample-fps must be positive")

    feature_extractor, gate_model, default_thresh = load_models(args.gate_checkpoint, args.backbone_checkpoint)
    thresh = args.threshold if args.threshold is not None else default_thresh
    print(f"[*] Loaded Rally-State Gate (threshold={thresh:.2f}, device={DEVICE})")

    cap = cv2.VideoCapture(str(args.video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if total_frames <= 0 or fps <= 0:
        cap.release()
        raise ValueError(f"Cannot inspect video: {args.video}")

    step = max(1, round(fps / args.sample_fps))
    print(f"[*] Scanning video {args.video.name}: {total_frames} frames ({total_frames/fps:.1f}s), step={step} (~{total_frames//step} samples)...")

    sample_centers = list(range(16, total_frames - 16, step))
    samples: list[GateSample] = []
    t_start = time.time()

    # Process in batches
    for b_start in range(0, len(sample_centers), args.batch_size):
        batch_centers = sample_centers[b_start : b_start + args.batch_size]
        clips = []
        scenes = []

        for center in batch_centers:
            start_f = max(0, center - 16)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            frames = []
            for _ in range(32):
                ok, f = cap.read()
                if not ok: break
                frames.append(f)

            if len(frames) < 16:
                # Fallback zero clip
                clip = torch.zeros(3, 16, 112, 112)
                scene = np.zeros(4, dtype=np.float32)
            else:
                scene = extract_scene_metrics(frames)
                idx = np.linspace(0, len(frames) - 1, 16, dtype=int)
                rgb_arr = np.stack([cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in idx])
                t = torch.from_numpy(rgb_arr).permute(0, 3, 1, 2).float().div_(255.0)
                t = VF.resize(t, [128, 171], antialias=False)
                t = VF.center_crop(t, [112, 112])
                clip = t.permute(1, 0, 2, 3)

            clips.append(clip)
            scenes.append(scene)

        batch_tensor = torch.stack(clips).to(DEVICE)
        batch_tensor = (batch_tensor - _MEAN) / _STD
        with torch.inference_mode():
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
                embs = feature_extractor(batch_tensor)
            scenes_tensor = torch.tensor(np.stack(scenes), dtype=torch.float32, device=DEVICE)
            feats = torch.cat([embs.float(), scenes_tensor], dim=1)
            logits = gate_model(feats)
            probs = logits.sigmoid().cpu().numpy()

        for center, prob in zip(batch_centers, probs):
            is_live = bool(prob >= thresh)
            samples.append(GateSample(frame=center, live=is_live, score=float(prob)))

        if len(samples) % 200 < args.batch_size:
            print(f"  [gate-progress] {len(samples)}/{len(sample_centers)} ({100*len(samples)/len(sample_centers):.1f}%)", flush=True)

    cap.release()

    # Convert samples to continuous segments
    segments = samples_to_segments(samples, total_frames=total_frames, max_gap_frames=step * 3)
    live_frames = sum(int(item["end_frame"]) - int(item["start_frame"]) + 1 for item in segments)
    live_fraction = live_frames / max(total_frames, 1)

    result = {
        "video": str(args.video.resolve()),
        "frames": total_frames,
        "fps": fps,
        "sample_fps": args.sample_fps,
        "gate_version": "v2_rally_gate_r2plus1d_scene",
        "threshold": thresh,
        "live_segments": segments,
        "live_frame_fraction": live_fraction,
        "live_duration_seconds": live_frames / fps,
        "total_duration_seconds": total_frames / fps,
        "scan_time_seconds": round(time.time() - t_start, 2),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n========================================================")
    print(f" [DONE] Rally-State Gate Segmentation Complete: {args.output}")
    print(f"   Detected Live Segments: {len(segments)}")
    print(f"   Live Duration: {live_frames / fps:.1f}s / {total_frames / fps:.1f}s ({live_fraction*100:.1f}%)")
    print(f"   Scan Time: {time.time() - t_start:.1f}s")
    for idx, seg in enumerate(segments, 1):
        dur = (int(seg["end_frame"]) - int(seg["start_frame"]) + 1) / fps
        print(f"     Segment {idx:>2}: frames {seg['start_frame']:>5} -> {seg['end_frame']:>5} ({dur:>4.1f}s) | score={seg['mean_score']:.3f}")
    print("========================================================")


if __name__ == "__main__":
    main()
