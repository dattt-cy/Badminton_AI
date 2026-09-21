"""Scan a full match video and produce per-frame hit scores using R(2+1)D-18.

This script replaces the Random Forest motion-grid scanner with the R(2+1)D
model. It performs dense sliding-window inference (configurable stride) and
outputs a list of HitEvent objects compatible with the existing NMS and
event-matching utilities.

Usage example:
    python scripts/inference/scan_video_hit_rgb.py \\
        --checkpoint work_dirs/r2plus1d18_hit_full/best.pth \\
        --video "C:/Users/ADMIN/Downloads/shuttleSEt/Test/39 ....mp4" \\
        --output work_dirs/scan_match39.json \\
        --stride 4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.video import r2plus1d_18
from torchvision.transforms import functional as VF

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.localization import HitEvent, temporal_nms  # noqa: E402

CLASSES = ["no_hit", "upper_hit", "lower_hit"]

_MEAN = torch.tensor([0.43216, 0.394666, 0.37645])
_STD = torch.tensor([0.22803, 0.22145, 0.216989])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path) -> tuple[nn.Module, dict]:
    """Return (model, checkpoint_meta)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = r2plus1d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(ckpt["model"])
    return model, ckpt


# ---------------------------------------------------------------------------
# Frame decoding helpers
# ---------------------------------------------------------------------------

def _preprocess_frames(frames: list[np.ndarray], n_frames: int) -> torch.Tensor:
    """Convert a list of RGB uint8 HxWxC frames into a C,T,H,W float tensor."""
    while len(frames) < n_frames:
        frames.append(frames[-1])
    arr = np.stack(frames[:n_frames])          # T,H,W,C uint8
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)  # T,C,H,W
    tensor = VF.resize(tensor, [128, 171], antialias=False)
    tensor = VF.center_crop(tensor, [112, 112])
    mean = _MEAN.view(1, 3, 1, 1)
    std = _STD.view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.permute(1, 0, 2, 3)  # C,T,H,W


# ---------------------------------------------------------------------------
# Dense scanning
# ---------------------------------------------------------------------------

def scan_video(
    video_path: Path,
    model: nn.Module,
    n_frames: int,
    *,
    stride: int,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> list[HitEvent]:
    """Produce one HitEvent per stride-th frame across the entire video.

    The score of each event is P(upper_hit) + P(lower_hit) — the total
    probability of "any hit". The side is whichever of upper/lower is higher.

    Args:
        video_path: Path to the broadcast match video.
        model: Loaded R(2+1)D-18 classifier.
        n_frames: Number of frames in each input clip (from checkpoint).
        stride: Evaluate every `stride`-th frame. Lower → denser, slower.
        batch_size: Number of clips processed per GPU forward pass.
        device: Torch device for inference.
        use_amp: Whether to use float16 autocast.
        max_frames: If set, stop after this many frames (for debugging).
        start_frame: Skip this many frames at the beginning.
    """
    half = n_frames // 2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    effective_max = min(total_frames, max_frames) if max_frames else total_frames
    print(
        f"  [scan] {video_path.name}  frames={effective_max}  stride={stride}  "
        f"~{effective_max // stride} evals",
        flush=True,
    )

    model.eval()
    events: list[HitEvent] = []
    frame_buffer: deque[np.ndarray] = deque(maxlen=n_frames)
    pending_clips: list[torch.Tensor] = []
    pending_centers: list[int] = []
    first_center = start_frame + half
    expected_evals = max(0, (effective_max - half - first_center + stride - 1) // stride)

    def flush_batch() -> None:
        if not pending_clips:
            return
        batch_tensor = torch.stack(pending_clips).to(device)  # B,C,T,H,W
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(batch_tensor)
            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        for center, prob in zip(pending_centers, probs):
            upper = float(prob[1])
            lower = float(prob[2])
            hit_score = upper + lower
            side = "upper" if upper >= lower else "lower"
            events.append(HitEvent(frame=center, side=side, score=hit_score))
        pending_clips.clear()
        pending_centers.clear()

    # Decode once from left to right. The old implementation performed a
    # random seek and decoded 16 frames for every candidate center.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    while frame_index < effective_max:
        ok, frame = cap.read()
        if not ok:
            break
        frame_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frame_buffer) == n_frames:
            window_start = frame_index - n_frames + 1
            center = window_start + half
            if (
                center >= first_center
                and center < effective_max - half
                and (center - first_center) % stride == 0
            ):
                pending_clips.append(_preprocess_frames(list(frame_buffer), n_frames))
                pending_centers.append(center)
                if len(pending_clips) >= batch_size:
                    flush_batch()
                    if len(events) % 2000 < batch_size:
                        print(
                            f"  [scan-progress] {len(events)}/{expected_evals} "
                            f"({100.0 * len(events) / max(expected_evals, 1):.1f}%)",
                            flush=True,
                        )
        frame_index += 1
    flush_batch()
    cap.release()
    return events


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="R(2+1)D checkpoint from train_shuttleset_hit_rgb_full.py")
    parser.add_argument("--video", type=Path, required=True,
                        help="Full match video to scan.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON with raw events (before NMS).")
    parser.add_argument("--stride", type=int, default=4,
                        help="Evaluate every N-th frame. Lower = denser (default 4).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Clips per GPU batch.")
    parser.add_argument("--nms-radius", type=int, default=15,
                        help="Temporal NMS radius in frames.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Hit score threshold for NMS.")
    parser.add_argument("--max-frames", type=int,
                        help="Process only first N frames (debugging).")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Skip this many frames at the beginning.")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    model, ckpt_meta = load_model(args.checkpoint)
    model.to(device)
    n_frames = int(ckpt_meta.get("frames", 16))
    print(f"[INFO] checkpoint epoch={ckpt_meta.get('epoch', '?')}  frames={n_frames}  device={device}")

    events = scan_video(
        args.video, model, n_frames,
        stride=args.stride,
        batch_size=args.batch_size,
        device=device,
        use_amp=use_amp,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
    )

    nms_events = temporal_nms(events, threshold=args.threshold, radius=args.nms_radius)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "video": str(args.video.resolve()),
        "stride": args.stride,
        "nms_radius": args.nms_radius,
        "threshold": args.threshold,
        "frames_scored": len(events),
        "candidate_count": len(nms_events),
        "events": [{"frame": e.frame, "side": e.side, "score": round(e.score, 6)} for e in nms_events],
        "raw_events": [{"frame": e.frame, "side": e.side, "score": round(e.score, 6)} for e in events],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] raw={len(events)} scored frames  nms_events={len(nms_events)}  output={args.output}")


if __name__ == "__main__":
    main()
