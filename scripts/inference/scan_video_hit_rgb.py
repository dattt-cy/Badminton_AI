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
import queue
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Callable

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
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_classifier.localization import HitEvent, two_stage_temporal_nms  # noqa: E402
from ai_classifier.models import BadmintonHitNet, BadmintonHitNetV2  # noqa: E402

CLASSES = ["no_hit", "upper_hit", "lower_hit"]

_MEAN = torch.tensor([0.43216, 0.394666, 0.37645])
_STD = torch.tensor([0.22803, 0.22145, 0.216989])
# Prepared batches the decode thread may run ahead of the GPU (8 clips ≈ 6 MB each).
SCAN_PREFETCH_BATCHES = 6
# Decoded frames the decode thread may run ahead of preparation (720p ≈ 2.8 MB each).
SCAN_PREFETCH_FRAMES = 32


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path) -> tuple[nn.Module, dict]:
    """Return (model, checkpoint_meta).
    
    Supports BadmintonHitNetV2, BadmintonHitNet, and legacy R(2+1)D-18.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    is_hitnet_v2 = (
        ckpt.get("arch") == "badminton_hitnet_v2"
        or "spatial_stage1.0.conv.2.fc.1.weight" in state
    )
    is_hitnet = (
        is_hitnet_v2
        or ckpt.get("arch") == "badminton_hitnet"
        or "spatial_stem.0.weight" in state
        or "spatial_stem.0.conv.weight" in state
    )
    if is_hitnet_v2:
        n_frames = int(ckpt.get("frames", 16))
        classes = ckpt.get("classes", CLASSES)
        model = BadmintonHitNetV2(num_classes=len(classes), in_frames=n_frames)
        model.load_state_dict(state)
    elif is_hitnet:
        n_frames = int(ckpt.get("frames", 16))
        classes = ckpt.get("classes", CLASSES)
        model = BadmintonHitNet(num_classes=len(classes), in_frames=n_frames)
        model.load_state_dict(state)
    else:
        model = r2plus1d_18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        model.load_state_dict(state)
    return model, ckpt


# ---------------------------------------------------------------------------
# Frame decoding helpers
# ---------------------------------------------------------------------------

def _preprocess_frame(frame_rgb: np.ndarray) -> torch.Tensor:
    """Normalise one RGB uint8 128x171x3 frame into a C,H,W float tensor 112x112.

    Every step is element-wise, so a clip stacked from these frames equals a clip
    normalised as a whole; with stride 4 each frame is now normalised once instead
    of four times.
    """
    crop = frame_rgb[8:120, 30:142, :]  # Center crop (112, 112, 3)
    tensor = torch.from_numpy(crop).permute(2, 0, 1).float().div_(255.0)  # C,H,W
    return (tensor - _MEAN.view(3, 1, 1)) / _STD.view(3, 1, 1)


def _put_unless_stopped(target: queue.Queue, item, stop: threading.Event) -> bool:
    while not stop.is_set():
        try:
            target.put(item, timeout=0.5)
            return True
        except queue.Full:
            continue
    return False


def _get_unless_stopped(source: queue.Queue, stop: threading.Event):
    while not stop.is_set():
        try:
            return source.get(timeout=0.5)
        except queue.Empty:
            continue
    return None


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
    gate_path: Path | None = None,
    gate_padding: int = 30,
    frame_callback: Callable[[int, np.ndarray], None] | None = None,
    on_scored: Callable[[list[HitEvent], int], bool | None] | None = None,
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
        on_scored: Called after every scored batch with all events so far (in frame
            order; the list keeps growing) and the latest scored centre frame. Returning
            True stops the scan there.
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
    gate_segments = None
    if gate_path is not None and Path(gate_path).exists():
        from ai_classifier.localization.live_gate import frame_is_live
        gate_data = json.loads(Path(gate_path).read_text(encoding="utf-8"))
        gate_segments = gate_data.get("live_segments", [])
        print(
            f"  [scan] Guided by Rally Gate: {len(gate_segments)} live segments (padding={gate_padding}f)",
            flush=True,
        )

    model.eval()
    events: list[HitEvent] = []
    first_center = start_frame + half

    # Three stages overlap: a decode thread, a preparation thread (resize, normalise,
    # batch) and the GPU on this thread. OpenCV and torch release the GIL, so decoding
    # becomes the only bound. Batches are formed exactly as in a single-threaded loop,
    # so scores are unchanged.
    decoded: queue.Queue = queue.Queue(maxsize=SCAN_PREFETCH_FRAMES)
    batches: queue.Queue = queue.Queue(maxsize=SCAN_PREFETCH_BATCHES)
    stop = threading.Event()
    done = object()

    def decode() -> None:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_index = start_frame
            while frame_index < effective_max and not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_callback is not None:
                    # Lets later stages reuse this single sequential decode instead of seeking.
                    frame_callback(frame_index, frame)
                if not _put_unless_stopped(decoded, (frame_index, frame), stop):
                    return
                frame_index += 1
        except BaseException as exc:  # surfaced in the consumer thread
            _put_unless_stopped(decoded, exc, stop)
        finally:
            _put_unless_stopped(decoded, done, stop)

    def prepare() -> None:
        frame_buffer: deque[torch.Tensor] = deque(maxlen=n_frames)
        pending_clips: list[torch.Tensor] = []
        pending_centers: list[int] = []
        try:
            while True:
                item = _get_unless_stopped(decoded, stop)
                if item is None or item is done:
                    break
                if isinstance(item, BaseException):
                    raise item
                frame_index, frame = item
                frame_small = cv2.resize(frame, (171, 128), interpolation=cv2.INTER_LINEAR)
                frame_buffer.append(_preprocess_frame(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)))
                if len(frame_buffer) == n_frames:
                    window_start = frame_index - n_frames + 1
                    center = window_start + half
                    if (
                        center >= first_center
                        and center < effective_max - half
                        and (center - first_center) % stride == 0
                        and (gate_segments is None or frame_is_live(center, gate_segments, padding=gate_padding))
                    ):
                        pending_clips.append(torch.stack(list(frame_buffer), dim=1))  # C,T,H,W
                        pending_centers.append(center)
                        if len(pending_clips) >= batch_size:
                            if not _put_unless_stopped(batches, (torch.stack(pending_clips), list(pending_centers)), stop):
                                return
                            pending_clips.clear()
                            pending_centers.clear()
            if pending_clips:
                _put_unless_stopped(batches, (torch.stack(pending_clips), list(pending_centers)), stop)
        except BaseException as exc:  # surfaced in the consumer thread
            _put_unless_stopped(batches, exc, stop)
        finally:
            _put_unless_stopped(batches, done, stop)

    workers = [
        threading.Thread(target=decode, name="scan-decode", daemon=True),
        threading.Thread(target=prepare, name="scan-prepare", daemon=True),
    ]
    # Torch's intra-op pool split every small per-frame op across all cores and fought
    # FFmpeg's decode threads (58.6 s -> 45.6 s on a 10-min clip with one thread).
    # The ops here are element-wise, so the thread count does not change any value.
    torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    for worker in workers:
        worker.start()
    try:
        while True:
            item = batches.get()
            if item is done:
                break
            if isinstance(item, BaseException):
                raise item
            batch_cpu, centers = item
            batch_tensor = batch_cpu.to(device)  # B,C,T,H,W
            with torch.inference_mode():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(batch_tensor)
                probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
            for center, prob in zip(centers, probs):
                upper = float(prob[1])
                lower = float(prob[2])
                hit_score = upper + lower
                side = "upper" if upper >= lower else "lower"
                events.append(HitEvent(frame=center, side=side, score=hit_score))
            if len(events) % 1000 < len(centers):
                print(f"  [scan-progress] scored={len(events)} centers", flush=True)
            if on_scored is not None:
                # The callback may run heavy work (stroke classification); it gets the
                # normal torch thread pool so its results match a run after the scan.
                torch.set_num_threads(torch_threads)
                try:
                    stop_requested = bool(on_scored(events, centers[-1]))
                finally:
                    torch.set_num_threads(1)
                if stop_requested:
                    break  # after a complete batch, so a later pass resumes on the batch grid
    finally:
        stop.set()
        for worker in workers:
            worker.join()
        torch.set_num_threads(torch_threads)
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
    parser.add_argument("--nms-radius", type=int, default=8,
                        help="Temporal NMS radius in frames.")
    parser.add_argument("--same-side-nms-radius", type=int, default=15,
                        help="Second-stage duplicate radius for the same hitter side.")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Hit score threshold for NMS (train-matches tuned default: 0.75).")
    parser.add_argument("--max-frames", type=int,
                        help="Process only first N frames (debugging).")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Skip this many frames at the beginning.")
    parser.add_argument("--gate", type=Path, default=None,
                        help="Path to Rally-State Gate JSON to scan only live rally segments.")
    parser.add_argument("--gate-padding", type=int, default=30,
                        help="Safety frame margin around live segments (default 30 frames = 1.0s).")
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
        gate_path=args.gate,
        gate_padding=args.gate_padding,
    )

    nms_events = two_stage_temporal_nms(
        events, threshold=args.threshold, radius=args.nms_radius,
        same_side_radius=args.same_side_nms_radius,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "video": str(args.video.resolve()),
        "stride": args.stride,
        "nms_radius": args.nms_radius,
        "same_side_nms_radius": args.same_side_nms_radius,
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
