"""Classify one pre-segmented badminton stroke clip with the RGB baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import r2plus1d_18
from torchvision.transforms import functional as vision_functional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def read_clip(path: Path, frame_count: int) -> tuple[torch.Tensor, dict[str, object]]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        total = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if total <= 0 or fps <= 0:
            raise ValueError(f"Invalid video metadata: frames={total}, fps={fps}")
        wanted = np.rint(np.linspace(0, total - 1, frame_count)).astype(np.int64)
        needed = set(int(index) for index in wanted)
        decoded: dict[int, np.ndarray] = {}
        for index in range(total):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Video stopped decoding at frame {index}/{total}")
            if index in needed:
                decoded[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    missing = needed.difference(decoded)
    if missing:
        raise ValueError(f"Missing sampled frames: {sorted(missing)}")
    frames = np.stack([decoded[int(index)] for index in wanted])
    tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)
    tensor = vision_functional.resize(tensor, [128, 171], antialias=False)
    tensor = vision_functional.center_crop(tensor, [112, 112])
    tensor = tensor.to(torch.float32).div_(255.0)
    mean = tensor.new_tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
    std = tensor.new_tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
    tensor = ((tensor - mean) / std).permute(1, 0, 2, 3).unsqueeze(0)
    return tensor, {
        "source_frames": total,
        "sampled_frames": wanted.tolist(),
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": total / fps,
    }


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = [str(item) for item in checkpoint["classes"]]
    frames = int(checkpoint.get("frames", 16))
    clip, metadata = read_clip(args.video, frames)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = r2plus1d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        probabilities = torch.softmax(model(clip.to(device))[0], dim=0).cpu()

    top_k = min(max(1, args.top_k), len(classes))
    values, indices = torch.topk(probabilities, top_k)
    ranked = [
        {
            "label": classes[int(index)],
            "probability": float(value),
        }
        for value, index in zip(values, indices)
    ]
    result = {
        "video": str(args.video.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "prediction": ranked[0],
        "top_k": ranked,
        "scores": {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(classes)
        },
        "video_metadata": metadata,
        "scope_warning": (
            "This model predicts one of eight coarse stroke families for a clip "
            "that should contain exactly one singles-match stroke. It does not "
            "predict forehand/backhand or the hitter side."
        ),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
