"""Fast in-memory windowed TrackNet inference module.

Runs TrackNetV3 specifically around an event frame (or full clip for short videos),
keeping the model loaded in GPU memory and matching exact TrackNet preprocessing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from external.TrackNetV3.model import TrackNet

TRACKNET_WIDTH = 512
TRACKNET_HEIGHT = 288
SEQ_LEN = 8
# Resized TrackNet frames kept per video (~0.44 MB each); a few hit windows' worth.
RESIZE_CACHE_FRAMES = 256


def predict_location(heatmap: np.ndarray) -> tuple[int, int, int, int]:
    """Get coordinates from the TrackNet binary heatmap."""
    if np.amax(heatmap) == 0:
        return 0, 0, 0, 0
    cnts, _ = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, 0, 0, 0
    rects = [cv2.boundingRect(ctr) for ctr in cnts]
    max_area_idx = 0
    max_area = rects[0][2] * rects[0][3]
    for i in range(1, len(rects)):
        area = rects[i][2] * rects[i][3]
        if area > max_area:
            max_area_idx = i
            max_area = area
    return rects[max_area_idx]


def _median_from_stack(stack: torch.Tensor) -> torch.Tensor:
    """Per-pixel median over the first axis of a uint8 device tensor.

    Equals ``np.median(frames, axis=0).astype(np.uint8)``: numpy averages the two
    middle values of an even count and the cast truncates, which is
    floor((a + b) / 2). Sorting on the GPU is about 20x faster than numpy's
    partition on a 50-frame window.
    """
    ordered = stack.sort(dim=0).values
    count = ordered.shape[0]
    if count % 2:
        median = ordered[count // 2]
    else:
        median = (ordered[count // 2 - 1].to(torch.int16) + ordered[count // 2].to(torch.int16)) // 2
    return median.to(torch.uint8)


class FastTrackNet:
    """In-memory TrackNetV3 optimized for localized window or rally inference."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if checkpoint_path is None:
            canonical_ckpt = Path(__file__).resolve().parents[1] / "models" / "checkpoints" / "tracknet_best.pt"
            checkpoint_path = canonical_ckpt if canonical_ckpt.exists() else "external/TrackNetV3/ckpts/TrackNet_best.pt"
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"TrackNet checkpoint not found at: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        self.seq_len = ckpt.get("param_dict", {}).get("seq_len", SEQ_LEN)
        self.bg_mode = ckpt.get("param_dict", {}).get("bg_mode", "concat")

        in_dim = (self.seq_len + 1) * 3 if self.bg_mode == "concat" else self.seq_len * 3
        out_dim = self.seq_len

        self.model = TrackNet(in_dim, out_dim).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # Warmup for cuDNN kernel selection
        if self.device.type == "cuda":
            dummy = torch.zeros(1, in_dim, TRACKNET_HEIGHT, TRACKNET_WIDTH, device=self.device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                _ = self.model(dummy)

    def predict_window(
        self,
        video_path: Union[str, Path],
        center_frame: int,
        window_before: int = 32,
        window_after: int = 32,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """Run TrackNet on a temporal window around center_frame (or full clip if short).

        Returns DataFrame with columns ['Frame', 'Visibility', 'X', 'Y'].
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # If clip is already short (<= 128 frames), process all frames from 0 to total_frames - 1
        # to ensure 100% exact alignment with TrackNet sequence boundaries and true video background median.
        if total_frames <= 128:
            start_f = 0
            end_f = total_frames - 1
        else:
            # Align start_f to a multiple of SEQ_LEN
            start_f = max(0, ((center_frame - window_before) // self.seq_len) * self.seq_len)
            end_f = min(total_frames - 1, center_frame + window_after)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frames = []
        frame_indices = []
        for f_idx in range(start_f, end_f + 1):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_indices.append(f_idx)
        cap.release()

        if not frames:
            return pd.DataFrame(columns=["Frame", "Visibility", "X", "Y"])

        return self.predict_frames(frames, frame_indices, orig_w, orig_h, batch_size=batch_size)

    def predict_frames(
        self,
        bgr_frames: list[np.ndarray],
        frame_indices: list[int],
        orig_w: int,
        orig_h: int,
        batch_size: int = 32,
        resize_cache: Optional[dict[int, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Predict shuttle coordinates on a provided sequence of BGR frames.

        ``resize_cache`` maps frame index -> resized RGB frame for one video; pass a fresh
        dict per video, since frame indices of different videos collide.
        """
        num_frames = len(bgr_frames)
        if num_frames == 0:
            return pd.DataFrame(columns=["Frame", "Visibility", "X", "Y"])

        w_scaler = orig_w / TRACKNET_WIDTH
        h_scaler = orig_h / TRACKNET_HEIGHT

        # Resize using OpenCV INTER_CUBIC (matches PIL Bicubic within <0.2 px), then convert to RGB.
        # Resizing filters each channel alike, so swapping channels afterwards gives the same bytes
        # while converting 9x fewer pixels.
        # Neighbouring hit windows overlap, so a per-video cache keyed by frame index skips repeated resizes.
        resized_rgb = []
        for img, f_idx in zip(bgr_frames, frame_indices):
            small = resize_cache.get(f_idx) if resize_cache is not None else None
            if small is None:
                small = cv2.cvtColor(
                    cv2.resize(img, (TRACKNET_WIDTH, TRACKNET_HEIGHT), interpolation=cv2.INTER_CUBIC),
                    cv2.COLOR_BGR2RGB,
                )
                if resize_cache is not None:
                    resize_cache[f_idx] = small
            resized_rgb.append(small)
        if resize_cache is not None:
            while len(resize_cache) > RESIZE_CACHE_FRAMES:
                resize_cache.pop(next(iter(resize_cache)))

        # Pad frames to a multiple of SEQ_LEN
        remainder = num_frames % self.seq_len
        pad_count = (self.seq_len - remainder) if remainder != 0 else 0
        padded_indices = list(frame_indices) + [frame_indices[-1]] * pad_count
        total_sequences = len(padded_indices) // self.seq_len
        seq_frame_indices = [
            padded_indices[s * self.seq_len : (s + 1) * self.seq_len] for s in range(total_sequences)
        ]

        # The input tensor is assembled on the GPU from one uint8 upload; building it in
        # float32 numpy took longer than the model itself. Two details keep the model
        # output bit-identical to the former numpy path: division by a device tensor
        # keeps true IEEE division, and the tensor is laid out channels-last (NHWC),
        # which is the stride pattern the numpy concatenation produced and which
        # selects the same cuDNN kernels.
        stack = torch.from_numpy(np.stack(resized_rgb)).to(self.device)  # (T, H, W, 3) uint8
        median = _median_from_stack(stack)  # (H, W, 3) uint8, background over the real frames
        if pad_count:
            stack = torch.cat((stack, stack[-1:].expand(pad_count, -1, -1, -1)), dim=0)
        height, width = stack.shape[1], stack.shape[2]
        # (S, 8, H, W, 3) -> (S, H, W, 8, 3) -> (S, H, W, 24): frame-major channel order.
        frames_t = stack.view(total_sequences, self.seq_len, height, width, 3).permute(0, 2, 3, 1, 4)
        frames_t = frames_t.reshape(total_sequences, height, width, self.seq_len * 3).float()
        if self.bg_mode == "concat":
            median_t = median.float().unsqueeze(0).expand(total_sequences, -1, -1, -1)
            sequences = torch.cat((median_t, frames_t), dim=3)  # (S, H, W, 27)
        else:
            sequences = frames_t
        sequences = (sequences / torch.tensor(255.0, device=self.device)).permute(0, 3, 1, 2)  # NCHW view

        # Inference in batches with AMP FP16
        pred_dict = {"Frame": [], "Visibility": [], "X": [], "Y": []}
        seen_frames = set()

        with torch.inference_mode():
            for b_start in range(0, total_sequences, batch_size):
                b_end = min(b_start + batch_size, total_sequences)
                batch_t = sequences[b_start:b_end]

                # Forward pass: output shape (B, 8, H, W) with FP16
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                    y_pred = self.model(batch_t)
                y_pred = (y_pred > 0.5).cpu().numpy().astype(np.uint8)

                for n in range(b_end - b_start):
                    seq_i = b_start + n
                    for f in range(self.seq_len):
                        f_idx = seq_frame_indices[seq_i][f]
                        if f_idx in seen_frames:
                            continue
                        seen_frames.add(f_idx)

                        heatmap = y_pred[n, f] * 255
                        bbox = predict_location(heatmap)
                        if bbox == (0, 0, 0, 0):
                            cx, cy, vis = 0, 0, 0
                        else:
                            cx = int((bbox[0] + bbox[2] / 2) * w_scaler)
                            cy = int((bbox[1] + bbox[3] / 2) * h_scaler)
                            vis = 1 if (cx > 0 or cy > 0) else 0

                        pred_dict["Frame"].append(int(f_idx))
                        pred_dict["Visibility"].append(int(vis))
                        pred_dict["X"].append(int(cx))
                        pred_dict["Y"].append(int(cy))

        df = pd.DataFrame(pred_dict)
        return df.sort_values("Frame").reset_index(drop=True)
