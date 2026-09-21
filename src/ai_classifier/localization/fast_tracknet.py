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


class FastTrackNet:
    """In-memory TrackNetV3 optimized for localized window or rally inference."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "external/TrackNetV3/ckpts/TrackNet_best.pt",
        device: Optional[Union[str, torch.device]] = None,
    ):
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

    def predict_window(
        self,
        video_path: Union[str, Path],
        center_frame: int,
        window_before: int = 32,
        window_after: int = 32,
        batch_size: int = 8,
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
        batch_size: int = 8,
    ) -> pd.DataFrame:
        """Predict shuttle coordinates on a provided sequence of BGR frames."""
        num_frames = len(bgr_frames)
        if num_frames == 0:
            return pd.DataFrame(columns=["Frame", "Visibility", "X", "Y"])

        w_scaler = orig_w / TRACKNET_WIDTH
        h_scaler = orig_h / TRACKNET_HEIGHT

        # Convert to RGB and resize using PIL Bicubic to match TrackNetV3 exact preprocessing
        resized_rgb = []
        for img in bgr_frames:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb).resize((TRACKNET_WIDTH, TRACKNET_HEIGHT))
            resized_rgb.append(np.array(pil_img))

        # Background median across frames (shape: 3, H, W)
        median_img = np.median(resized_rgb, axis=0).astype(np.uint8)
        median_ch = np.moveaxis(median_img, -1, 0).astype(np.float32)  # (3, H, W)

        # Pad frames to a multiple of SEQ_LEN
        remainder = num_frames % self.seq_len
        pad_count = (self.seq_len - remainder) if remainder != 0 else 0

        padded_rgb = list(resized_rgb)
        padded_indices = list(frame_indices)
        last_frame = resized_rgb[-1]
        last_idx = frame_indices[-1]
        for _ in range(pad_count):
            padded_rgb.append(last_frame)
            padded_indices.append(last_idx)

        total_sequences = len(padded_rgb) // self.seq_len

        # Assemble sequences of shape (total_sequences, 27, H, W)
        sequences = []
        seq_frame_indices = []
        for s in range(total_sequences):
            seq_imgs = padded_rgb[s * self.seq_len : (s + 1) * self.seq_len]
            seq_idxs = padded_indices[s * self.seq_len : (s + 1) * self.seq_len]

            # Stack 8 frames along channel dimension -> (24, H, W)
            stacked_frames = [np.moveaxis(im, -1, 0).astype(np.float32) for im in seq_imgs]
            frames_tensor = np.concatenate(stacked_frames, axis=0)  # (24, H, W)

            if self.bg_mode == "concat":
                full_input = np.concatenate((median_ch, frames_tensor), axis=0)  # (27, H, W)
            else:
                full_input = frames_tensor

            full_input /= 255.0
            sequences.append(full_input)
            seq_frame_indices.append(seq_idxs)

        sequences = np.stack(sequences, axis=0)  # (N, 27, H, W)

        # Inference in batches (FP32 exact matching)
        pred_dict = {"Frame": [], "Visibility": [], "X": [], "Y": []}
        seen_frames = set()

        with torch.inference_mode():
            for b_start in range(0, total_sequences, batch_size):
                b_end = min(b_start + batch_size, total_sequences)
                batch_t = torch.from_numpy(sequences[b_start:b_end]).float().to(self.device)

                # Forward pass: output shape (B, 8, H, W)
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
