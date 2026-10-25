"""Train BadmintonHitNetV2 from scratch on ShuttleSet with Resume & Live Logging.

Architectural and Training Foundations:
- TTNet (CVPRW 2020) & E2E-Spot (ECCV 2022) + MobileNetV3 Squeeze-and-Excitation.
- Upgraded 4-Scale MS-TCN (d ∈ {1, 2, 4, 8}) spanning the entire 16-frame rally sequence.
- Squeeze-and-Excitation (SE) channel attention focusing on tiny, high-speed shuttlecock pixels.
- Multi-task objective:
  1. Focal Loss for 3-class Hit Spotting (no_hit, upper_hit, lower_hit).
  2. Smooth L1 Loss for continuous Temporal Contact Offset Regression.
- 100% Train-From-Scratch (Kaiming Normal initialization, zero pre-trained weights).
- GPU acceleration with PyTorch AMP (float16) on NVIDIA RTX 3050.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as VF

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_classifier.models import BadmintonHitNetV2  # noqa: E402

CLASSES = ["no_hit", "upper_hit", "lower_hit"]
SIDE_LABEL: dict[str, int] = {
    "top": 1, "upper": 1,
    "bottom": 2, "lower": 2,
}

_MEAN = [0.43216, 0.394666, 0.37645]
_STD = [0.22803, 0.22145, 0.216989]


# ---------------------------------------------------------------------------
# Logging Helper
# ---------------------------------------------------------------------------

class Logger:
    """Synchronized logger that writes to both stdout and a log file with auto-flush."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_file = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = log_path.open("a", encoding="utf-8")

    def log(self, msg: str) -> None:
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
        formatted = f"{timestamp} {msg}"
        try:
            print(formatted, flush=True)
        except UnicodeEncodeError:
            print(formatted.encode("ascii", errors="replace").decode("ascii"), flush=True)
        if self.log_file:
            self.log_file.write(formatted + "\n")
            self.log_file.flush()

    def close(self) -> None:
        if self.log_file:
            self.log_file.close()


# ---------------------------------------------------------------------------
# Data structures & Manifest Loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    match_id: str
    video_path: Path
    center_frame: int
    label: int        # 0: no_hit, 1: upper_hit, 2: lower_hit
    offset: float     # Sub-frame regression target in [-2.0, +2.0]
    sample_key: str   # Unique identifier for disk cache lookup


def build_samples_from_manifest(
    manifest_path: Path,
    split: str,
    frames: int,
    neg_ratio: float = 1.0,
    seed: int = 42,
    smoke: bool = False,
) -> list[Sample]:
    rng = random.Random(seed)
    hits_by_match: dict[str, list[dict]] = {}

    with manifest_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != split:
                continue
            match_id = row.get("match_id", "")
            hits_by_match.setdefault(match_id, []).append(row)

    if smoke:
        selected_matches = sorted(hits_by_match.keys())[:2]
        hits_by_match = {m: hits_by_match[m] for m in selected_matches}

    samples: list[Sample] = []
    half = frames // 2

    for match_id, rows in hits_by_match.items():
        if not rows:
            continue
        vpath = Path(rows[0]["video_path"])
        ground_truth_frames = [int(float(r["hit_frame"])) for r in rows]
        gt_set = set(ground_truth_frames)

        for row in rows:
            hf = int(float(row["hit_frame"]))
            side = row.get("player_side", "").lower()
            label = SIDE_LABEL.get(side, 0)

            # Center sample (offset 0.0)
            sample_key = f"{match_id}_f{hf}_label{label}_off0"
            samples.append(Sample(match_id, vpath, hf, label, 0.0, sample_key))

            # Jittered sample (+/- 1 or 2 frames) for offset regression
            jitter = rng.choice([-2, -1, 1, 2])
            shifted_center = hf + jitter
            jitter_sample_key = f"{match_id}_f{shifted_center}_label{label}_off{-jitter}"
            samples.append(Sample(match_id, vpath, shifted_center, label, float(-jitter), jitter_sample_key))

        # Generate hard negatives
        num_neg = int(len(rows) * 2 * neg_ratio)
        min_f = min(ground_truth_frames)
        max_f = max(ground_truth_frames)
        neg_candidates = []
        for _ in range(num_neg * 3):
            cf = rng.randint(min_f, max_f)
            if all(abs(cf - gf) > half + 4 for gf in ground_truth_frames):
                neg_candidates.append(cf)
            if len(neg_candidates) >= num_neg:
                break

        for cf in neg_candidates[:num_neg]:
            sample_key = f"{match_id}_f{cf}_label0_off0"
            samples.append(Sample(match_id, vpath, cf, 0, 0.0, sample_key))

    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Dataset with Fast Disk Caching
# ---------------------------------------------------------------------------

class FastHitDataset(Dataset):
    """High-performance dataset that checks pre-cached .npy files before reading video."""

    def __init__(
        self,
        samples: list[Sample],
        frames: int = 16,
        cache_dir: Path | None = None,
        is_train: bool = True,
    ) -> None:
        self.samples = samples
        self.frames = frames
        self.cache_dir = cache_dir
        self.is_train = is_train
        self.mean = torch.tensor(_MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(_STD).view(3, 1, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def _read_from_video(self, vpath: Path, center: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            return np.zeros((self.frames, 112, 112, 3), dtype=np.uint8)

        half = self.frames // 2
        start = max(0, center - half)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames_list = []
        for _ in range(self.frames):
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            scale = 128.0 / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            y1 = max(0, (new_h - 112) // 2)
            x1 = max(0, (new_w - 112) // 2)
            crop = resized[y1:y1+112, x1:x1+112]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            frames_list.append(crop_rgb)
        cap.release()

        while len(frames_list) < self.frames:
            frames_list.append(frames_list[-1] if frames_list else np.zeros((112, 112, 3), dtype=np.uint8))
        return np.stack(frames_list[:self.frames])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, float]:
        sample = self.samples[idx]
        arr = None

        if self.cache_dir:
            npy_path = self.cache_dir / f"{sample.sample_key}.npy"
            if npy_path.is_file():
                try:
                    loaded = np.load(str(npy_path))
                    # Handle (16, 3, 112, 112) -> (16, 112, 112, 3)
                    if loaded.ndim == 4 and loaded.shape[1] == 3:
                        arr = np.transpose(loaded, (0, 2, 3, 1))
                    else:
                        arr = loaded
                except Exception:
                    arr = None

        if arr is None:
            arr = self._read_from_video(sample.video_path, sample.center_frame)

        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).float().div_(255.0)  # (3, T, H, W)

        if self.is_train and random.random() < 0.5:
            tensor = torch.flip(tensor, dims=[-1])
            if sample.label == 1:
                label = 2
            elif sample.label == 2:
                label = 1
            else:
                label = sample.label
        else:
            label = sample.label

        tensor = (tensor - self.mean) / self.std
        return tensor, label, sample.offset


# ---------------------------------------------------------------------------
# Multi-Task Loss: Focal Loss + Smooth L1
# ---------------------------------------------------------------------------

class MultiTaskHitLoss(nn.Module):
    """Multi-task loss combining Focal Loss for detection with Smooth L1 for offset."""

    def __init__(
        self,
        focal_gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        offset_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights
        self.offset_weight = offset_weight

    def forward(
        self,
        logits: torch.Tensor,
        pred_offset: torch.Tensor,
        targets: torch.Tensor,
        true_offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        p = torch.exp(-ce_loss)
        focal_loss = ((1.0 - p) ** self.focal_gamma * ce_loss).mean()

        hit_mask = (targets > 0)
        if hit_mask.any():
            reg_loss = F.smooth_l1_loss(
                pred_offset[hit_mask].squeeze(-1),
                true_offset[hit_mask],
                beta=1.0,
            )
        else:
            reg_loss = torch.tensor(0.0, device=logits.device)

        total_loss = focal_loss + self.offset_weight * reg_loss
        return total_loss, focal_loss, reg_loss


# ---------------------------------------------------------------------------
# Validation Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    preds: list[int],
    targets: list[int],
    pred_offsets: list[float],
    true_offsets: list[float],
) -> dict[str, float]:
    p = np.array(preds)
    t = np.array(targets)

    acc = float(np.mean(p == t))

    pred_hit = (p > 0)
    true_hit = (t > 0)

    tp = int(np.sum(pred_hit & true_hit))
    fp = int(np.sum(pred_hit & (~true_hit)))
    fn = int(np.sum((~pred_hit) & true_hit))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    hit_idx = np.where(true_hit)[0]
    if len(hit_idx) > 0:
        p_off = np.array(pred_offsets)[hit_idx]
        t_off = np.array(true_offsets)[hit_idx]
        offset_mae = float(np.mean(np.abs(p_off - t_off)))
    else:
        offset_mae = 0.0

    return {
        "accuracy": acc,
        "hit_precision": precision,
        "hit_recall": recall,
        "hit_f1": f1,
        "offset_mae": offset_mae,
    }


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("AI_Classifier/data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path(r"C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_hit_full"))
    parser.add_argument("--output-dir", type=Path, default=Path("AI_Classifier/work_dirs/badminton_hitnet_v2"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--offset-weight", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--smoke", action="store_true", help="Quick run on 2 matches")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "train.log")

    logger.log("=" * 80)
    logger.log(">>> TRAINING BADMINTON-HITNET-V2 FROM SCRATCH (SE-Attention + 4-Scale MS-TCN) <<<")
    logger.log(f"[*] Manifest:    {args.manifest}")
    logger.log(f"[*] Cache Dir:   {args.cache_dir}")
    logger.log(f"[*] Output Dir:  {out_dir}")
    logger.log(f"[*] Batch Size:  {args.batch_size} | Epochs: {args.epochs} | LR: {args.lr}")
    logger.log("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Load samples
    logger.log("[*] Parsing dataset manifests...")
    train_samples = build_samples_from_manifest(
        args.manifest, split="train", frames=args.frames, smoke=args.smoke
    )
    val_samples = build_samples_from_manifest(
        args.manifest, split="test", frames=args.frames, smoke=args.smoke
    )
    logger.log(f"[+] Loaded {len(train_samples):,} train clips and {len(val_samples):,} validation clips.")

    train_dataset = FastHitDataset(
        train_samples, frames=args.frames, cache_dir=args.cache_dir, is_train=True
    )
    val_dataset = FastHitDataset(
        val_samples, frames=args.frames, cache_dir=args.cache_dir, is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    # Instantiate BadmintonHitNetV2
    model = BadmintonHitNetV2(
        num_classes=len(CLASSES),
        in_frames=args.frames,
        dropout=args.dropout,
        spatial_dim=160,
        temporal_dim=224,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"[*] Model Architecture: BadmintonHitNetV2 (SE + 4-Scale MS-TCN) | Parameters: {total_params:,}")

    # Loss & Optimizer
    class_weights = torch.tensor([1.0, 2.5, 2.5], device=device)
    criterion = MultiTaskHitLoss(
        focal_gamma=2.0,
        class_weights=class_weights,
        offset_weight=args.offset_weight,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

    start_epoch = 1
    best_f1 = -1.0
    latest_checkpoint = out_dir / "latest.pth"

    # Auto-resume logic
    resume_path = Path(args.resume) if args.resume else (latest_checkpoint if latest_checkpoint.is_file() else None)
    if resume_path and resume_path.is_file():
        logger.log(f"[*] Resuming training from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_f1 = ckpt.get("best_f1", ckpt.get("val_metrics", {}).get("hit_f1", -1.0))
        logger.log(f"[+] Successfully resumed! Starting from epoch {start_epoch} (Best F1: {best_f1*100:.2f}%)")

    interrupted = False

    def handle_interrupt(sig, frame):
        nonlocal interrupted
        logger.log("\n[!] PAUSE SIGNAL RECEIVED. Saving checkpoint...")
        interrupted = True

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    epochs = args.epochs
    num_batches = len(train_loader)

    for epoch in range(start_epoch, epochs + 1):
        if interrupted:
            break

        model.train()
        t0 = time.time()
        total_loss_accum = 0.0
        cls_loss_accum = 0.0
        reg_loss_accum = 0.0

        logger.log(f"\n--- Epoch {epoch:02d}/{epochs:02d} (LR: {optimizer.param_groups[0]['lr']:.6f}) ---")

        for batch_idx, (clips, labels, offsets) in enumerate(train_loader, 1):
            if interrupted:
                break

            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            offsets = offsets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=args.use_amp):
                logits, pred_offset = model(clips, return_offset=True)
                loss, cls_l, reg_l = criterion(logits, pred_offset, labels, offsets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss_accum += loss.item()
            cls_loss_accum += cls_l.item()
            reg_loss_accum += reg_l.item() if isinstance(reg_l, torch.Tensor) else reg_l

            if batch_idx % args.log_interval == 0 or batch_idx == num_batches:
                elapsed = time.time() - t0
                it_per_sec = batch_idx / max(elapsed, 0.001)
                lr_curr = optimizer.param_groups[0]["lr"]
                logger.log(
                    f"Epoch [{epoch:02d}/{epochs:02d}] "
                    f"Batch [{batch_idx:03d}/{num_batches:03d}] "
                    f"Loss: {loss.item():.4f} (Cls: {cls_l.item():.4f}, Reg: {reg_l.item():.4f}) | "
                    f"LR: {lr_curr:.6f} | Speed: {it_per_sec:.1f} it/s"
                )

        if interrupted:
            logger.log(f"[!] Pausing during Epoch {epoch}. State preserved in checkpoint.")
            break

        scheduler.step()
        train_duration = time.time() - t0
        avg_loss = total_loss_accum / max(num_batches, 1)

        # Validation
        logger.log(f"[*] Validating Epoch {epoch:02d} on {len(val_samples)} test samples...")
        model.eval()
        val_preds: list[int] = []
        val_targets: list[int] = []
        val_pred_offsets: list[float] = []
        val_true_offsets: list[float] = []

        with torch.inference_mode():
            for clips, labels, offsets in val_loader:
                clips = clips.to(device)
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=args.use_amp):
                    logits, pred_offset = model(clips, return_offset=True)
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                val_preds.extend(preds)
                val_targets.extend(labels.tolist())
                val_pred_offsets.extend(pred_offset.squeeze(-1).cpu().tolist())
                val_true_offsets.extend(offsets.tolist())

        metrics = compute_metrics(val_preds, val_targets, val_pred_offsets, val_true_offsets)
        logger.log(
            f"=== [EPOCH {epoch:02d}/{epochs:02d} SUMMARY] ({train_duration:.1f}s) ===\n"
            f"    * Avg Train Loss: {avg_loss:.4f}\n"
            f"    * Val Hit F1:    {metrics['hit_f1']*100:.2f}% (Precision: {metrics['hit_precision']*100:.2f}%, Recall: {metrics['hit_recall']*100:.2f}%)\n"
            f"    * Accuracy:      {metrics['accuracy']*100:.2f}%\n"
            f"    * Offset MAE:    {metrics['offset_mae']:.2f} frames"
        )

        # Checkpoint payload
        checkpoint_payload = {
            "epoch": epoch,
            "arch": "badminton_hitnet_v2",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_f1": max(best_f1, metrics["hit_f1"]),
            "val_metrics": metrics,
            "frames": args.frames,
            "classes": CLASSES,
        }

        # Always save latest
        torch.save(checkpoint_payload, latest_checkpoint)

        # Save best champion model
        if metrics["hit_f1"] > best_f1:
            best_f1 = metrics["hit_f1"]
            best_path = out_dir / "best.pth"
            torch.save(checkpoint_payload, best_path)
            logger.log(f"    [BEST] NEW CHAMPION HITNET-V2 CHECKPOINT SAVED: {best_path} (F1: {best_f1*100:.2f}%)")

    logger.log("\n" + "=" * 80)
    if not interrupted:
        logger.log(f"[SUCCESS] BadmintonHitNetV2 Training Completed 100%! Best F1: {best_f1*100:.2f}%")
        logger.log(f"[*] Champion weights saved to: {out_dir / 'best.pth'}")
    else:
        logger.log(f"[PAUSED] Training paused at Epoch {epoch}. Resume anytime with --resume.")
    logger.log("=" * 80)
    logger.close()


if __name__ == "__main__":
    main()

