"""Train BadmintonHitNet from scratch on ShuttleSet with Resume & Live Logging.

Architectural and Training Foundations:
- TTNet (CVPRW 2020) & E2E-Spot (ECCV 2022): Spatio-temporal micro-event spotting
  with frame differencing and multi-scale temporal convolutions (MS-TCN).
- Multi-task objective:
  1. Focal Loss for 3-class Hit Spotting (no_hit, upper_hit, lower_hit).
  2. Smooth L1 Loss for continuous Temporal Contact Offset Regression.
- 100% Train-From-Scratch (Kaiming Normal initialization, zero pre-trained weights).
- GPU acceleration with PyTorch AMP (float16) on NVIDIA RTX 3050.
- Checkpoint Resume & Graceful Pause: automatically saves `latest.pth` and resumes anytime.
- Real-time logging to stdout and `train.log`.

Usage:
    # Full training from scratch (or automatically resumes if interrupted)
    python scripts/training/train_badminton_hitnet.py --epochs 15 --batch-size 16

    # Force resume from a specific checkpoint
    python scripts/training/train_badminton_hitnet.py --resume work_dirs/badminton_hitnet/latest.pth

    # Smoke test (fast verification on 2 matches)
    python scripts/training/train_badminton_hitnet.py --smoke
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

from ai_classifier.models import BadmintonHitNet, BadmintonHitNetV2  # noqa: E402

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
    center: int
    label: int        # 0=no_hit, 1=upper_hit, 2=lower_hit
    offset: float     # Ground-truth temporal offset to true contact (frames)


def load_matches(manifest: Path, split: str, *, match_ids: set[str] | None = None) -> dict[str, dict]:
    """Return {match_id: {video_path, hits: [(frame, label), ...]}}."""
    matches: dict[str, dict] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("split") != split:
                continue
            mid = row["match_id"]
            if match_ids is not None and mid not in match_ids:
                continue
            side = SIDE_LABEL.get(row["player_side"].strip().lower())
            if side is None:
                continue
            item = matches.setdefault(mid, {"video_path": Path(row["video_path"]), "hits": []})
            item["hits"].append((int(float(row["hit_frame"])), side))
    for item in matches.values():
        item["hits"] = sorted(set(item["hits"]))
    return matches


def _video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 1)


def build_samples(
    matches: dict[str, dict],
    positives_per_match: int,
    negative_ratio: float,
    exclusion: int,
    radius: int,
    seed: int,
    *,
    full_video_negatives: bool = True,
) -> list[Sample]:
    """Build balanced sample list with jittered contact offsets for regression."""
    rng = random.Random(seed)
    samples: list[Sample] = []
    for match_id, item in sorted(matches.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        hits = list(item["hits"])
        rng.shuffle(hits)
        positives = hits[:min(positives_per_match, len(hits))]
        for frame, label in positives:
            jitter = rng.randint(-radius, radius) if radius > 0 else 0
            true_offset = float(-jitter)
            samples.append(Sample(match_id, item["video_path"], max(0, frame + jitter), label, true_offset))

        all_frames = np.asarray([f for f, _ in item["hits"]], dtype=np.int64)
        if len(all_frames) == 0:
            continue

        if full_video_negatives:
            n_total = _video_frame_count(item["video_path"])
            low, high = 0, n_total - 1
        else:
            low, high = int(all_frames.min()), int(all_frames.max())

        wanted = int(round(len(positives) * negative_ratio))
        negatives: set[int] = set()
        attempts = 0
        while len(negatives) < wanted and attempts < wanted * 500:
            attempts += 1
            frame = rng.randint(low, high)
            if int(np.min(np.abs(all_frames - frame))) > exclusion:
                negatives.add(frame)
        samples.extend(Sample(match_id, item["video_path"], f, 0, 0.0) for f in negatives)

    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HitDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        frames: int = 16,
        training: bool = False,
        cache_dir: Path | None = None,
    ) -> None:
        self.samples = samples
        self.frames = frames
        self.training = training
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.samples)

    def _decode(self, sample: Sample) -> torch.Tensor:
        key = f"{sample.match_id}_{sample.center}_{self.frames}.npy"
        cache_path = self.cache_dir / key if self.cache_dir else None
        if cache_path is not None and cache_path.is_file():
            try:
                return torch.from_numpy(np.load(cache_path))
            except Exception:
                pass

        half = self.frames // 2
        start = max(0, sample.center - half)
        cap = cv2.VideoCapture(str(sample.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        decoded: list[np.ndarray] = []
        try:
            for _ in range(self.frames):
                ok, frame = cap.read()
                if not ok:
                    break
                decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()

        if not decoded:
            decoded = [np.zeros((128, 171, 3), dtype=np.uint8) for _ in range(self.frames)]

        while len(decoded) < self.frames:
            decoded.append(decoded[-1])

        tensor = torch.from_numpy(np.stack(decoded)).permute(0, 3, 1, 2)  # T,C,H,W
        tensor = VF.resize(tensor, [128, 171], antialias=False)
        tensor = VF.center_crop(tensor, [112, 112]).contiguous()

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp.npy")
            np.save(tmp, tensor.numpy())
            tmp.replace(cache_path)
        return tensor

    def __getitem__(self, index: int):
        sample = self.samples[index]
        tensor = self._decode(sample).float().div_(255.0)
        mean = tensor.new_tensor(_MEAN).view(1, 3, 1, 1)
        std = tensor.new_tensor(_STD).view(1, 3, 1, 1)
        tensor = ((tensor - mean) / std).permute(1, 0, 2, 3)  # C,T,H,W
        if self.training:
            if torch.rand(()) < 0.5:
                tensor = torch.flip(tensor, dims=(-1,))  # Horizontal flip
        return tensor, sample.label, sample.offset


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Multi-class focal loss for heavy foreground/background imbalance."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=-1)
        prob = log_prob.exp()
        log_p_t = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        loss = -focal_weight * log_p_t
        if self.weight is not None:
            class_weight = self.weight[targets]
            loss = loss * class_weight
        return loss.mean()


class MultiTaskHitLoss(nn.Module):
    """Joint Hit Classification + Temporal Offset Regression Loss."""

    def __init__(
        self,
        focal_gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        offset_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, weight=class_weights)
        self.offset_weight = offset_weight

    def forward(
        self,
        logits: torch.Tensor,
        pred_offset: torch.Tensor,
        labels: torch.Tensor,
        true_offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_loss = self.focal(logits, labels)
        pos_mask = labels > 0
        if pos_mask.sum() > 0:
            pred_pos = pred_offset.squeeze(-1)[pos_mask]
            true_pos = true_offset[pos_mask]
            reg_loss = F.smooth_l1_loss(pred_pos, true_pos, beta=1.0)
        else:
            reg_loss = torch.tensor(0.0, device=logits.device)
        total_loss = cls_loss + self.offset_weight * reg_loss
        return total_loss, cls_loss, reg_loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    all_preds: list[int],
    all_targets: list[int],
    all_pred_offsets: list[float] | None = None,
    all_true_offsets: list[float] | None = None,
) -> dict[str, float]:
    preds = np.array(all_preds)
    targets = np.array(all_targets)

    pred_hit = preds > 0
    target_hit = targets > 0

    tp = int(np.logical_and(pred_hit, target_hit).sum())
    fp = int(np.logical_and(pred_hit, ~target_hit).sum())
    fn = int(np.logical_and(~pred_hit, target_hit).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    acc = float((preds == targets).mean())

    metrics = {
        "accuracy": acc,
        "hit_precision": precision,
        "hit_recall": recall,
        "hit_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }

    if all_pred_offsets and all_true_offsets:
        pos_indices = np.where(target_hit)[0]
        if len(pos_indices) > 0:
            po = np.array(all_pred_offsets)[pos_indices]
            to = np.array(all_true_offsets)[pos_indices]
            mae = float(np.abs(po - to).mean())
            metrics["offset_mae"] = mae

    return metrics


# ---------------------------------------------------------------------------
# Main Training Function with Pause & Resume
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "train.log")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.log("=" * 70)
    logger.log("[*] STARTING BADMINTONHITNET TRAINING (Train-From-Scratch)")
    logger.log(f"[*] Hardware: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    logger.log(f"[*] Smoke Test: {args.smoke} | Output Directory: {out_dir}")
    logger.log("=" * 70)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Build matches
    smoke_train = {"1"} if args.smoke else None
    smoke_val = {"35"} if args.smoke else None
    train_matches = load_matches(manifest_path, "train", match_ids=smoke_train)
    val_matches = load_matches(manifest_path, "val", match_ids=smoke_val)

    positives_train = 16 if args.smoke else args.positives_per_match
    positives_val = 8 if args.smoke else 150

    train_samples = build_samples(
        train_matches,
        positives_per_match=positives_train,
        negative_ratio=args.negative_ratio,
        exclusion=args.exclusion,
        radius=args.radius,
        seed=seed,
    )
    val_samples = build_samples(
        val_matches,
        positives_per_match=positives_val,
        negative_ratio=args.negative_ratio,
        exclusion=args.exclusion,
        radius=0,
        seed=seed + 1,
        full_video_negatives=False,
    )

    logger.log(f"[*] Dataset: {len(train_samples):,} train clips | {len(val_samples):,} val clips")

    train_cache = (Path(args.cache_dir) / "train") if args.cache_dir else None
    val_cache = (Path(args.cache_dir) / "val") if args.cache_dir else None

    train_dataset = HitDataset(train_samples, frames=args.frames, training=True, cache_dir=train_cache)
    val_dataset = HitDataset(val_samples, frames=args.frames, training=False, cache_dir=val_cache)

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

    # Instantiate model
    if getattr(args, "arch", "badminton_hitnet") == "badminton_hitnet_v2":
        model = BadmintonHitNetV2(
            num_classes=len(CLASSES),
            in_frames=args.frames,
            dropout=args.dropout,
        ).to(device)
    else:
        model = BadmintonHitNet(
            num_classes=len(CLASSES),
            in_frames=args.frames,
            dropout=args.dropout,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"[*] Model Architecture: {getattr(args, 'arch', 'badminton_hitnet')} | Parameters: {total_params:,}")

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
        logger.log(f"[+] Successfully resumed! Starting from epoch {start_epoch} (Best F1 so far: {best_f1*100:.2f}%)")

    # Graceful interrupt handler
    interrupted = False

    def handle_interrupt(sig, frame):
        nonlocal interrupted
        logger.log("\n[!] PAUSE SIGNAL RECEIVED (Ctrl+C / Kill). Saving current progress...")
        interrupted = True

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    epochs = 1 if args.smoke else args.epochs

    if start_epoch > epochs:
        logger.log(f"[*] Training already completed up to epoch {epochs}. Nothing to do.")
        return

    logger.log(f"[*] Starting training loop from Epoch {start_epoch} to {epochs}...")

    for epoch in range(start_epoch, epochs + 1):
        if interrupted:
            break

        t0 = time.time()
        model.train()
        total_loss_accum = 0.0
        cls_loss_accum = 0.0
        reg_loss_accum = 0.0
        num_batches = len(train_loader)

        for batch_idx, (clips, labels, offsets) in enumerate(train_loader, start=1):
            if interrupted:
                break

            clips = clips.to(device)
            labels = labels.to(device)
            offsets = offsets.float().to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=args.use_amp):
                logits, pred_offset = model(clips, return_offset=True)
                loss, cls_l, reg_l = criterion(logits, pred_offset, labels, offsets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss_accum += loss.item()
            cls_loss_accum += cls_l.item()
            reg_loss_accum += reg_l.item() if isinstance(reg_l, torch.Tensor) else reg_l

            # Log batch progress every log_interval batches
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
            f"    * Offset MAE:    {metrics.get('offset_mae', 0.0):.2f} frames"
        )

        # Save checkpoint (includes full optimizer and scheduler states for resume)
        checkpoint_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_f1": best_f1,
            "arch": getattr(args, "arch", "badminton_hitnet"),
            "frames": args.frames,
            "classes": CLASSES,
            "val_metrics": metrics,
        }
        torch.save(checkpoint_data, latest_checkpoint)

        if metrics["hit_f1"] > best_f1:
            best_f1 = metrics["hit_f1"]
            checkpoint_data["best_f1"] = best_f1
            torch.save(checkpoint_data, out_dir / "best.pth")
            logger.log(f"    [BEST] New best model saved to best.pth! (Hit F1: {best_f1*100:.2f}%)")

        logger.log("-" * 70)

    logger.log(f"[*] Training finished or paused. Best validation Hit F1: {best_f1*100:.2f}%")
    logger.log(f"[*] Checkpoints: {latest_checkpoint} (can be resumed anytime with --resume)")
    logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BadmintonHitNet from scratch with pause & resume")
    parser.add_argument("--arch", type=str, default="badminton_hitnet", choices=["badminton_hitnet", "badminton_hitnet_v2"])
    parser.add_argument("--manifest", type=str, default="AI_Classifier/data/manifests/shuttleset_rgb.csv")
    parser.add_argument("--output-dir", type=str, default="AI_Classifier/work_dirs/badminton_hitnet")
    parser.add_argument("--cache-dir", type=str, default=r"C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_hit_full")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--positives-per-match", type=int, default=400)
    parser.add_argument("--negative-ratio", type=float, default=2.5)
    parser.add_argument("--exclusion", type=int, default=12)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--offset-weight", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=50, help="Batches between progress prints")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--smoke", action="store_true", help="Quick 1-epoch test on 2 matches")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
