"""Train R(2+1)D-18 hit detector on full ShuttleSet (all train matches).

Improvements over the pilot script:
- Uses ALL hits in train split (capped per-match) instead of a fixed small count.
- Negatives are sampled across the full video duration, not just between first/last hit.
- Focal loss handles heavy class imbalance.
- Progressive layer unfreezing: fc only (epoch 1-2) → +layer4 (epoch 3+).
- Warm-starts from an existing checkpoint if provided.
- AMP (float16) for RTX 3050 speedup.
- --smoke flag for quick integration test (2 matches, 1 epoch).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import r2plus1d_18
from torchvision.transforms import functional as VF

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training.train_fine_badminton_rgb import confusion_metrics  # noqa: E402

CLASSES = ["no_hit", "upper_hit", "lower_hit"]
SIDE_LABEL: dict[str, int] = {
    "top": 1, "upper": 1,
    "bottom": 2, "lower": 2,
}

# Kinetics-400 normalisation constants (same as pilot)
_MEAN = [0.43216, 0.394666, 0.37645]
_STD = [0.22803, 0.22145, 0.216989]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    match_id: str
    video_path: Path
    center: int
    label: int  # 0=no_hit, 1=upper_hit, 2=lower_hit


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_matches(manifest: Path, split: str, *, match_ids: set[str] | None = None) -> dict[str, dict]:
    """Return {match_id: {video_path, hits: [(frame, label), ...]}}."""
    matches: dict[str, dict] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["split"] != split:
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


# ---------------------------------------------------------------------------
# Sample building
# ---------------------------------------------------------------------------

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
    """Build balanced train/val sample list.

    full_video_negatives=True: sample negatives across the entire video
    duration (more representative than only between first/last hit).
    """
    rng = random.Random(seed)
    samples: list[Sample] = []
    for match_id, item in sorted(matches.items(), key=lambda kv: int(kv[0])):
        hits = list(item["hits"])
        rng.shuffle(hits)
        positives = hits[:min(positives_per_match, len(hits))]
        for frame, label in positives:
            jitter = rng.randint(-radius, radius) if radius > 0 else 0
            samples.append(Sample(match_id, item["video_path"], max(0, frame + jitter), label))

        all_frames = np.asarray([f for f, _ in item["hits"]], dtype=np.int64)
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
        samples.extend(Sample(match_id, item["video_path"], f, 0) for f in negatives)

    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HitDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        frames: int,
        training: bool,
        cache_dir: Path | None,
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
            return torch.from_numpy(np.load(cache_path))

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
            raise ValueError(f"Cannot decode {sample.video_path} at frame {start}")
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
                tensor = torch.flip(tensor, dims=(-1,))
        return tensor, sample.label


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Multi-class focal loss.

    Reduces the relative loss for easy examples and focuses training on hard
    misclassified examples. Particularly useful when negatives dominate.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)  # type: ignore[attr-defined]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=-1)
        prob = log_prob.exp()
        # Gather the log_prob for the correct class
        log_p_t = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        loss = -focal_weight * log_p_t
        if self.weight is not None:
            class_weight = self.weight[targets]
            loss = loss * class_weight
        return loss.mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(source_checkpoint: Path | None, unfreeze_layer4: bool, is_resume: bool = False) -> nn.Module:
    """Load R(2+1)D-18, optionally warm-start from a checkpoint."""
    model = r2plus1d_18(weights=None)
    if is_resume:
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    if source_checkpoint is not None and source_checkpoint.is_file():
        ckpt = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        if is_resume:
            model.load_state_dict(state, strict=True)
            print(f"[INFO] exact resume from {source_checkpoint.name} (epoch {ckpt.get('epoch', '?')})")
        else:
            stripped = {
                k.removeprefix("backbone."): v
                for k, v in state.items()
                if not k.startswith("backbone.fc.") and not k.startswith("fc.")
            }
            missing, unexpected = model.load_state_dict(stripped, strict=False)
            non_fc_missing = [k for k in missing if not k.startswith("fc.")]
            if unexpected or non_fc_missing:
                print(f"[WARN] checkpoint mismatch: missing={non_fc_missing}, unexpected={unexpected}")
            else:
                print(f"[INFO] warm-start from {source_checkpoint.name}")
    if not is_resume:
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    # Freeze all layers initially; unfreeze progressively during training
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    if unfreeze_layer4:
        for p in model.layer4.parameters():
            p.requires_grad = True

    return model


def unfreeze_layer4(model: nn.Module, lr: float, optimizer: torch.optim.Optimizer) -> None:
    """Progressively unfreeze layer4 and add to optimizer param group."""
    for p in model.layer4.parameters():
        p.requires_grad = True
    # Check if layer4 already in optimizer
    already = any(
        id(p) in {id(ep) for ep in pg["params"]}
        for pg in optimizer.param_groups
        for p in model.layer4.parameters()
    )
    if not already:
        optimizer.add_param_group({"params": list(model.layer4.parameters()), "lr": lr})
        print(f"[INFO] unfreezing layer4, lr={lr}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
) -> tuple[float, torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    if training:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm3d) and not any(p.requires_grad for p in module.parameters()):
                module.eval()

    confusion = torch.zeros(3, 3, dtype=torch.int64)
    loss_sum = count = 0
    ctx = torch.enable_grad() if training else torch.inference_mode()
    with ctx:
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(clips)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            for truth, pred in zip(labels.cpu(), logits.argmax(1).cpu()):
                confusion[int(truth), int(pred)] += 1
            loss_sum += float(loss.detach()) * len(labels)
            count += len(labels)

    return loss_sum / max(count, 1), confusion


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("work_dirs/r2plus1d18_hit_full"))
    parser.add_argument("--source-checkpoint", type=Path,
                        default=Path("work_dirs/r2plus1d18_shuttleset_hit_pilot/best.pth"),
                        help="Warm-start from this checkpoint (pilot or multitask).")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--positive-radius", type=int, default=2,
                        help="Jitter in frames applied to positive samples during train.")
    parser.add_argument("--negative-exclusion", type=int, default=12,
                        help="Minimum distance from any hit frame to count as negative.")
    parser.add_argument("--positives-per-match", type=int, default=400,
                        help="Max positive clips sampled per match (train split).")
    parser.add_argument("--val-positives-per-match", type=int, default=150)
    parser.add_argument("--negative-ratio", type=float, default=2.5,
                        help="Negatives per positive.")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Gamma for focal loss (0 = standard cross-entropy).")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="LR for the classification head (fc layer).")
    parser.add_argument("--backbone-lr", type=float, default=3e-5,
                        help="LR applied to backbone layers when unfrozen.")
    parser.add_argument("--unfreeze-layer4-epoch", type=int, default=3,
                        help="Epoch at which layer4 is unfrozen (0 = from start, 999 = never).")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cache-dir", type=Path,
                        help="Optional directory to cache decoded uint8 clip tensors.")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Enable automatic mixed precision (default: True if CUDA).")
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test: 2 train matches, 1 val match, 1 epoch.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training completely from --source-checkpoint (keeps FC and optimizer state).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ---- dataset ----
    smoke_train_ids = {"13", "14"} if args.smoke else None
    smoke_val_ids = {"35"} if args.smoke else None
    smoke_epochs = 1 if args.smoke else args.epochs
    smoke_pos = 20 if args.smoke else args.positives_per_match
    smoke_vpos = 10 if args.smoke else args.val_positives_per_match

    train_matches = load_matches(args.manifest, "train", match_ids=smoke_train_ids)
    val_matches = load_matches(args.manifest, "val", match_ids=smoke_val_ids)

    print(f"[INFO] train: {len(train_matches)} matches  val: {len(val_matches)} matches")

    # For smoke test skip full-video sampling (slow)
    train_samples = build_samples(
        train_matches, smoke_pos, args.negative_ratio,
        args.negative_exclusion, args.positive_radius, args.seed,
        full_video_negatives=not args.smoke,
    )
    val_samples = build_samples(
        val_matches, smoke_vpos, args.negative_ratio,
        args.negative_exclusion, 0, args.seed + 1,
        full_video_negatives=False,
    )

    print(f"[INFO] train samples: {len(train_samples)}  val samples: {len(val_samples)}")
    counts = np.bincount([s.label for s in train_samples], minlength=3)
    print(f"[INFO] train class counts: {counts.tolist()}")

    # ---- loaders ----
    train_cache = (args.cache_dir / "train") if args.cache_dir else None
    val_cache = (args.cache_dir / "val") if args.cache_dir else None
    train_loader = DataLoader(
        HitDataset(train_samples, args.frames, True, train_cache),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        HitDataset(val_samples, args.frames, False, val_cache),
        batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True,
    )

    # ---- model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_ckpt = args.source_checkpoint if not args.smoke else None
    resume_epoch = 0
    if args.resume and source_ckpt is not None and source_ckpt.is_file():
        resume_epoch = int(torch.load(source_ckpt, map_location="cpu", weights_only=False).get("epoch", 0))
    layer4_should_be_unfrozen = (
        args.unfreeze_layer4_epoch == 0
        or (args.resume and resume_epoch >= args.unfreeze_layer4_epoch)
    )
    model = build_model(
        source_ckpt, unfreeze_layer4=layer4_should_be_unfrozen, is_resume=args.resume
    ).to(device)

    # ---- loss ----
    class_weights = torch.tensor(
        len(train_samples) / (3.0 * np.maximum(counts, 1)),
        dtype=torch.float32, device=device,
    )
    criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights).to(device)

    # ---- optimizer ----
    optimizer = torch.optim.AdamW(
        model.fc.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if layer4_should_be_unfrozen:
        optimizer.add_param_group(
            {"params": list(model.layer4.parameters()), "lr": args.backbone_lr}
        )
    
    # Load optimizer state if doing a true resume
    if args.resume and source_ckpt is not None and source_ckpt.is_file():
        ckpt = torch.load(source_ckpt, map_location="cpu", weights_only=False)
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("[INFO] Optimizer state loaded from checkpoint.")
            except Exception as e:
                print(f"[WARN] Could not load optimizer state: {e}")

    # ---- AMP ----
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # ---- training ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    start_epoch = 1
    best_f1 = -1.0
    if args.resume and source_ckpt is not None and source_ckpt.is_file():
        resume_checkpoint = torch.load(source_ckpt, map_location="cpu", weights_only=False)
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        if "val_metrics" in resume_checkpoint:
            best_f1 = float(resume_checkpoint["val_metrics"].get("macro_f1", -1.0))
        print(f"[INFO] resuming at epoch {start_epoch}; previous best_f1={best_f1:.4f}")

    for epoch in range(start_epoch, smoke_epochs + 1):
        # Progressive unfreeze
        if epoch == args.unfreeze_layer4_epoch and epoch > 0:
            unfreeze_layer4(model, args.backbone_lr, optimizer)

        train_loss, train_conf = run_epoch(model, train_loader, device, criterion, optimizer, use_amp, scaler)
        val_loss, val_conf = run_epoch(model, val_loader, device, criterion, None, use_amp, scaler)

        tm = confusion_metrics(train_conf)
        vm = confusion_metrics(val_conf)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train": tm, "val": vm}
        history.append(row)
        print(
            f"epoch={epoch}/{smoke_epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_macro_f1={vm['macro_f1']:.4f}  "
            f"val_acc={vm['accuracy']:.4f}",
            flush=True,
        )

        checkpoint_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "epoch": epoch,
            "classes": CLASSES,
            "frames": args.frames,
            "source_checkpoint": str(args.source_checkpoint),
            "val_metrics": vm,
        }
        torch.save(checkpoint_payload, args.output_dir / "latest.pth")
        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            torch.save(checkpoint_payload, args.output_dir / "best.pth")
            print(f"  [OK] saved best checkpoint (val_macro_f1={best_f1:.4f})")

    # Save metrics
    meta = {
        "device": str(device),
        "train_matches": len(train_matches),
        "val_matches": len(val_matches),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "class_counts": counts.tolist(),
        "classes": CLASSES,
        "args": {
            "epochs": smoke_epochs,
            "batch_size": args.batch_size,
            "frames": args.frames,
            "positives_per_match": smoke_pos,
            "negative_ratio": args.negative_ratio,
            "focal_gamma": args.focal_gamma,
            "learning_rate": args.learning_rate,
            "backbone_lr": args.backbone_lr,
            "unfreeze_layer4_epoch": args.unfreeze_layer4_epoch,
        },
        "history": history,
        "best_val_macro_f1": best_f1,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n[DONE] best val_macro_f1={best_f1:.4f}  output={args.output_dir}")


if __name__ == "__main__":
    main()
