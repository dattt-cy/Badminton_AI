"""Fine-tune R(2+1)D-18 on ShuttleSet stroke type and stroke side."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torchvision.models.video import r2plus1d_18
from torchvision.transforms import functional as vision_functional
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training.train_fine_badminton_rgb import confusion_metrics
from ai_classifier.datasets import decode_virtual_clip, load_fine_badminton_manifest


STROKE_CLASSES = [
    "serve", "clear", "smash", "drop", "net_shot", "lift", "drive", "net_attack"
]
SIDE_CLASSES = ["forehand", "backhand", "aroundhead"]


@dataclass(frozen=True)
class Record:
    sample_id: str
    video_path: Path
    start_frame: int
    end_frame: int
    coarse_label: str
    stroke_side: str
    player_side: str = "bottom"
    hit_frame: int = 0
    raw_label: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv")
    )
    parser.add_argument(
        "--fine-checkpoint",
        type=Path,
        default=Path("work_dirs/r2plus1d18_fine_badminton_8class_full/best.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work_dirs/r2plus1d18_shuttleset_multitask"),
    )
    parser.add_argument("--resume", type=Path, help="Resume model state from a multitask checkpoint.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=112)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--crop-hitter", action="store_true")
    parser.add_argument("--crop-padding", type=float, default=0.30)
    parser.add_argument(
        "--pose-model", type=Path, default=Path("models/checkpoints/pose/yolov8n-pose.pt")
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--side-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--fine-stroke-loss-weight", type=float, default=1.0,
        help="Stroke-loss multiplier for Fine-Badminton samples.",
    )
    parser.add_argument(
        "--shuttle-stroke-loss-weight", type=float, default=1.0,
        help="Stroke-loss multiplier for ShuttleSet samples.",
    )
    parser.add_argument(
        "--shuttle-stroke-exclude-raw-label", nargs="*", default=[],
        help="Keep side supervision but set stroke loss to zero for these raw labels.",
    )
    parser.add_argument("--shuttle-top-sample-boost", type=float, default=1.0)
    parser.add_argument("--shuttle-drive-sample-boost", type=float, default=1.0)
    parser.add_argument("--shuttle-forehand-sample-boost", type=float, default=1.0)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--unfreeze-layer4", action="store_true",
        help="Freeze the backbone except layer4; intended for crop/domain adaptation.",
    )
    parser.add_argument("--backbone-learning-rate", type=float, default=3e-6)
    parser.add_argument(
        "--reset-optimizer", action="store_true",
        help="Load model weights from --resume but start a fresh optimizer/scaler.",
    )
    parser.add_argument("--fine-dataset-root", type=Path)
    parser.add_argument(
        "--fine-train-manifest", type=Path,
        default=Path("data/manifests/fine_badminton_splits/train.csv"),
    )
    parser.add_argument(
        "--fine-val-manifest", type=Path,
        default=Path("data/manifests/fine_badminton_splits/val.csv"),
    )
    parser.add_argument("--fine-samples-per-class", type=int)
    parser.add_argument("--fine-val-samples-per-class", type=int)
    parser.add_argument("--fine-cache-dir", type=Path)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument(
        "--samples-per-side",
        type=int,
        help="Optional balanced cap per stroke-side class for smoke runs.",
    )
    parser.add_argument(
        "--val-samples-per-side", type=int,
        help="Optional balanced validation cap per stroke-side class.",
    )
    return parser.parse_args()


def windows_to_wsl(path: str) -> Path:
    match = __import__("re").match(r"^([A-Za-z]):[\\/](.*)$", path)
    if match and Path("/mnt").is_dir():
        drive, rest = match.groups()
        return Path("/mnt") / drive.lower() / Path(rest.replace("\\", "/"))
    return Path(path)


def load_records(path: Path, split: str) -> list[Record]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    return [
        Record(
            sample_id=row["sample_id"],
            video_path=windows_to_wsl(row["video_path"]),
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
            coarse_label=row["coarse_label"],
            stroke_side=row["stroke_side"],
            player_side=row["player_side"],
            hit_frame=int(row["hit_frame"]),
            raw_label=row.get("raw_label", ""),
        )
        for row in rows
    ]


def balanced_subset(records: list[Record], count: int | None, seed: int) -> list[Record]:
    if count is None:
        return records
    if count <= 0:
        raise ValueError("samples-per-side must be positive")
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.stroke_side].append(record)
    rng = random.Random(seed)
    selected = []
    for name in SIDE_CLASSES:
        candidates = sorted(groups[name], key=lambda item: item.sample_id)
        selected.extend(rng.sample(candidates, min(count, len(candidates))))
    rng.shuffle(selected)
    return selected


def select_hitter_box(result, player_side: str, width: int, height: int) -> np.ndarray | None:
    if result.boxes is None or result.keypoints is None:
        return None
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()
    keypoints = result.keypoints.data.detach().cpu().numpy()
    candidates = []
    for box, class_id, pose in zip(boxes, classes, keypoints):
        if int(class_id) != 0:
            continue
        visible = pose[:, 2] >= 0.25
        center = pose[visible, :2].mean(axis=0) if visible.any() else np.array(
            [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        )
        x, y = center
        min_x, max_x = (0.32, 0.68) if player_side == "top" else (0.22, 0.78)
        min_y = 0.25 if player_side == "top" else 0.42
        if not (min_x * width <= x <= max_x * width and min_y * height <= y <= 0.92 * height):
            continue
        if player_side == "top" and y >= 0.58 * height:
            continue
        candidates.append((float(y), box))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1] if player_side == "top" else candidates[-1][1]


def padded_square(box: np.ndarray, width: int, height: int, padding: float = 0.30) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(max(x2 - x1, y2 - y1) * (1.0 + 2.0 * padding), min(width, height) * 0.22)
    return (
        max(0, int(round(cx - side / 2))), max(0, int(round(cy - side * 0.58))),
        min(width, int(round(cx + side / 2))), min(height, int(round(cy + side * 0.42))),
    )


def detect_hitter_crop(
    record: Record, pose_model: YOLO, padding: float = 0.30
) -> tuple[int, int, int, int]:
    capture = cv2.VideoCapture(str(record.video_path))
    try:
        for offset in (0, -2, 2, -4, 4, -6, 6):
            frame_index = max(0, record.hit_frame + offset)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            result = pose_model.predict(frame, imgsz=960, conf=0.20, verbose=False)[0]
            height, width = frame.shape[:2]
            box = select_hitter_box(result, record.player_side, width, height)
            if box is not None:
                return padded_square(box, width, height, padding)
    finally:
        capture.release()
    if 'frame' not in locals():
        raise ValueError(f"Cannot decode frames for {record.sample_id}")
    # Rare low-resolution/occluded cases should not abort a multi-hour run.
    # Keep the relevant half-court and report the fallback in stderr.
    warnings.warn(f"Pose fallback to half-court crop for {record.sample_id}")
    side = int(round(min(width, height) * 0.62))
    cx = width // 2
    cy = int(round(height * (0.40 if record.player_side == "top" else 0.70)))
    return (
        max(0, cx - side // 2), max(0, cy - side // 2),
        min(width, cx + side // 2), min(height, cy + side // 2),
    )


def decode_clip(
    record: Record, frame_count: int, crop_box: tuple[int, int, int, int] | None = None,
    crop_size: int = 112,
) -> torch.Tensor:
    capture = cv2.VideoCapture(str(record.video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {record.video_path}")
        wanted = np.rint(
            np.linspace(record.start_frame, record.end_frame, frame_count)
        ).astype(np.int64)
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(wanted[0]))
        decoded: dict[int, np.ndarray] = {}
        wanted_set = set(int(value) for value in wanted)
        for frame_index in range(int(wanted[0]), int(wanted[-1]) + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(
                    f"Decode stopped at frame {frame_index}: {record.video_path}"
                )
            if frame_index in wanted_set:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if crop_box is not None:
                    x1, y1, x2, y2 = crop_box
                    rgb = rgb[y1:y2, x1:x2]
                decoded[frame_index] = rgb
    finally:
        capture.release()
    missing = wanted_set.difference(decoded)
    if missing:
        raise ValueError(f"Missing frames {sorted(missing)} for {record.sample_id}")
    tensor = torch.from_numpy(
        np.stack([decoded[int(index)] for index in wanted]).copy()
    ).permute(0, 3, 1, 2)
    if crop_box is not None:
        return vision_functional.resize(tensor, [crop_size, crop_size], antialias=False).contiguous()
    tensor = vision_functional.resize(tensor, [128, 171], antialias=False)
    return vision_functional.center_crop(tensor, [112, 112]).contiguous()


class ShuttleSetRGBDataset(Dataset):
    def __init__(
        self, records: list[Record], frame_count: int, training: bool,
        *, crop_hitter: bool = False, pose_model: YOLO | None = None, cache_dir: Path | None = None,
        crop_padding: float = 0.30, crop_size: int = 112,
    ) -> None:
        self.records = records
        self.frame_count = frame_count
        self.training = training
        self.crop_hitter = crop_hitter
        self.pose_model = pose_model
        self.cache_dir = cache_dir
        self.crop_padding = crop_padding
        self.crop_size = crop_size
        self.cached_created = 0
        self.stroke_to_id = {name: index for index, name in enumerate(STROKE_CLASSES)}
        self.side_to_id = {name: index for index, name in enumerate(SIDE_CLASSES)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        cache_path = (
            self.cache_dir / (
                f"frames_{self.frame_count}" if self.crop_size == 112
                else f"frames_{self.frame_count}_crop_{self.crop_size}"
            ) / f"{record.sample_id}.npy"
            if self.cache_dir is not None else None
        )
        if cache_path is not None and cache_path.is_file():
            tensor = torch.from_numpy(np.load(cache_path))
        else:
            crop_box = None
            if self.crop_hitter:
                if self.pose_model is None:
                    raise RuntimeError("crop-hitter requires a pose model")
                crop_box = detect_hitter_crop(record, self.pose_model, self.crop_padding)
            tensor = decode_clip(record, self.frame_count, crop_box, self.crop_size)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp.npy")
                np.save(temporary, tensor.numpy())
                temporary.replace(cache_path)
                self.cached_created += 1
                if self.cached_created % 50 == 0:
                    print(
                        f"cache_progress split={'train' if self.training else 'val'} "
                        f"created={self.cached_created}", flush=True
                    )
        tensor = tensor.to(torch.float32).div_(255.0)
        mean = tensor.new_tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
        std = tensor.new_tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
        tensor = ((tensor - mean) / std).permute(1, 0, 2, 3)
        side_id = self.side_to_id[record.stroke_side]
        if self.training and torch.rand(()) < 0.5:
            tensor = torch.flip(tensor, dims=(-1,))
            # Swap forehand(0) and backhand(1), assuming they are 0 and 1
            if side_id in (0, 1):
                side_id = 1 - side_id

        return (
            tensor,
            self.stroke_to_id[record.coarse_label],
            side_id,
            record.sample_id,
        )


class FineJointDataset(Dataset):
    """Fine-Badminton samples supervise stroke only; side uses ignore_index."""

    def __init__(
        self, records, dataset_root: Path, frame_count: int, training: bool,
        cache_dir: Path | None = None, crop_size: int = 112,
    ) -> None:
        self.records = records
        self.dataset_root = dataset_root
        self.frame_count = frame_count
        self.training = training
        self.cache_dir = cache_dir
        self.crop_size = crop_size
        self.stroke_to_id = {name: index for index, name in enumerate(STROKE_CLASSES)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        cache_path = (
            self.cache_dir / (
                f"frames_{self.frame_count}" if self.crop_size == 112
                else f"frames_{self.frame_count}_crop_{self.crop_size}"
            ) / f"{record.sample_id}.npy"
            if self.cache_dir is not None else None
        )
        if cache_path is not None and cache_path.is_file():
            tensor = torch.from_numpy(np.load(cache_path))
        else:
            frames = decode_virtual_clip(record, self.dataset_root, frame_count=self.frame_count)
            tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)
            resize_short = max(self.crop_size, round(self.crop_size * 128 / 112))
            resize_long = max(self.crop_size, round(self.crop_size * 171 / 112))
            tensor = vision_functional.resize(tensor, [resize_short, resize_long], antialias=False)
            tensor = vision_functional.center_crop(tensor, [self.crop_size, self.crop_size]).contiguous()
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp.npy")
                np.save(temporary, tensor.numpy())
                temporary.replace(cache_path)
        tensor = tensor.to(torch.float32).div_(255.0)
        mean = tensor.new_tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
        std = tensor.new_tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
        tensor = ((tensor - mean) / std).permute(1, 0, 2, 3)
        if self.training and torch.rand(()) < 0.5:
            tensor = torch.flip(tensor, dims=(-1,))
        return tensor, self.stroke_to_id[record.coarse_label], -100, f"fine:{record.sample_id}"


def balanced_fine_subset(records, count: int | None, seed: int):
    if count is None:
        return records
    grouped = defaultdict(list)
    for record in records:
        grouped[record.coarse_label].append(record)
    rng = random.Random(seed)
    selected = []
    for name in STROKE_CLASSES:
        candidates = sorted(grouped[name], key=lambda item: item.sample_id)
        selected.extend(rng.sample(candidates, min(count, len(candidates))))
    rng.shuffle(selected)
    return selected


class MultiTaskR2Plus1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = r2plus1d_18(weights=None)
        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.stroke_head = nn.Linear(features, len(STROKE_CLASSES))
        self.side_head = nn.Linear(features, len(SIDE_CLASSES))

    def forward(self, clips: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(clips)
        return self.stroke_head(features), self.side_head(features)


def load_fine_checkpoint(model: MultiTaskR2Plus1D, path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if list(checkpoint["classes"]) != STROKE_CLASSES:
        raise ValueError("Fine-Badminton checkpoint classes do not match")
    state = checkpoint["model"]
    backbone_state = {
        key: value for key, value in state.items() if not key.startswith("fc.")
    }
    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
    if missing or unexpected:
        raise ValueError(f"Unexpected checkpoint keys: missing={missing}, unexpected={unexpected}")
    model.stroke_head.load_state_dict(
        {"weight": state["fc.weight"], "bias": state["fc.bias"]}
    )
    return int(checkpoint["epoch"])


def class_weights(records: list[Record], classes: list[str], attribute: str) -> torch.Tensor:
    counts = Counter(getattr(record, attribute) for record in records)
    return torch.tensor(
        [
            len(records) / (len(classes) * counts[name]) if counts[name] else 0.0
            for name in classes
        ],
        dtype=torch.float32,
    )


def source_weighted_stroke_loss(
    losses: torch.Tensor,
    sample_ids: list[str] | tuple[str, ...],
    *,
    fine_weight: float,
    shuttle_weight: float,
    zero_weight_sample_ids: set[str] | None = None,
) -> torch.Tensor:
    """Apply absolute per-source weights without cancelling homogeneous batches."""
    if fine_weight < 0 or shuttle_weight < 0:
        raise ValueError("Source loss weights must be non-negative")
    zero_weight_sample_ids = zero_weight_sample_ids or set()
    weights = losses.new_tensor([
        0.0 if str(sample_id) in zero_weight_sample_ids else (
            fine_weight if str(sample_id).startswith("fine:") else shuttle_weight
        )
        for sample_id in sample_ids
    ])
    return (losses * weights).mean()


def run_epoch(
    model: MultiTaskR2Plus1D,
    loader: DataLoader,
    stroke_criterion: nn.Module,
    side_criterion: nn.Module,
    device: torch.device,
    *,
    side_loss_weight: float,
    fine_stroke_loss_weight: float,
    shuttle_stroke_loss_weight: float,
    zero_stroke_sample_ids: set[str],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    stroke_confusion = torch.zeros(len(STROKE_CLASSES), len(STROKE_CLASSES), dtype=torch.int64)
    side_confusion = torch.zeros(len(SIDE_CLASSES), len(SIDE_CLASSES), dtype=torch.int64)
    total_loss = 0.0
    sample_count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for clips, stroke_labels, side_labels, sample_ids in loader:
            clips = clips.to(device)
            stroke_labels = stroke_labels.to(device)
            side_labels = side_labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                stroke_logits, side_logits = model(clips)
                stroke_losses = stroke_criterion(stroke_logits, stroke_labels)
                loss = source_weighted_stroke_loss(
                    stroke_losses,
                    sample_ids,
                    fine_weight=fine_stroke_loss_weight,
                    shuttle_weight=shuttle_stroke_loss_weight,
                    zero_weight_sample_ids=zero_stroke_sample_ids,
                )
                side_mask = side_labels != -100
                if side_mask.any():
                    loss = loss + side_loss_weight * side_criterion(
                        side_logits[side_mask], side_labels[side_mask]
                    )
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            for truth, prediction in zip(stroke_labels.cpu(), stroke_logits.argmax(1).cpu()):
                stroke_confusion[int(truth), int(prediction)] += 1
            for truth, prediction in zip(side_labels.cpu(), side_logits.argmax(1).cpu()):
                if int(truth) != -100:
                    side_confusion[int(truth), int(prediction)] += 1
            batch = clips.shape[0]
            total_loss += float(loss.detach()) * batch
            sample_count += batch
    return total_loss / max(sample_count, 1), stroke_confusion, side_confusion


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_records = balanced_subset(load_records(args.manifest, "train"), args.samples_per_side, args.seed)
    val_cap = args.val_samples_per_side if args.val_samples_per_side is not None else args.samples_per_side
    val_records = balanced_subset(load_records(args.manifest, "val"), val_cap, args.seed + 1)
    raw_label_aliases = {"soft_drive": "小平球"}
    excluded_raw_labels = {
        raw_label_aliases.get(value, value)
        for value in args.shuttle_stroke_exclude_raw_label
    }
    train_zero_stroke_ids = {
        record.sample_id for record in train_records if record.raw_label in excluded_raw_labels
    }
    val_zero_stroke_ids = {
        record.sample_id for record in val_records if record.raw_label in excluded_raw_labels
    }
    pose_model = YOLO(str(args.pose_model)) if args.crop_hitter else None
    shuttle_train_dataset = ShuttleSetRGBDataset(
        train_records, args.frames, training=True, crop_hitter=args.crop_hitter,
        pose_model=pose_model, cache_dir=args.cache_dir, crop_padding=args.crop_padding,
        crop_size=args.crop_size,
    )
    shuttle_val_dataset = ShuttleSetRGBDataset(
        val_records, args.frames, training=False, crop_hitter=args.crop_hitter,
        pose_model=pose_model, cache_dir=args.cache_dir, crop_padding=args.crop_padding,
        crop_size=args.crop_size,
    )
    fine_train_records = []
    fine_val_records = []
    if args.fine_dataset_root is not None:
        fine_train_records = balanced_fine_subset(
            load_fine_badminton_manifest(args.fine_train_manifest),
            args.fine_samples_per_class, args.seed,
        )
        fine_val_records = balanced_fine_subset(
            load_fine_badminton_manifest(args.fine_val_manifest),
            args.fine_val_samples_per_class, args.seed + 1,
        )
        fine_train_dataset = FineJointDataset(
            fine_train_records, args.fine_dataset_root, args.frames, True, args.fine_cache_dir,
            args.crop_size,
        )
        fine_val_dataset = FineJointDataset(
            fine_val_records, args.fine_dataset_root, args.frames, False, args.fine_cache_dir,
            args.crop_size,
        )
        train_dataset = ConcatDataset([shuttle_train_dataset, fine_train_dataset])
        val_dataset = ConcatDataset([shuttle_val_dataset, fine_val_dataset])
        # Draw both sources equally even when their selected subset sizes differ.
        shuttle_multipliers = []
        for record in train_records:
            multiplier = 1.0
            if record.player_side == "top":
                multiplier *= args.shuttle_top_sample_boost
            if record.coarse_label == "drive":
                multiplier *= args.shuttle_drive_sample_boost
            if record.stroke_side == "forehand":
                multiplier *= args.shuttle_forehand_sample_boost
            shuttle_multipliers.append(multiplier)
        shuttle_total = sum(shuttle_multipliers)
        train_weights = (
            [0.5 * value / shuttle_total for value in shuttle_multipliers]
            + [0.5 / len(fine_train_dataset)] * len(fine_train_dataset)
        )
        sampler = WeightedRandomSampler(
            train_weights, num_samples=len(train_dataset), replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers
        )
    else:
        train_dataset = shuttle_train_dataset
        val_dataset = shuttle_val_dataset
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
        )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = MultiTaskR2Plus1D()
    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if list(resume_checkpoint["stroke_classes"]) != STROKE_CLASSES:
            raise ValueError("Resume checkpoint stroke classes do not match")
        if list(resume_checkpoint["side_classes"]) != SIDE_CLASSES:
            raise ValueError("Resume checkpoint side classes do not match")
        model.load_state_dict(resume_checkpoint["model"])
        source_epoch = int(resume_checkpoint.get("source_epoch", 0))
    else:
        source_epoch = load_fine_checkpoint(model, args.fine_checkpoint)
    if args.freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    if args.unfreeze_layer4:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        for parameter in model.backbone.layer4.parameters():
            parameter.requires_grad = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    stroke_weights = class_weights(
        [record for record in train_records if record.sample_id not in train_zero_stroke_ids]
        + list(fine_train_records), STROKE_CLASSES, "coarse_label"
    )
    # Boost hard classes once. Keep these multipliers explicit so repeated
    # fine-tuning edits cannot accidentally compound a class weight.
    lift_idx = STROKE_CLASSES.index("lift")
    clear_idx = STROKE_CLASSES.index("clear")
    drive_idx = STROKE_CLASSES.index("drive")
    stroke_weights[lift_idx] *= 2.2
    stroke_weights[drive_idx] *= 1.4
    stroke_weights[clear_idx] *= 1.8
    print(f"[INFO] Custom Stroke Class Weights:")
    for cls_name, w in zip(STROKE_CLASSES, stroke_weights):
        print(f"  {cls_name:<12}: {float(w):.3f}")
    stroke_criterion = nn.CrossEntropyLoss(
        weight=stroke_weights.to(device), reduction="none"
    )
    side_criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_records, SIDE_CLASSES, "stroke_side").to(device)
    )
    if args.unfreeze_layer4:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.layer4.parameters(), "lr": args.backbone_learning_rate},
                {"params": model.stroke_head.parameters(), "lr": args.learning_rate},
                {"params": model.side_head.parameters(), "lr": args.learning_rate},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate, weight_decay=args.weight_decay,
        )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if resume_checkpoint is not None and "optimizer" in resume_checkpoint and not args.reset_optimizer:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
    if resume_checkpoint is not None and "scaler" in resume_checkpoint and not args.reset_optimizer:
        scaler.load_state_dict(resume_checkpoint["scaler"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    start_epoch = int(resume_checkpoint["epoch"]) + 1 if resume_checkpoint is not None else 1
    best_score = (
        (float(resume_checkpoint["val_stroke_metrics"]["macro_f1"])
         + float(resume_checkpoint["val_side_metrics"]["macro_f1"])) / 2
        if resume_checkpoint is not None and "val_stroke_metrics" in resume_checkpoint and float(resume_checkpoint["val_side_metrics"]["macro_f1"]) < 0.99
        else -1.0
    )
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_stroke, train_side = run_epoch(
                model, train_loader, stroke_criterion, side_criterion, device,
                side_loss_weight=args.side_loss_weight,
                fine_stroke_loss_weight=args.fine_stroke_loss_weight,
                shuttle_stroke_loss_weight=args.shuttle_stroke_loss_weight,
                zero_stroke_sample_ids=train_zero_stroke_ids,
                optimizer=optimizer, scaler=scaler, use_amp=use_amp,
            )
            val_loss, val_stroke, val_side = run_epoch(
                model, val_loader, stroke_criterion, side_criterion, device,
                side_loss_weight=args.side_loss_weight,
                fine_stroke_loss_weight=args.fine_stroke_loss_weight,
                shuttle_stroke_loss_weight=args.shuttle_stroke_loss_weight,
                zero_stroke_sample_ids=val_zero_stroke_ids,
                optimizer=None, scaler=scaler, use_amp=use_amp,
            )
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_stroke": confusion_metrics(train_stroke),
                "train_side": confusion_metrics(train_side),
                "val_stroke": confusion_metrics(val_stroke),
                "val_side": confusion_metrics(val_side),
            }
            history.append(metrics)
            score = (metrics["val_stroke"]["macro_f1"] + metrics["val_side"]["macro_f1"]) / 2
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_stroke_f1={metrics['val_stroke']['macro_f1']:.4f} "
                f"val_side_f1={metrics['val_side']['macro_f1']:.4f}", flush=True
            )
            # Always save latest checkpoint after every epoch
            state_dict_payload = {
                "model": model.state_dict(), "epoch": epoch,
                "source_checkpoint": str(args.fine_checkpoint), "source_epoch": source_epoch,
                "stroke_classes": STROKE_CLASSES, "side_classes": SIDE_CLASSES,
                "frames": args.frames, "crop_size": args.crop_size,
                "val_stroke_metrics": metrics["val_stroke"],
                "val_side_metrics": metrics["val_side"],
                "crop_hitter": args.crop_hitter, "crop_padding": args.crop_padding,
                "unfreeze_layer4": args.unfreeze_layer4,
                "fine_stroke_loss_weight": args.fine_stroke_loss_weight,
                "shuttle_stroke_loss_weight": args.shuttle_stroke_loss_weight,
                "shuttle_stroke_exclude_raw_label": args.shuttle_stroke_exclude_raw_label,
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            }
            torch.save(state_dict_payload, args.output_dir / "latest.pth")

            if score > best_score:
                best_score = score
                torch.save(state_dict_payload, args.output_dir / "best.pth")
                print(f"[INFO] New best model saved to {args.output_dir / 'best.pth'} (score={score:.4f})", flush=True)

            (args.output_dir / "train.log").write_text(
                "\n".join(
                    f"epoch={row['epoch']} train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
                    f"val_stroke_f1={row['val_stroke']['macro_f1']:.4f} val_side_f1={row['val_side']['macro_f1']:.4f}"
                    for row in history
                ) + "\n", encoding="utf-8"
            )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user! Gracefully saving interrupted.pth...", flush=True)
        torch.save(
            {
                "model": model.state_dict(), "epoch": epoch if 'epoch' in locals() else start_epoch,
                "source_checkpoint": str(args.fine_checkpoint), "source_epoch": source_epoch,
                "stroke_classes": STROKE_CLASSES, "side_classes": SIDE_CLASSES,
                "frames": args.frames, "crop_size": args.crop_size,
                "crop_hitter": args.crop_hitter, "crop_padding": args.crop_padding,
                "unfreeze_layer4": args.unfreeze_layer4,
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            },
            args.output_dir / "interrupted.pth",
        )
        print(f"[INFO] Successfully saved {args.output_dir / 'interrupted.pth'}", flush=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({
            "device": str(device),
            "shuttle_train_samples": len(train_records), "shuttle_val_samples": len(val_records),
            "fine_train_samples": len(fine_train_records), "fine_val_samples": len(fine_val_records),
            "crop_hitter": args.crop_hitter, "crop_padding": args.crop_padding,
            "crop_size": args.crop_size,
            "unfreeze_layer4": args.unfreeze_layer4,
            "fine_stroke_loss_weight": args.fine_stroke_loss_weight,
            "shuttle_stroke_loss_weight": args.shuttle_stroke_loss_weight,
            "shuttle_stroke_exclude_raw_label": args.shuttle_stroke_exclude_raw_label,
            "shuttle_top_sample_boost": args.shuttle_top_sample_boost,
            "shuttle_drive_sample_boost": args.shuttle_drive_sample_boost,
            "shuttle_forehand_sample_boost": args.shuttle_forehand_sample_boost,
            "cache_dir": str(args.cache_dir) if args.cache_dir else None,
            "history": history,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
