"""Train a lightweight RGB + BST-modality late-fusion pilot."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training.train_fine_badminton_rgb import confusion_metrics
from scripts.training.train_shuttleset_rgb_multitask import MultiTaskR2Plus1D, SIDE_CLASSES, STROKE_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_npy.csv"))
    parser.add_argument("--npy-root", type=Path, default=Path(r"C:\Users\ADMIN\Downloads\dataset_npy_between_2_hits_with_max_limits\dataset_npy_between_2_hits_with_max_limits"))
    parser.add_argument("--rgb-cache", type=Path, default=Path(r"C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_v3_full\frames_16"))
    parser.add_argument("--rgb-checkpoint", type=Path, default=Path("work_dirs/r2plus1d18_shuttleset_mixed_crop_full_e8_e10_b4/best.pth"))
    parser.add_argument("--feature-cache", type=Path, default=Path("work_dirs/shuttleset_fusion_pilot/features.npz"))
    parser.add_argument("--source-feature-cache", type=Path, help="Reuse RGB embeddings by sample_id.")
    parser.add_argument("--output-dir", type=Path, default=Path("work_dirs/shuttleset_fusion_pilot"))
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--eval-per-class", type=int, default=75)
    parser.add_argument(
        "--full-data", action="store_true",
        help="Use every valid sample with an existing RGB cache entry.",
    )
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--target-aligned", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def select_rows(path: Path, cache: Path, train_count: int, eval_count: int, seed: int, full_data: bool = False) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["valid_coarse_label"] == "True" and (cache / f"{row['sample_id']}.npy").is_file()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["coarse_label"])].append(row)
    if full_data:
        return rows
    rng = random.Random(seed)
    selected = []
    for split in ("train", "val", "test"):
        count = train_count if split == "train" else eval_count
        for label in STROKE_CLASSES:
            candidates = sorted(grouped[(split, label)], key=lambda row: row["sample_id"])
            if len(candidates) < count:
                raise ValueError(f"Not enough cached samples for {split}/{label}: {len(candidates)} < {count}")
            selected.extend(rng.sample(candidates, count))
    return selected


def resample(array: np.ndarray, length: int) -> np.ndarray:
    values = np.nan_to_num(array.astype(np.float32), nan=0.0).reshape(len(array), -1)
    if len(values) == length:
        return values
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, length)
    return np.stack([np.interp(target, source, values[:, index]) for index in range(values.shape[1])], axis=1).astype(np.float32)


def structured_feature_arrays(joints: np.ndarray, position: np.ndarray, shuttle: np.ndarray, length: int) -> np.ndarray:
    arrays = [resample(array, length) for array in (joints, position, shuttle)]
    sequence = np.concatenate(arrays, axis=1)
    velocity = np.diff(sequence, axis=0, prepend=sequence[:1])
    return np.concatenate([sequence.ravel(), velocity.mean(0), velocity.std(0)]).astype(np.float32)


def target_aligned_feature_arrays(
    joints: np.ndarray, position: np.ndarray, shuttle: np.ndarray,
    target_index: int, length: int, before: int = 15, after: int = 30,
) -> np.ndarray:
    window_length = before + 1 + after
    arrays = []
    for array in (joints, position, shuttle):
        # A few annotations point outside the exported between-hit array.
        # Clamp the target and copy only the overlapping source/destination span.
        safe_target = min(max(int(target_index), 0), max(len(array) - 1, 0))
        output = np.zeros((window_length, *array.shape[1:]), dtype=np.float32)
        source_start = max(0, safe_target - before)
        source_end = min(len(array), safe_target + after + 1)
        destination_start = before - min(before, safe_target)
        span = min(source_end - source_start, window_length - destination_start)
        if span > 0:
            output[destination_start:destination_start + span] = array[source_start:source_start + span]
        arrays.append(output)
    modalities = [resample(array, length) for array in arrays]
    target = np.zeros((window_length, 1), dtype=np.float32)
    target[before] = 1.0
    modalities.append(resample(target, length))
    sequence = np.concatenate(modalities, axis=1)
    velocity = np.diff(sequence[:, :-1], axis=0, prepend=sequence[:1, :-1])
    return np.concatenate([sequence.ravel(), velocity.mean(0), velocity.std(0)]).astype(np.float32)


def structured_feature(row: dict, root: Path, length: int) -> np.ndarray:
    return structured_feature_arrays(
        np.load(root / row["joints_path"]), np.load(root / row["position_path"]),
        np.load(root / row["shuttle_path"]), length,
    )


def normalized_rgb(path: Path) -> torch.Tensor:
    tensor = torch.from_numpy(np.load(path)).float().div_(255.0)
    mean = tensor.new_tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
    std = tensor.new_tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
    return ((tensor - mean) / std).permute(1, 0, 2, 3)


def target_indices(rows: list[dict], fps: int = 30) -> dict[str, int]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["match_id"], row["set_id"], row["rally"])].append(row)
    output = {}
    for group in grouped.values():
        ordered = sorted(group, key=lambda row: int(row["ball_round"]))
        for index, row in enumerate(ordered):
            current = int(float(row["hit_frame"]))
            previous = int(float(ordered[index - 1]["hit_frame"])) if index else current - fps // 2
            start = max(previous, current - fps * 3 // 2)
            output[row["sample_id"]] = current - start
    return output


def extract_features(rows: list[dict], args: argparse.Namespace) -> dict[str, np.ndarray]:
    checkpoint = torch.load(args.rgb_checkpoint, map_location="cpu", weights_only=False)
    model = MultiTaskR2Plus1D()
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    source_rgb = None
    if args.source_feature_cache:
        source = np.load(args.source_feature_cache)
        source_rgb = {str(sample_id): feature for sample_id, feature in zip(source["sample_id"], source["rgb"])}
    aligned_indices = target_indices(rows) if args.target_aligned else {}
    rgb_features, structured = [], []
    batch, batch_rows = [], []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            if source_rgb is None:
                batch.append(normalized_rgb(args.rgb_cache / f"{row['sample_id']}.npy"))
            batch_rows.append(row)
            if len(batch_rows) == 4 or index == len(rows) - 1:
                if source_rgb is None:
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        embedding = model.backbone(torch.stack(batch).to(device)).float().cpu().numpy()
                else:
                    embedding = np.stack([source_rgb[item["sample_id"]] for item in batch_rows])
                rgb_features.extend(embedding)
                for item in batch_rows:
                    arrays = [np.load(args.npy_root / item[key]) for key in ("joints_path", "position_path", "shuttle_path")]
                    feature = (
                        target_aligned_feature_arrays(*arrays, aligned_indices[item["sample_id"]], args.sequence_length)
                        if args.target_aligned else structured_feature_arrays(*arrays, args.sequence_length)
                    )
                    structured.append(feature)
                batch, batch_rows = [], []
                if len(rgb_features) % 400 == 0:
                    print(f"feature_progress={len(rgb_features)}/{len(rows)}", flush=True)
    return {
        "rgb": np.asarray(rgb_features, dtype=np.float32),
        "structured": np.asarray(structured, dtype=np.float32),
        "stroke": np.asarray([STROKE_CLASSES.index(row["coarse_label"]) for row in rows], dtype=np.int64),
        "side": np.asarray([SIDE_CLASSES.index(row["stroke_side"]) for row in rows], dtype=np.int64),
        "split": np.asarray([row["split"] for row in rows]),
        "sample_id": np.asarray([row["sample_id"] for row in rows]),
    }


class FusionHead(nn.Module):
    def __init__(self, rgb_dim: int, structured_dim: int) -> None:
        super().__init__()
        self.structured = nn.Sequential(nn.LayerNorm(structured_dim), nn.Linear(structured_dim, 256), nn.GELU(), nn.Dropout(0.25))
        self.fusion = nn.Sequential(nn.LayerNorm(rgb_dim + 256), nn.Linear(rgb_dim + 256, 256), nn.GELU(), nn.Dropout(0.25))
        self.stroke = nn.Linear(256, len(STROKE_CLASSES))
        self.side = nn.Linear(256, len(SIDE_CLASSES))

    def forward(self, rgb, structured):
        hidden = self.fusion(torch.cat([rgb, self.structured(structured)], 1))
        return self.stroke(hidden), self.side(hidden)


def evaluate(model, rgb, structured, stroke, side, indices, device) -> tuple[dict, dict]:
    model.eval()
    stroke_confusion = torch.zeros(8, 8, dtype=torch.int64)
    side_confusion = torch.zeros(3, 3, dtype=torch.int64)
    with torch.inference_mode():
        for start in range(0, len(indices), 128):
            ids = indices[start:start + 128]
            stroke_logits, side_logits = model(rgb[ids].to(device), structured[ids].to(device))
            for truth, prediction in zip(stroke[ids], stroke_logits.argmax(1).cpu()): stroke_confusion[int(truth), int(prediction)] += 1
            for truth, prediction in zip(side[ids], side_logits.argmax(1).cpu()): side_confusion[int(truth), int(prediction)] += 1
    return confusion_metrics(stroke_confusion), confusion_metrics(side_confusion)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.feature_cache.is_file():
        loaded = np.load(args.feature_cache)
        data = {key: loaded[key] for key in loaded.files}
    else:
        rows = select_rows(
            args.manifest, args.rgb_cache, args.train_per_class,
            args.eval_per_class, args.seed, args.full_data,
        )
        data = extract_features(rows, args)
        args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.feature_cache, **data)
    rgb = torch.from_numpy(data["rgb"]).float()
    structured_np = data["structured"].astype(np.float32)
    train_mask = data["split"] == "train"
    mean, std = structured_np[train_mask].mean(0), structured_np[train_mask].std(0)
    structured = torch.from_numpy((structured_np - mean) / np.maximum(std, 1e-5)).float()
    stroke, side = torch.from_numpy(data["stroke"]), torch.from_numpy(data["side"])
    indices = {name: np.flatnonzero(data["split"] == name) for name in ("train", "val", "test")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionHead(rgb.shape[1], structured.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    def weights(labels: torch.Tensor, ids: np.ndarray, classes: int) -> torch.Tensor:
        counts = torch.bincount(labels[ids], minlength=classes).float()
        return (len(ids) / (classes * counts.clamp_min(1))).to(device)
    stroke_loss = nn.CrossEntropyLoss(weight=weights(stroke, indices["train"], 8))
    side_loss = nn.CrossEntropyLoss(weight=weights(side, indices["train"], 3))
    rng = np.random.default_rng(args.seed)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(indices["train"])
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            stroke_logits, side_logits = model(rgb[ids].to(device), structured[ids].to(device))
            loss = stroke_loss(stroke_logits, stroke[ids].to(device)) + side_loss(side_logits, side[ids].to(device))
            loss.backward(); optimizer.step()
        val_stroke, val_side = evaluate(model, rgb, structured, stroke, side, indices["val"], device)
        history.append({"epoch": epoch, "val_stroke": val_stroke, "val_side": val_side})
        score = val_stroke["macro_f1"] + val_side["macro_f1"]
        print(f"epoch={epoch} val_stroke_f1={val_stroke['macro_f1']:.4f} val_side_f1={val_side['macro_f1']:.4f}", flush=True)
        if score > best:
            best = score
            torch.save({
                "model": model.state_dict(), "structured_mean": mean, "structured_std": std,
                "epoch": epoch, "target_aligned": args.target_aligned,
                "sequence_length": args.sequence_length, "target_before": 15, "target_after": 30,
            }, args.output_dir / "best.pth")
    best_checkpoint = torch.load(args.output_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    test_stroke, test_side = evaluate(model, rgb, structured, stroke, side, indices["test"], device)
    # Frozen RGB heads on the identical samples are the fair baseline.
    checkpoint = torch.load(args.rgb_checkpoint, map_location="cpu", weights_only=False)
    rgb_stroke_logits = rgb @ checkpoint["model"]["stroke_head.weight"].T + checkpoint["model"]["stroke_head.bias"]
    rgb_side_logits = rgb @ checkpoint["model"]["side_head.weight"].T + checkpoint["model"]["side_head.bias"]
    class LinearHeads(nn.Module):
        def forward(self, r, _s): return rgb_stroke_logits.to(device), rgb_side_logits.to(device)
    # Evaluate RGB arrays directly without coupling to the fusion model.
    def array_metrics(logits, labels, ids, classes):
        confusion = torch.zeros(classes, classes, dtype=torch.int64)
        for truth, prediction in zip(labels[ids], logits[ids].argmax(1)): confusion[int(truth), int(prediction)] += 1
        return confusion_metrics(confusion)
    result = {
        "best_epoch": int(best_checkpoint["epoch"]), "samples": {key: len(value) for key, value in indices.items()},
        "fusion_test_stroke": test_stroke, "fusion_test_side": test_side,
        "rgb_test_stroke": array_metrics(rgb_stroke_logits, stroke, indices["test"], 8),
        "rgb_test_side": array_metrics(rgb_side_logits, side, indices["test"], 3),
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"test fusion_stroke_f1={test_stroke['macro_f1']:.4f} rgb_stroke_f1={result['rgb_test_stroke']['macro_f1']:.4f} fusion_side_f1={test_side['macro_f1']:.4f} rgb_side_f1={result['rgb_test_side']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
