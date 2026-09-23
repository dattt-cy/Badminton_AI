"""Training and evaluation script for Contact-Aware Reliability-Gated Multimodal Fusion (CR-Gated Fusion)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.models import CRGatedFusionModel, SIDE_CLASSES, STROKE_CLASSES
from scripts.training.train_fine_badminton_rgb import confusion_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("work_dirs/cr_gated_fusion/sequence_features.npz"),
        help="Path to sequence_features.npz containing [T, D] structured modalities.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work_dirs/cr_gated_fusion"),
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "no_gate", "no_contact", "no_temporal"],
        help="Ablation mode: full, no_gate, no_contact, or no_temporal.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--drive-boost", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260923)
    return parser.parse_args()


def load_dataset_splits(data_path: Path):
    data = np.load(data_path)
    rgb = data["rgb"]
    pose = data["pose"]
    court = data["court"]
    shuttle = data["shuttle"]
    contact = data["contact_dist"]
    quality = data["quality"]
    stroke = data["stroke"]
    side = data["side"]
    splits = data["split"]

    datasets = {}
    for split_name in ("train", "val", "test"):
        mask = splits == split_name
        datasets[split_name] = TensorDataset(
            torch.from_numpy(rgb[mask]),
            torch.from_numpy(pose[mask]),
            torch.from_numpy(court[mask]),
            torch.from_numpy(shuttle[mask]),
            torch.from_numpy(contact[mask]),
            torch.from_numpy(quality[mask]),
            torch.from_numpy(stroke[mask]),
            torch.from_numpy(side[mask]),
        )
    return datasets


def evaluate_split(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    stroke_preds, stroke_targets = [], []
    side_preds, side_targets = [], []
    all_gate_weights = []

    with torch.no_grad():
        for rgb, pose, court, shuttle, contact, quality, stroke_gt, side_gt in loader:
            rgb = rgb.to(device)
            pose = pose.to(device)
            court = court.to(device)
            shuttle = shuttle.to(device)
            contact = contact.to(device)
            quality = quality.to(device)

            stroke_logits, side_logits, gates = model(rgb, pose, court, shuttle, contact, quality)
            stroke_preds.append(stroke_logits.argmax(dim=-1).cpu())
            stroke_targets.append(stroke_gt)
            side_preds.append(side_logits.argmax(dim=-1).cpu())
            side_targets.append(side_gt)

            if gates is not None:
                all_gate_weights.append(gates.cpu())

    s_preds = torch.cat(stroke_preds)
    s_targets = torch.cat(stroke_targets)
    side_p = torch.cat(side_preds)
    side_t = torch.cat(side_targets)

    # Stroke confusion matrix
    stroke_cm = torch.zeros(len(STROKE_CLASSES), len(STROKE_CLASSES), dtype=torch.int64)
    for t, p in zip(s_targets, s_preds):
        stroke_cm[int(t), int(p)] += 1
    stroke_metrics = confusion_metrics(stroke_cm)

    # Side confusion matrix
    side_cm = torch.zeros(len(SIDE_CLASSES), len(SIDE_CLASSES), dtype=torch.int64)
    for t, p in zip(side_t, side_p):
        side_cm[int(t), int(p)] += 1
    side_metrics = confusion_metrics(side_cm)

    mean_gates = None
    if all_gate_weights:
        mean_gates = torch.cat(all_gate_weights).mean(dim=0).tolist()

    return stroke_metrics, side_metrics, stroke_cm, mean_gates


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = args.output_dir / f"cr_gated_{args.ablation}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("HUAN LUYEN CONTACT-AWARE RELIABILITY-GATED MULTIMODAL FUSION")
    print(f"Che do ablation : {args.ablation.upper()}")
    print(f"Tap du lieu     : {args.features}")
    print(f"Thu muc output  : {run_dir}")
    print("=" * 65)

    datasets = load_dataset_splits(args.features)
    print(f"Loaded splits: Train={len(datasets['train'])}, Val={len(datasets['val'])}, Test={len(datasets['test'])}")

    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False)

    # Calculate class weights with drive boost
    train_strokes = datasets["train"].tensors[6].numpy()
    counts = np.bincount(train_strokes, minlength=len(STROKE_CLASSES))
    weights = len(train_strokes) / (len(STROKE_CLASSES) * np.maximum(counts, 1).astype(np.float32))
    drive_idx = STROKE_CLASSES.index("drive")
    weights[drive_idx] *= args.drive_boost
    stroke_loss_weight = torch.from_numpy(weights).float()

    print("\nCustom Class Weights:")
    for cls_name, w in zip(STROKE_CLASSES, stroke_loss_weight):
        print(f"  {cls_name:<12}: {float(w):.3f}")

    # Build model based on ablation mode
    use_temporal = args.ablation != "no_temporal"
    use_contact = args.ablation not in ("no_temporal", "no_contact")
    use_gate = args.ablation not in ("no_temporal", "no_gate")

    model = CRGatedFusionModel(
        use_temporal=use_temporal,
        use_contact=use_contact,
        use_gate=use_gate,
        use_cross_attention=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    stroke_loss_weight = stroke_loss_weight.to(device)

    criterion_stroke = nn.CrossEntropyLoss(weight=stroke_loss_weight)
    criterion_side = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = -1.0
    best_ckpt_path = run_dir / "best.pth"

    print("\nBat dau huan luyen...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for rgb, pose, court, shuttle, contact, quality, stroke_gt, side_gt in train_loader:
            rgb = rgb.to(device)
            pose = pose.to(device)
            court = court.to(device)
            shuttle = shuttle.to(device)
            contact = contact.to(device)
            quality = quality.to(device)
            stroke_gt = stroke_gt.to(device)
            side_gt = side_gt.to(device)

            optimizer.zero_grad()
            stroke_logits, side_logits, _ = model(rgb, pose, court, shuttle, contact, quality)
            loss_stroke = criterion_stroke(stroke_logits, stroke_gt)
            loss_side = criterion_side(side_logits, side_gt)
            loss = loss_stroke + 0.5 * loss_side

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * rgb.size(0)

        scheduler.step()
        train_loss = total_loss / len(datasets["train"])

        val_stroke_metrics, val_side_metrics, _, _ = evaluate_split(model, val_loader, device)
        val_f1 = val_stroke_metrics["macro_f1"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ablation": args.ablation,
                "val_f1": val_f1,
                "val_metrics": val_stroke_metrics,
            }, best_ckpt_path)

        if epoch % 2 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"  Epoch {epoch:2d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val F1: {val_f1*100:.2f}% (Best: {best_val_f1*100:.2f}%)", flush=True)

    # Load best checkpoint and perform final test evaluation
    print(f"\nTai checkpoint tot nhat (Val F1: {best_val_f1*100:.2f}%) de danh gia tren Test Set...", flush=True)
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_stroke, test_side, test_cm, gate_weights = evaluate_split(model, test_loader, device)

    print("\n" + "=" * 65)
    print(f"KET QUA KIEM THU TRUOC HOI DONG - TEST SET ({len(datasets['test'])} SAMPLES)")
    print(f"Che do: {args.ablation.upper()}")
    print("=" * 65)
    print(f"  * Stroke Top-1 Accuracy : {test_stroke['accuracy'] * 100:.2f}%")
    print(f"  * Stroke Macro-F1       : {test_stroke['macro_f1'] * 100:.2f}%")
    print(f"  * Side Macro-F1         : {test_side['macro_f1'] * 100:.2f}%")

    if gate_weights is not None:
        print("\nTRONG SO TIN CAY TRUNG BINH TUNG MODALITY (Reliability Gate):")
        mod_names = ["RGB Video", "Pose (Joints)", "Court Position", "Shuttlecock"]
        for name, w in zip(mod_names, gate_weights):
            print(f"  {name:<16}: {w*100:.1f}%")

    print("\nCHI TIET F1 TUNG LOAI CU DANH:")
    print(f"{'Class':<14} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Samples'}")
    print("-" * 60)
    for idx, name in enumerate(STROKE_CLASSES):
        p = test_stroke["precision"][idx] * 100
        r = test_stroke["recall"][idx] * 100
        f1 = test_stroke["f1"][idx] * 100
        supp = test_stroke["support"][idx]
        print(f"{name:<14} | {p:>8.1f}%  | {r:>8.1f}%  | {f1:>8.1f}%  | {supp}")
    print("-" * 60)

    # Save metrics JSON
    out_metrics = {
        "ablation": args.ablation,
        "test_stroke": test_stroke,
        "test_side": test_side,
        "gate_weights": gate_weights,
    }
    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)
    print(f"Da luu ket qua vao: {run_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
