"""Reproduce the original KAP online balanced model on Matches 13-17 validated on Match 41."""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from ai_classifier.models import KAPScratchModel, SIDE_CLASSES, STROKE_CLASSES
from scripts.evaluation.evaluate_kap_full_shuttleset_test import extract_sample_sequences, norm_clip, RGB
from scripts.training.train_kap_scratch import confusion_metrics

def main():
    seed = 20260926
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    manifest_path = ROOT / "data/manifests/shuttleset_npy.csv"
    rgb_cache = Path(r"C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_v3_full\frames_16")
    npy_root = Path(r"C:\Users\ADMIN\Downloads\dataset_npy_between_2_hits_with_max_limits\dataset_npy_between_2_hits_with_max_limits")
    rgb_ckpt = ROOT / "src/ai_classifier/models/checkpoints/visual_backbone.pth"
    output_ckpt = ROOT / "src/ai_classifier/models/checkpoints/kap_classifier.pth"
    cache_dir = ROOT / "work_dirs/kap_online_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RESTORING ORIGINAL ROBUST KAP BALANCED CHECKPOINT")
    print(f"Target Checkpoint: {output_ckpt}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with manifest_path.open(encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))

    # Calculate hit relative targets
    groups = defaultdict(list)
    for r in all_rows:
        groups[(r["match_id"], r["set_id"], r["rally"])].append(r)
    targets = {}
    for g in groups.values():
        ordered = sorted(g, key=lambda x: int(x["ball_round"]))
        for i, r in enumerate(ordered):
            hit = int(float(r["hit_frame"]))
            prev = int(float(ordered[i-1]["hit_frame"])) if i else hit - 15
            targets[r["sample_id"]] = hit - max(prev, hit - 45)

    # 1. Feature extraction helper
    rgb_model = RGB()
    rgb_model.load_state_dict(torch.load(rgb_ckpt, map_location="cpu", weights_only=False)["model"])
    rgb_model.to(device).eval()

    def get_features(match_ids: list[str], cache_name: str):
        cache_file = cache_dir / f"{cache_name}.npz"
        if cache_file.exists():
            print(f"Loading {cache_name} from cache {cache_file}...")
            data = np.load(cache_file)
            arrays = [data[k] for k in ("emb", "pose", "court", "shuttle", "contact", "quality")]
            ys = data["ys"]
            yd = data["yd"]
            return arrays, ys, yd

        m_rows = [r for r in all_rows if r["match_id"] in match_ids and r["valid_coarse_label"] == "True" and r["valid_stroke_side"] == "True"]
        print(f"Extracting features for matches {match_ids} ({len(m_rows)} samples)...")
        emb = []
        with torch.inference_mode():
            for s in range(0, len(m_rows), 32):
                batch = m_rows[s:s+32]
                clips = torch.stack([norm_clip(rgb_cache / (r["sample_id"] + ".npy")) for r in batch]).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                    feats = rgb_model.backbone(clips).float().cpu().numpy()
                emb.extend(feats)

        pose, court, shuttle, contact, quality = [], [], [], [], []
        for r in m_rows:
            j = np.load(npy_root / r["joints_path"])
            p = np.load(npy_root / r["position_path"])
            s = np.load(npy_root / r["shuttle_path"])
            x = extract_sample_sequences(j, p, s, targets[r["sample_id"]])
            pose.append(x[0]); court.append(x[1]); shuttle.append(x[2]); contact.append(x[3]); quality.append(x[4])

        arrays = [np.asarray(x, dtype=np.float32) for x in (emb, pose, court, shuttle, contact, quality)]
        ys = np.array([STROKE_CLASSES.index(r["coarse_label"]) for r in m_rows], dtype=np.int64)
        yd = np.array([SIDE_CLASSES.index(r["stroke_side"]) for r in m_rows], dtype=np.int64)

        np.savez_compressed(cache_file, emb=arrays[0], pose=arrays[1], court=arrays[2], shuttle=arrays[3], contact=arrays[4], quality=arrays[5], ys=ys, yd=yd)
        return arrays, ys, yd

    # Training matches: 13, 14, 15, 16, 17
    train_arrays, train_ys, train_yd = get_features(["13", "14", "15", "16", "17"], "train_matches13_17")
    # Validation match: 41
    val_arrays, val_ys, val_yd = get_features(["41"], "val_match41")

    print(f"Train samples: {len(train_ys)} | Val samples: {len(val_ys)}")

    # Class distribution
    counts = np.bincount(train_ys, minlength=8)
    for i, c in enumerate(STROKE_CLASSES):
        print(f"  {c:12s}: {counts[i]} ({counts[i]/len(train_ys)*100:.1f}%)")

    # DataLoaders
    train_ds = TensorDataset(
        *[torch.from_numpy(arr).float() for arr in train_arrays],
        torch.from_numpy(train_ys).long(),
        torch.from_numpy(train_yd).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    val_ds = TensorDataset(
        *[torch.from_numpy(arr).float() for arr in val_arrays],
        torch.from_numpy(val_ys).long(),
        torch.from_numpy(val_yd).long(),
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    # Model initialization
    model = KAPScratchModel()
    # Compute normalization from training set
    pose_val = torch.from_numpy(train_arrays[1])
    court_val = torch.from_numpy(train_arrays[2])
    shuttle_val = torch.from_numpy(train_arrays[3])
    quality_val = torch.from_numpy(train_arrays[5])

    model.set_normalization(
        pose_mean=pose_val.mean(dim=(0, 1)),
        pose_std=pose_val.std(dim=(0, 1)).clamp_min(1e-5),
        court_mean=court_val.mean(dim=(0, 1)),
        court_std=court_val.std(dim=(0, 1)).clamp_min(1e-5),
        shuttle_mean=shuttle_val.mean(dim=(0, 1)),
        shuttle_std=shuttle_val.std(dim=(0, 1)).clamp_min(1e-5),
        quality_mean=quality_val.mean(dim=0),
        quality_std=quality_val.std(dim=0).clamp_min(1e-5),
    )
    model.to(device)

    # Balanced class weights (mild power 0.5 to prevent distortion of common classes)
    weights = (len(train_ys) / (8.0 * np.maximum(counts, 1))) ** 0.5
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    class_weights /= class_weights.mean()

    criterion_stroke = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    criterion_side = nn.CrossEntropyLoss(label_smoothing=0.03)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    epochs = 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = 0.0
    best_metrics = None
    best_state = None
    best_epoch = 0

    print("\n--- TRAINING BALANCED ROBUST KAP ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inp = [x.to(device) for x in batch[:6]]
            target_stroke = batch[6].to(device)
            target_side = batch[7].to(device)

            out_stroke, out_side, _ = model(*inp)
            loss = criterion_stroke(out_stroke, target_stroke) + 0.5 * criterion_side(out_side, target_side)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(target_stroke)

        scheduler.step()

        # Validation on Match 41
        model.eval()
        cm = torch.zeros(8, 8, dtype=torch.int64)
        with torch.inference_mode():
            for batch in val_loader:
                inp = [x.to(device) for x in batch[:6]]
                out_stroke, _, _ = model(*inp)
                preds = out_stroke.argmax(dim=-1).cpu()
                for t, p in zip(batch[6], preds):
                    cm[int(t), int(p)] += 1

        metrics = confusion_metrics(cm)
        acc = metrics["accuracy"]
        bal_acc = metrics["balanced_accuracy"]
        macro_f1 = metrics["macro_f1"]

        # Selection metric: balanced_accuracy (same as 20261008 protocol)
        score = bal_acc
        print(f"Epoch {epoch:02d}/{epochs}: Val Acc={acc*100:.2f}% | BalAcc={bal_acc*100:.2f}% | Macro-F1={macro_f1*100:.2f}%", flush=True)

        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print("\n" + "=" * 70)
    print(f"Best Epoch: {best_epoch:02d} | Balanced Accuracy: {best_score*100:.2f}% | Accuracy: {best_metrics['accuracy']*100:.2f}% | Macro-F1: {best_metrics['macro_f1']*100:.2f}%")
    print("Class F1 scores on Validation Match 41:")
    for i, c in enumerate(STROKE_CLASSES):
        print(f"  {c:12s}: {best_metrics['f1'][i]*100:.2f}% (Recall={best_metrics['recall'][i]*100:.2f}%)")
    print("=" * 70)

    # Save to production checkpoint
    torch.save({
        "model": best_state,
        "epoch": best_epoch,
        "base_checkpoint": "reproduced_clean_online_balanced",
        "online_train_matches": ["13", "14", "15", "16", "17"],
        "online_validation_match": "41",
        "val_metrics": best_metrics,
        "selection_metric": "balanced_accuracy",
        "selection_score": best_score,
    }, output_ckpt)
    print(f"\n[+] Successfully saved restored checkpoint to: {output_ckpt}")

if __name__ == "__main__":
    main()
