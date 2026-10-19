"""Full two-stage KAP training:
Stage 1: Train base KAP on entire ShuttleSet train split (24,834 samples).
Stage 2: Online adaptation fine-tuning on Matches 13-17 with validation on Match 41.
"""
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
    out_dir = ROOT / "work_dirs/full_shuttleset_kap"
    out_dir.mkdir(parents=True, exist_ok=True)
    prod_ckpt = ROOT / "src/ai_classifier/models/checkpoints/kap_classifier.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("FULL TWO-STAGE ROBUST KAP RECONSTRUCTION PIPELINE")
    print(f"Device: {device} | Production Target: {prod_ckpt}")
    print("=" * 75, flush=True)

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

    # Load frozen RGB backbone
    rgb_model = RGB()
    rgb_model.load_state_dict(torch.load(rgb_ckpt, map_location="cpu", weights_only=False)["model"])
    rgb_model.to(device).eval()

    def extract_and_cache(rows, cache_file):
        if cache_file.exists():
            print(f"[*] Loading features from cache: {cache_file}...", flush=True)
            data = np.load(cache_file)
            arrays = [data[k] for k in ("emb", "pose", "court", "shuttle", "contact", "quality")]
            return arrays, data["ys"], data["yd"]

        print(f"[*] Extracting features for {len(rows)} samples into {cache_file.name}...", flush=True)
        t0 = time.time()
        emb = []
        batch_size = 64
        with torch.inference_mode():
            for s in range(0, len(rows), batch_size):
                batch = rows[s:s+batch_size]
                clips = torch.stack([norm_clip(rgb_cache / (r["sample_id"] + ".npy")) for r in batch]).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                    feats = rgb_model.backbone(clips).float().cpu().numpy()
                emb.extend(feats)
                if (s // batch_size) % 50 == 0 or s + batch_size >= len(rows):
                    print(f"    RGB progress: {min(s + batch_size, len(rows))}/{len(rows)} ({time.time() - t0:.1f}s)", flush=True)

        pose, court, shuttle, contact, quality = [], [], [], [], []
        for r in rows:
            j = np.load(npy_root / r["joints_path"])
            p = np.load(npy_root / r["position_path"])
            s = np.load(npy_root / r["shuttle_path"])
            x = extract_sample_sequences(j, p, s, targets[r["sample_id"]])
            pose.append(x[0]); court.append(x[1]); shuttle.append(x[2]); contact.append(x[3]); quality.append(x[4])

        arrays = [np.asarray(x, dtype=np.float32) for x in (emb, pose, court, shuttle, contact, quality)]
        ys = np.array([STROKE_CLASSES.index(r["coarse_label"]) for r in rows], dtype=np.int64)
        yd = np.array([SIDE_CLASSES.index(r["stroke_side"]) for r in rows], dtype=np.int64)

        np.savez_compressed(cache_file, emb=arrays[0], pose=arrays[1], court=arrays[2], shuttle=arrays[3], contact=arrays[4], quality=arrays[5], ys=ys, yd=yd)
        print(f"[+] Finished extracting in {time.time() - t0:.1f}s. Saved cache: {cache_file}", flush=True)
        return arrays, ys, yd

    # 1. Full ShuttleSet train split (24,834 samples)
    train_full_rows = [r for r in all_rows if r["split"] == "train" and r["valid_coarse_label"] == "True" and r["valid_stroke_side"] == "True"]
    val_rows = [r for r in all_rows if r["match_id"] == "41" and r["valid_coarse_label"] == "True" and r["valid_stroke_side"] == "True"]
    online_rows = [r for r in all_rows if r["match_id"] in ("13", "14", "15", "16", "17") and r["valid_coarse_label"] == "True" and r["valid_stroke_side"] == "True"]

    train_arrays, train_ys, train_yd = extract_and_cache(train_full_rows, out_dir / "shuttleset_train_24k.npz")
    val_arrays, val_ys, val_yd = extract_and_cache(val_rows, out_dir / "match41_val.npz")

    print(f"\n[+] Total Base Train Samples: {len(train_ys)} | Val Samples: {len(val_ys)}", flush=True)
    counts = np.bincount(train_ys, minlength=8)
    for i, c in enumerate(STROKE_CLASSES):
        print(f"    {c:12s}: {counts[i]} ({counts[i]/len(train_ys)*100:.1f}%)", flush=True)

    # =========================================================================
    # STAGE 1: TRAIN BASE KAP ON ENTIRE 24,834 SAMPLES
    # =========================================================================
    print("\n" + "=" * 75)
    print("STAGE 1: TRAINING BASE KAP ON FULL SHUTTLESET (24,834 SAMPLES)")
    print("=" * 75, flush=True)

    train_ds = TensorDataset(
        *[torch.from_numpy(arr).float() for arr in train_arrays],
        torch.from_numpy(train_ys).long(),
        torch.from_numpy(train_yd).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    val_ds = TensorDataset(
        *[torch.from_numpy(arr).float() for arr in val_arrays],
        torch.from_numpy(val_ys).long(),
        torch.from_numpy(val_yd).long(),
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    base_model = KAPScratchModel()
    pose_val = torch.from_numpy(train_arrays[1])
    court_val = torch.from_numpy(train_arrays[2])
    shuttle_val = torch.from_numpy(train_arrays[3])
    quality_val = torch.from_numpy(train_arrays[5])

    base_model.set_normalization(
        pose_mean=pose_val.mean(dim=(0, 1)),
        pose_std=pose_val.std(dim=(0, 1)).clamp_min(1e-5),
        court_mean=court_val.mean(dim=(0, 1)),
        court_std=court_val.std(dim=(0, 1)).clamp_min(1e-5),
        shuttle_mean=shuttle_val.mean(dim=(0, 1)),
        shuttle_std=shuttle_val.std(dim=(0, 1)).clamp_min(1e-5),
        quality_mean=quality_val.mean(dim=0),
        quality_std=quality_val.std(dim=0).clamp_min(1e-5),
    )
    base_model.to(device)

    weights = (len(train_ys) / (8.0 * np.maximum(counts, 1))) ** 0.5
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    class_weights /= class_weights.mean()

    criterion_stroke = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    criterion_side = nn.CrossEntropyLoss(label_smoothing=0.03)

    optimizer = torch.optim.AdamW(base_model.parameters(), lr=3e-4, weight_decay=1e-4)
    stage1_epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_epochs)

    best_base_macro = 0.0
    best_base_state = None

    for epoch in range(1, stage1_epochs + 1):
        base_model.train()
        total_loss = 0.0
        for batch in train_loader:
            inp = [x.to(device) for x in batch[:6]]
            target_stroke = batch[6].to(device)
            target_side = batch[7].to(device)

            out_stroke, out_side, _ = base_model(*inp)
            loss = criterion_stroke(out_stroke, target_stroke) + 0.5 * criterion_side(out_side, target_side)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(target_stroke)

        scheduler.step()

        # Validate on Match 41
        base_model.eval()
        cm = torch.zeros(8, 8, dtype=torch.int64)
        with torch.inference_mode():
            for batch in val_loader:
                inp = [x.to(device) for x in batch[:6]]
                out_stroke, _, _ = base_model(*inp)
                preds = out_stroke.argmax(dim=-1).cpu()
                for t, p in zip(batch[6], preds):
                    cm[int(t), int(p)] += 1

        metrics = confusion_metrics(cm)
        print(f"Stage 1 Epoch {epoch:02d}/{stage1_epochs}: Val Acc={metrics['accuracy']*100:.2f}% | BalAcc={metrics['balanced_accuracy']*100:.2f}% | Macro-F1={metrics['macro_f1']*100:.2f}%", flush=True)

        if metrics["macro_f1"] > best_base_macro:
            best_base_macro = metrics["macro_f1"]
            best_base_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

    base_ckpt_path = out_dir / "kap_base_24k_best.pth"
    torch.save({"model": best_base_state, "macro_f1": best_base_macro}, base_ckpt_path)
    print(f"\n[+] Saved Stage 1 Base Model to: {base_ckpt_path} (Val Macro-F1: {best_base_macro*100:.2f}%)", flush=True)

    # =========================================================================
    # STAGE 2: ONLINE ADAPTATION FINE-TUNING ON MATCHES 13-17
    # =========================================================================
    print("\n" + "=" * 75)
    print("STAGE 2: ONLINE ADAPTATION FINE-TUNING (MATCHES 13-17)")
    print("=" * 75, flush=True)

    online_arrays, online_ys, online_yd = extract_and_cache(online_rows, out_dir / "matches13_17_online.npz")
    online_ds = TensorDataset(
        *[torch.from_numpy(arr).float() for arr in online_arrays],
        torch.from_numpy(online_ys).long(),
        torch.from_numpy(online_yd).long(),
    )
    online_loader = DataLoader(online_ds, batch_size=32, shuffle=True)

    # Load Stage 1 base model weights
    model = KAPScratchModel()
    model.load_state_dict(best_base_state)
    model.to(device)

    # Fine-tune with low learning rate and replay anchor
    optimizer_ft = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    stage2_epochs = 10
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=stage2_epochs)

    best_score = 0.0
    best_metrics = None
    best_final_state = None
    best_final_epoch = 0

    replay_iter = iter(train_loader)
    for epoch in range(1, stage2_epochs + 1):
        model.train()
        for batch in online_loader:
            try:
                clean_batch = next(replay_iter)
            except StopIteration:
                replay_iter = iter(train_loader)
                clean_batch = next(replay_iter)

            optimizer_ft.zero_grad()
            online_x = [v.to(device) for v in batch[:6]]
            clean_x = [v.to(device) for v in clean_batch[:6]]

            out_on, side_on, _ = model(*online_x)
            out_cl, side_cl, _ = model(*clean_x)

            loss_on = criterion_stroke(out_on, batch[6].to(device)) + 0.5 * criterion_side(side_on, batch[7].to(device))
            loss_cl = criterion_stroke(out_cl, clean_batch[6].to(device)) + 0.5 * criterion_side(side_cl, clean_batch[7].to(device))

            loss = loss_on + 0.5 * loss_cl
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_ft.step()

        scheduler_ft.step()

        # Validate on Match 41
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
        score = metrics["balanced_accuracy"]
        print(f"Stage 2 Epoch {epoch:02d}/{stage2_epochs}: Val Acc={metrics['accuracy']*100:.2f}% | BalAcc={metrics['balanced_accuracy']*100:.2f}% | Macro-F1={metrics['macro_f1']*100:.2f}%", flush=True)

        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_final_epoch = epoch
            best_final_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print("\n" + "=" * 75)
    print("FINAL RESTORED MODEL EVALUATION (MATCH 41)")
    print(f"Selected Epoch: {best_final_epoch:02d} | Balanced Acc: {best_score*100:.2f}% | Accuracy: {best_metrics['accuracy']*100:.2f}% | Macro-F1: {best_metrics['macro_f1']*100:.2f}%")
    print("Class F1:")
    for i, c in enumerate(STROKE_CLASSES):
        print(f"  {c:12s}: {best_metrics['f1'][i]*100:.2f}% (Recall={best_metrics['recall'][i]*100:.2f}%)")
    print("=" * 75)

    # Save to production checkpoint
    torch.save({
        "model": best_final_state,
        "epoch": best_final_epoch,
        "base_checkpoint": "full_shuttleset_24k_base",
        "online_train_matches": ["13", "14", "15", "16", "17"],
        "online_validation_match": "41",
        "val_metrics": best_metrics,
        "selection_metric": "balanced_accuracy",
        "selection_score": best_score,
    }, prod_ckpt)
    print(f"\n[+] Production Checkpoint RESTORED: {prod_ckpt}", flush=True)

if __name__ == "__main__":
    main()
