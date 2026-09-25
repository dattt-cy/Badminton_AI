"""Neural Disambiguation Head (NDH) for Multimodal Stroke Classification.

Takes the pre-trained KAPFusionModel fused representation + logits + exit kinematics
and learns a neural residual adjustment to push Rank-2 predictions into Rank-1.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_classifier.models import SIDE_CLASSES, STROKE_CLASSES
from scripts.training.train_fine_badminton_rgb import confusion_metrics
from train_kap_fusion import KAPFusionModel, augment_shuttle_kinematics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_exit_kinematics(shuttle: torch.Tensor, court: torch.Tensor) -> torch.Tensor:
    """
    Extracts post-impact directional exit kinematics:
    1. vy_post: shuttle[16, 1] - shuttle[11, 1]
    2. vx_post: |shuttle[16, 0] - shuttle[11, 0]|
    3. speed_post: norm(shuttle[16] - shuttle[11]) * 20.0
    4. court_y: court[10, 1]
    5. vy_fwd: vy_post * sign(court_y - 0.5)
    Total: 5 dimensions.
    """
    vy_post = shuttle[:, 16, 1:2] - shuttle[:, 11, 1:2]
    vx_post = torch.abs(shuttle[:, 16, 0:1] - shuttle[:, 11, 0:1])
    diff = shuttle[:, 16] - shuttle[:, 11]
    speed_post = torch.norm(diff, dim=-1, keepdim=True) * 20.0
    court_y = court[:, 10, 1:2]
    fwd_sign = torch.sign(court_y - 0.5 + 1e-5)
    vy_fwd = vy_post * fwd_sign

    kin = torch.cat([vy_post, vx_post, speed_post, court_y, vy_fwd], dim=-1)
    return torch.nan_to_num(kin, nan=0.0, posinf=0.0, neginf=0.0)


class NeuralDisambiguationHead(nn.Module):
    """Residual Neural Disambiguation Head."""

    def __init__(self, base_kap_path: str = "work_dirs/cr_gated_fusion/cr_gated_kap/best.pth"):
        super().__init__()
        self.base_model = KAPFusionModel().to(DEVICE)
        ckpt = torch.load(base_kap_path, map_location=DEVICE, weights_only=False)
        self.base_model.load_state_dict(ckpt["model"])
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = False  # Freeze base backbone

        # Kinematics projection: 5-D -> 64-D
        self.kin_proj = nn.Sequential(
            nn.Linear(5, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
        )

        # Disambiguation Residual MLP:
        # Input: base_logits (8) + fused_feat (128) + kin_feat (64) = 200-D
        self.residual_mlp = nn.Sequential(
            nn.Linear(8 + 128 + 64, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 8),
        )

        # Scale factor for residual adjustment
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, rgb, pose, court, shuttle, contact, quality):
        with torch.no_grad():
            s_logits, side_logits, _, fused = self.base_model(
                rgb, pose, court, shuttle, contact, quality, return_features=True
            )

        raw_shuttle = shuttle[..., :2] if shuttle.size(-1) > 2 else shuttle
        raw_court = court[..., :4] if court.size(-1) > 4 else court
        kin = extract_exit_kinematics(raw_shuttle, raw_court)
        z_kin = self.kin_proj(kin)

        combined = torch.cat([s_logits, fused, z_kin], dim=-1)
        delta = self.residual_mlp(combined)

        refined_logits = s_logits + self.gamma * delta
        return refined_logits, side_logits


def train_ndh():
    print("=" * 75)
    print("HUAN LUYEN NEURAL DISAMBIGUATION HEAD (NDH)")
    print("MUC TIEU: DUNG HOC MAY DA PHUONG THUC DE DAY TOP-2 LEN TOP-1")
    print("=" * 75)

    data_path = Path("work_dirs/cr_gated_fusion/sequence_features.npz")
    data = np.load(data_path)
    splits = data["split"]

    datasets = {}
    for sp in ("train", "val", "test"):
        mask = splits == sp
        datasets[sp] = TensorDataset(
            torch.from_numpy(data["rgb"][mask]),
            torch.from_numpy(data["pose"][mask]),
            torch.from_numpy(data["court"][mask]),
            torch.from_numpy(data["shuttle"][mask]),
            torch.from_numpy(data["contact_dist"][mask]),
            torch.from_numpy(data["quality"][mask]),
            torch.from_numpy(data["stroke"][mask]),
            torch.from_numpy(data["side"][mask]),
        )

    train_loader = DataLoader(datasets["train"], batch_size=256, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=256, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=256, shuffle=False)

    model = NeuralDisambiguationHead().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)

    best_val_f1 = -1.0
    best_ckpt = Path("work_dirs/cr_gated_fusion/cr_gated_ndh.pth")

    t0 = time.time()
    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0
        for rgb, pose, court, shuttle, contact, quality, s_gt, _ in train_loader:
            rgb = rgb.to(DEVICE)
            pose = pose.to(DEVICE)
            court = court.to(DEVICE)
            shuttle = shuttle.to(DEVICE)
            contact = contact.to(DEVICE)
            quality = quality.to(DEVICE)
            s_gt = s_gt.to(DEVICE)

            optimizer.zero_grad()
            s_logits, _ = model(rgb, pose, court, shuttle, contact, quality)
            loss = criterion(s_logits, s_gt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if epoch % 2 == 0 or epoch == 20:
            model.eval()
            s_preds, s_targets = [], []
            with torch.no_grad():
                for rgb, pose, court, shuttle, contact, quality, s_gt, _ in val_loader:
                    s_logits, _ = model(
                        rgb.to(DEVICE), pose.to(DEVICE), court.to(DEVICE), shuttle.to(DEVICE), contact.to(DEVICE), quality.to(DEVICE)
                    )
                    s_preds.append(s_logits.argmax(dim=-1).cpu())
                    s_targets.append(s_gt)

            s_p = torch.cat(s_preds)
            s_t = torch.cat(s_targets)
            cm = torch.zeros(len(STROKE_CLASSES), len(STROKE_CLASSES), dtype=torch.int64)
            for t, p in zip(s_t, s_p):
                cm[int(t), int(p)] += 1
            metrics = confusion_metrics(cm)
            val_f1 = metrics["macro_f1"]

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({"epoch": epoch, "model": model.state_dict(), "val_f1": val_f1}, best_ckpt)

            print(f"  Epoch {epoch:2d}/20 | Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1*100:.2f}% (Best: {best_val_f1*100:.2f}%)")

    print(f"\nHuan luyen xong trong {time.time()-t0:.1f}s! Tai best checkpoint ({best_val_f1*100:.2f}%)...")
    ckpt = torch.load(best_ckpt, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Test Set
    s_preds, s_targets = [], []
    with torch.no_grad():
        for rgb, pose, court, shuttle, contact, quality, s_gt, _ in test_loader:
            s_logits, _ = model(
                rgb.to(DEVICE), pose.to(DEVICE), court.to(DEVICE), shuttle.to(DEVICE), contact.to(DEVICE), quality.to(DEVICE)
            )
            s_preds.append(s_logits.argmax(dim=-1).cpu())
            s_targets.append(s_gt)

    s_p = torch.cat(s_preds)
    s_t = torch.cat(s_targets)
    cm = torch.zeros(len(STROKE_CLASSES), len(STROKE_CLASSES), dtype=torch.int64)
    for t, p in zip(s_t, s_p):
        cm[int(t), int(p)] += 1
    test_metrics = confusion_metrics(cm)
    print("\n" + "=" * 65)
    print("KET QUA TEST SET (871 SAMPLES) - NDH")
    print("=" * 65)
    print(f"  * Top-1 Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"  * Macro-F1      : {test_metrics['macro_f1']*100:.2f}%")

    # Match 44
    records_m44 = torch.load("work_dirs/match44_5min_features.pt", weights_only=False)
    with open("work_dirs/match44_5min_eval_results.json", "r") as f:
        gt_m44 = [{"frame": int(ev["frame"]), "stroke": ev["gt_stroke"]} for ev in json.load(f)["detailed_events"]]
    gt_m44.sort(key=lambda x: x["frame"])

    preds_m44 = []
    with torch.no_grad():
        for r in records_m44:
            s_logits, _ = model(r["emb"].to(DEVICE), r["s_j"].to(DEVICE), r["s_p"].to(DEVICE), r["s_s"].to(DEVICE), r["s_c"].to(DEVICE), r["s_q"].to(DEVICE))
            probs = torch.softmax(s_logits[0], dim=-1).cpu().numpy()
            top2 = probs.argsort()[-2:][::-1]
            preds_m44.append({"frame": r["hit_frame"], "top2": [STROKE_CLASSES[i] for i in top2]})

    used = set()
    matched = []
    for gt in gt_m44:
        gf = gt["frame"]
        cands = [(abs(p["frame"] - gf), idx, p) for idx, p in enumerate(preds_m44) if idx not in used and abs(p["frame"] - gf) <= 12]
        if cands:
            cands.sort(key=lambda x: x[0])
            used.add(cands[0][1])
            matched.append({"gt": gt["stroke"], "pred": cands[0][2]["top2"][0], "top2": cands[0][2]["top2"]})

    t1_m44 = sum(1 for m in matched if m["pred"] == m["gt"])
    t2_m44 = sum(1 for m in matched if m["gt"] in m["top2"])
    tot_m44 = len(gt_m44)

    print("\n" + "=" * 65)
    print("DANH GIA TRAN 44 (AXELSEN VS ANTONSEN, 96 GT)")
    print("=" * 65)
    print(f"Top-1 Accuracy on Match 44: {t1_m44}/{tot_m44} ({t1_m44/tot_m44*100:.2f}%) [Matched: {t1_m44/len(matched)*100:.2f}%]")
    print(f"Top-2 Accuracy on Match 44: {t2_m44}/{tot_m44} ({t2_m44/tot_m44*100:.2f}%) [Matched: {t2_m44/len(matched)*100:.2f}%]")

    # Per class on Match 44
    per_cls = defaultdict(lambda: {"tot": 0, "t1": 0, "t2": 0})
    for m in matched:
        c = m["gt"]
        per_cls[c]["tot"] += 1
        if m["pred"] == c:
            per_cls[c]["t1"] += 1
        if c in m["top2"]:
            per_cls[c]["t2"] += 1

    print("\nChi tiet tung loai tren Tran 44:")
    print(f"{'Stroke':<14} | {'Total':<6} | {'Top-1':<10} | {'Top-2':<10}")
    print("-" * 50)
    for c in sorted(per_cls.keys()):
        c_tot = per_cls[c]["tot"]
        c_t1 = per_cls[c]["t1"]
        c_t2 = per_cls[c]["t2"]
        print(f"{c:<14} | {c_tot:<6} | {c_t1}/{c_tot} ({c_t1/c_tot*100:4.1f}%) | {c_t2}/{c_tot} ({c_t2/c_tot*100:4.1f}%)")
    print("-" * 50)

    # Match 39
    import csv
    gt_m39 = []
    with open("data/manifests/shuttleset_rgb.csv", mode="r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if "39 Anthony_Sinisuka_Ginting" in row["video_path"]:
                abs_hf = int(row["hit_frame"])
                if 10000 <= abs_hf <= 19000:
                    gt_m39.append({"frame": abs_hf - 10000, "stroke": row["coarse_label"]})
    gt_m39.sort(key=lambda x: x["frame"])

    records_m39 = torch.load("work_dirs/match39_5min_features.pt", weights_only=False)
    preds_m39 = []
    with torch.no_grad():
        for r in records_m39:
            s_logits, _ = model(r["emb"].to(DEVICE), r["s_j"].to(DEVICE), r["s_p"].to(DEVICE), r["s_s"].to(DEVICE), r["s_c"].to(DEVICE), r["s_q"].to(DEVICE))
            probs = torch.softmax(s_logits[0], dim=-1).cpu().numpy()
            top2 = probs.argsort()[-2:][::-1]
            preds_m39.append({"frame": r["hit_frame"], "top2": [STROKE_CLASSES[i] for i in top2]})

    used = set()
    matched = []
    for gt in gt_m39:
        gf = gt["frame"]
        cands = [(abs(p["frame"] - gf), idx, p) for idx, p in enumerate(preds_m39) if idx not in used and abs(p["frame"] - gf) <= 12]
        if cands:
            cands.sort(key=lambda x: x[0])
            used.add(cands[0][1])
            matched.append({"gt": gt["stroke"], "pred": cands[0][2]["top2"][0], "top2": cands[0][2]["top2"]})

    t1_m39 = sum(1 for m in matched if m["pred"] == m["gt"])
    t2_m39 = sum(1 for m in matched if m["gt"] in m["top2"])
    tot_m39 = len(gt_m39)
    print("\n" + "=" * 65)
    print("DANH GIA TRAN 39 (GINTING VS LEE ZII JIA, 92 GT)")
    print("=" * 65)
    print(f"Top-1 Accuracy on Match 39: {t1_m39}/{tot_m39} ({t1_m39/tot_m39*100:.2f}%) [Matched: {t1_m39/len(matched)*100:.2f}%]")
    print(f"Top-2 Accuracy on Match 39: {t2_m39}/{tot_m39} ({t2_m39/tot_m39*100:.2f}%) [Matched: {t2_m39/len(matched)*100:.2f}%]")


if __name__ == "__main__":
    train_ndh()
