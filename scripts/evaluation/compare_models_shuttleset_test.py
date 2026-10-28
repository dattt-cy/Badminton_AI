import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("e:/HK1-2026/PBL6/AI_Classifier")
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from ai_classifier.models import (
    KAPScratchModel,
    KAPCrossTransModel,
    KAPCrossTransV2Model,
    STROKE_CLASSES,
    SIDE_CLASSES
)
from scripts.evaluation.evaluate_kap_full_shuttleset_test import args, load_rows

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load test rows to get ground truth labels
a = args()
rows, targets = load_rows(a)
ys = np.array([STROKE_CLASSES.index(r.stroke) for r in rows])
yd = np.array([SIDE_CLASSES.index(r.side) for r in rows])

# Load precomputed features
feat_path = ROOT / "work_dirs/kap_full_test5_features.npz"
feat = np.load(feat_path)
emb = feat["emb"]
pose = feat["pose"]
court = feat["court"]
shuttle = feat["shuttle"]
contact = feat["contact"]
quality = feat["quality"]

def evaluate_model(model, name):
    model.to(device).eval()
    all_preds_s = []
    all_preds_d = []
    
    with torch.inference_mode():
        for i in range(0, len(rows), 128):
            e_b = torch.from_numpy(emb[i:i+128]).float().to(device)
            p_b = torch.from_numpy(pose[i:i+128]).float().to(device)
            c_b = torch.from_numpy(court[i:i+128]).float().to(device)
            s_b = torch.from_numpy(shuttle[i:i+128]).float().to(device)
            ct_b = torch.from_numpy(contact[i:i+128]).float().to(device)
            q_b = torch.from_numpy(quality[i:i+128]).float().to(device)
            
            os, od, _ = model(e_b, p_b, c_b, s_b, ct_b, q_b)
            all_preds_s.append(os.argmax(dim=-1).cpu().numpy())
            all_preds_d.append(od.argmax(dim=-1).cpu().numpy())
            
    pred_s = np.concatenate(all_preds_s)
    pred_d = np.concatenate(all_preds_d)
    
    cm = np.zeros((8, 8), dtype=np.int64)
    np.add.at(cm, (ys, pred_s), 1)
    
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(8), where=cm.sum(0) != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(8), where=cm.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(8), where=(precision + recall) != 0)
    acc = float(tp.sum() / cm.sum())
    macro_f1 = float(f1.mean())
    side_acc = float((yd == pred_d).mean())
    
    print(f"\n=== Model: {name} ===")
    print(f"Accuracy: {acc*100:.2f}% | Macro-F1: {macro_f1*100:.2f}% | Side Acc: {side_acc*100:.2f}%")
    for cls_name, cls_f1, prec, rec, sup in zip(STROKE_CLASSES, f1, precision, recall, cm.sum(1)):
        print(f"  {cls_name:<12}: F1={cls_f1*100:5.2f}% (Prec={prec*100:5.2f}%, Rec={rec*100:5.2f}%, N={sup})")
        
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "side_accuracy": side_acc,
        "per_class_f1": f1.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "confusion_matrix": cm.tolist()
    }

results = {}

# 1. Base KAP
base_model = KAPScratchModel()
ckpt = torch.load(ROOT / "work_dirs/kap_robust_seed20260926/kap_best.pth", map_location=device, weights_only=False)
base_model.load_state_dict(ckpt["model"])
results["Base_KAP"] = evaluate_model(base_model, "Base KAP (TCN Early-Fusion)")

# 2. CrossTrans v1
cross_v1 = KAPCrossTransModel()
ckpt = torch.load(ROOT / "work_dirs/kap_cross_seed20260926/kap_best.pth", map_location=device, weights_only=False)
cross_v1.load_state_dict(ckpt["model"])
results["CrossTrans_v1"] = evaluate_model(cross_v1, "CrossTrans v1 (Global 68-d)")

# 3. CrossTrans v2 Lift-Boosted (Our current single model!)
cross_v2_lb = KAPCrossTransV2Model()
ckpt = torch.load(ROOT / "work_dirs/kap_cross_v2_liftboost_seed20260926/kap_best.pth", map_location=device, weights_only=False)
cross_v2_lb.load_state_dict(ckpt["model"])
results["CrossTrans_v2_LiftBoost"] = evaluate_model(cross_v2_lb, "CrossTrans v2 Lift-Boosted (Active Hitter Disambiguation)")

# Load BST official collapsed 8-class for direct comparison
with open(ROOT / "work_dirs/bst_common_test_evaluation.json") as f:
    bst_data = json.load(f)["bst_collapsed_8class"]

results["BST_Official"] = {
    "accuracy": bst_data["accuracy"],
    "macro_f1": bst_data["macro_f1"],
    "per_class_f1": bst_data["per_class_f1"],
    "support": bst_data["support"]
}

with open(ROOT / "work_dirs/head_to_head_bst_vs_kap.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to work_dirs/head_to_head_bst_vs_kap.json")
