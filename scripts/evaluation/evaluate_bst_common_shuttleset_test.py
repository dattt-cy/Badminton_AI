"""Evaluate official BST-CG-AP and collapse its 25 classes to KAP's 8 classes.

The 25-class result reproduces the official ShuttleSet merged protocol.  The
8-class result uses only sample IDs that have a valid KAP coarse label, making
it directly comparable with ``evaluate_kap_full_shuttleset_test.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


COARSE_CLASSES = [
    "serve", "clear", "smash", "drop", "net_shot", "lift", "drive", "net_attack"
]
RAW_TO_COARSE = {
    "發短球": "serve", "發長球": "serve", "長球": "clear",
    "殺球": "smash", "切球": "drop", "放小球": "net_shot",
    "擋小球": "net_shot", "勾球": "net_shot", "挑球": "lift",
    "平球": "drive", "推球": "net_attack", "撲球": "net_attack",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    workspace = root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bst-root", type=Path, default=workspace / "BST_official")
    parser.add_argument(
        "--collated", type=Path,
        default=workspace / "BST_official/downloads/shuttleset_test_collated_seq100/test",
    )
    parser.add_argument(
        "--raw-test", type=Path,
        default=Path(r"C:\Users\ADMIN\Downloads\dataset_npy_between_2_hits_with_max_limits\dataset_npy_between_2_hits_with_max_limits\test"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=workspace / "BST_official/downloads/shuttleset_weights/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged.pt",
    )
    parser.add_argument("--manifest", type=Path, default=root / "data/manifests/shuttleset_npy.csv")
    parser.add_argument("--output", type=Path, default=root / "work_dirs/bst_common_test_evaluation.json")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def metrics(truth: np.ndarray, prediction: np.ndarray, classes: int) -> dict:
    cm = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(cm, (truth, prediction), 1)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(classes), where=cm.sum(0) != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(classes), where=cm.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(classes), where=(precision + recall) != 0)
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "min_f1": float(f1.min()),
        "per_class_f1": f1.tolist(),
        "support": cm.sum(1).tolist(),
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.bst_root / "stroke_classification"))
    from model.bst import BST_CG_AP
    from preparing_data.shuttleset_dataset import get_merged_stroke_types

    pose = np.load(args.collated / "JnB_bone.npy", mmap_mode="r")
    position = np.load(args.collated / "pos.npy", mmap_mode="r")
    shuttle = np.load(args.collated / "shuttle.npy", mmap_mode="r")
    lengths = np.load(args.collated / "videos_len.npy", mmap_mode="r")
    labels = np.load(args.collated / "labels.npy")

    # Reconstruct the exact order used by the official collator.
    sample_ids: list[str] = []
    for class_dir in args.raw_test.iterdir():
        sample_ids.extend(p.name.removesuffix("_pos.npy") for p in sorted(class_dir.glob("*_pos.npy")))
    if len(sample_ids) != len(labels):
        raise RuntimeError(f"ID/data mismatch: {len(sample_ids)} != {len(labels)}")

    dataset = TensorDataset(
        torch.from_numpy(np.asarray(pose)), torch.from_numpy(np.asarray(position)),
        torch.from_numpy(np.asarray(shuttle)), torch.from_numpy(np.asarray(lengths)),
        torch.from_numpy(labels),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BST_CG_AP(in_dim=72, seq_len=100, n_class=25, depth_tem=2, depth_inter=1)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()

    logits_parts = []
    with torch.inference_mode():
        for human_pose, pos, ball, video_len, _ in loader:
            human_pose = human_pose.to(device).view(*human_pose.shape[:-2], -1)
            logits_parts.append(model(human_pose, ball.to(device), pos.to(device), video_len.to(device)).cpu())
    logits = torch.cat(logits_parts).numpy()
    pred25 = logits.argmax(1)
    top2 = np.argpartition(logits, -2, axis=1)[:, -2:]
    result25 = metrics(labels, pred25, 25)
    result25["top2_accuracy"] = float(np.any(top2 == labels[:, None], axis=1).mean())

    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        manifest = {row["sample_id"]: row for row in csv.DictReader(stream)}
    class_names = get_merged_stroke_types()
    class_to_coarse = np.full(25, -1, dtype=np.int64)
    for index, name in enumerate(class_names):
        raw = name.removeprefix("Top_").removeprefix("Bottom_")
        if raw in RAW_TO_COARSE:
            class_to_coarse[index] = COARSE_CLASSES.index(RAW_TO_COARSE[raw])

    common_indices = []
    truth8 = []
    for index, sample_id in enumerate(sample_ids):
        row = manifest.get(sample_id)
        if row and row["split"] == "test" and row["valid_coarse_label"] == "True" and row["valid_stroke_side"] == "True":
            common_indices.append(index)
            truth8.append(COARSE_CLASSES.index(row["coarse_label"]))
    common_indices = np.asarray(common_indices)
    truth8 = np.asarray(truth8)

    # Sum class probabilities, not logits, when merging mutually exclusive classes.
    probability25 = torch.from_numpy(logits[common_indices]).softmax(1).numpy()
    probability8 = np.zeros((len(common_indices), len(COARSE_CLASSES)), dtype=np.float32)
    for source_class, target_class in enumerate(class_to_coarse):
        if target_class >= 0:
            probability8[:, target_class] += probability25[:, source_class]
    pred8 = probability8.argmax(1)

    result = {
        "protocol": "official_bst_cg_ap_seq100_serial1_and_common_8class",
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "official_test_samples": int(len(labels)),
        "common_test_samples": int(len(common_indices)),
        "test_matches": ["39", "40", "42", "43", "44"],
        "bst_25class": result25,
        "bst_collapsed_8class": metrics(truth8, pred8, len(COARSE_CLASSES)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "samples_25": len(labels),
        "bst_25": {key: result25[key] for key in ("accuracy", "macro_f1", "min_f1", "top2_accuracy")},
        "samples_common_8": len(common_indices),
        "bst_common_8": {key: result["bst_collapsed_8class"][key] for key in ("accuracy", "macro_f1", "min_f1")},
    }, indent=2))


if __name__ == "__main__":
    main()
