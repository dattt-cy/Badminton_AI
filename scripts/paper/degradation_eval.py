"""Controlled degradation of shuttle and pose inputs for KAP ablations and BST-CG-AP.

For every rate r in {0, 10, ..., 50}%, each frame of the shuttle trajectory and
each (frame, joint) pair of both players' skeletons is independently set to the
missing value (0) with probability r.  Three conditions are evaluated: shuttle
only, pose only, and both.  Two further conditions model detector failures seen
on broadcast video: ``player`` drops a player's whole skeleton in a frame with
probability r (missed person detection), and ``burst`` removes one contiguous
segment covering r of the sequence from both the skeletons and the shuttle
trajectory (occlusion or motion blur).  Masks use a fixed seed per (condition, rate), so all
KAP checkpoints see identical inputs; BST sees the same rate on its own
100-step sequences (bones are rebuilt from the masked joints).

All metrics use the 3,334 common 8-class test events (matches 39/40/42/43/44).
Per-event clean predictions are also written for paired significance tests.

Example:
    python scripts/paper/degradation_eval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "training", REPO_ROOT / "scripts" / "paper"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_kap_ablation import KAPAblationModel  # noqa: E402
from train_kap_scratch import VideoDomainAugmentor  # noqa: E402

COARSE_CLASSES = ["serve", "clear", "smash", "drop", "net_shot", "lift", "drive", "net_attack"]
RAW_TO_COARSE = {
    "發短球": "serve", "發長球": "serve", "長球": "clear",
    "殺球": "smash", "切球": "drop", "放小球": "net_shot",
    "擋小球": "net_shot", "勾球": "net_shot", "挑球": "lift",
    "平球": "drive", "推球": "net_attack", "撲球": "net_attack",
}
RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
CONDITIONS = ("shuttle", "pose", "both", "player", "burst")


def parse_args() -> argparse.Namespace:
    bst = REPO_ROOT / "external/BST_official"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=REPO_ROOT / "work_dirs/context_window_pilot/features_w15_30.npz")
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "work_dirs/paper_final")
    parser.add_argument("--bst-root", type=Path, default=bst)
    parser.add_argument("--collated", type=Path, default=bst / "downloads/shuttleset_test_collated_seq100/test")
    parser.add_argument(
        "--raw-test", type=Path,
        default=Path(r"C:\Users\ADMIN\Downloads\dataset_npy_between_2_hits_with_max_limits\dataset_npy_between_2_hits_with_max_limits\test"),
    )
    parser.add_argument(
        "--bst-checkpoints", type=Path, nargs="+",
        default=[
            bst / "downloads/shuttleset_weights/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged.pt",
            bst / "downloads/shuttleset_weights/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt",
            bst / "downloads/shuttleset_weights/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2_actual.pt",
        ],
    )
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/manifests/shuttleset_npy.csv")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "work_dirs/paper_final/degradation.json")
    parser.add_argument("--mask-seed", type=int, default=20261101)
    return parser.parse_args()


def scores(truth: np.ndarray, prediction: np.ndarray) -> dict:
    classes = len(COARSE_CLASSES)
    cm = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(cm, (truth, prediction), 1)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(classes), where=cm.sum(0) != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(classes), where=cm.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(classes), where=(precision + recall) != 0)
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1.mean())}


def mask_rng(seed: int, condition: str, rate: float, family: str) -> np.random.Generator:
    return np.random.default_rng([seed, CONDITIONS.index(condition), int(rate * 100), family == "bst"])


# ---------------------------------------------------------------- KAP

def kap_inputs(features: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    raw = np.load(features)
    test = np.flatnonzero(raw["split"] == "test")
    tensors = {
        name: torch.from_numpy(raw[name][test]).float()
        for name in ("rgb", "pose", "court", "shuttle", "contact_dist", "quality")
    }
    return tensors, raw["sample_id"][test], raw["stroke"][test]


def degrade_kap(tensors: dict, condition: str, rate: float, rng: np.random.Generator) -> dict:
    if rate == 0:
        return tensors
    pose, shuttle = tensors["pose"].clone(), tensors["shuttle"].clone()
    count, frames = pose.shape[:2]
    if condition in ("pose", "both"):
        joints = pose.reshape(count, frames, -1, 2)
        drop = torch.from_numpy(rng.random(joints.shape[:3]) < rate).unsqueeze(-1)
        pose = joints.masked_fill(drop, 0.0).reshape_as(pose)
    if condition in ("shuttle", "both"):
        drop = torch.from_numpy(rng.random((count, frames)) < rate).unsqueeze(-1)
        shuttle = shuttle.masked_fill(drop, 0.0)
    if condition == "player":
        players = pose.reshape(count, frames, 2, -1)
        drop = torch.from_numpy(rng.random((count, frames, 2)) < rate).unsqueeze(-1)
        pose = players.masked_fill(drop, 0.0).reshape_as(pose)
    if condition == "burst":
        length = int(round(rate * frames))
        starts = rng.integers(0, frames - length + 1, count)
        steps = np.arange(frames)[None, :]
        drop = torch.from_numpy((steps >= starts[:, None]) & (steps < starts[:, None] + length)).unsqueeze(-1)
        pose = pose.masked_fill(drop, 0.0)
        shuttle = shuttle.masked_fill(drop, 0.0)
    # Update the stored quality vector by the change the training-time quality
    # function measures, so missing-rate entries rise with the mask.
    before = VideoDomainAugmentor._quality(tensors["pose"], tensors["shuttle"])
    after = VideoDomainAugmentor._quality(pose, shuttle)
    return {**tensors, "pose": pose, "shuttle": shuttle, "quality": tensors["quality"] + after - before}


@torch.inference_mode()
def kap_predict(model, tensors: dict, device: torch.device) -> np.ndarray:
    names = ("rgb", "pose", "court", "shuttle", "contact_dist", "quality")
    out = []
    for start in range(0, len(tensors["rgb"]), 512):
        stroke, _, _ = model(*[tensors[n][start:start + 512].to(device) for n in names])
        out.append(stroke.softmax(1).cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- BST

def bst_setup(args: argparse.Namespace):
    sys.path.insert(0, str(args.bst_root / "stroke_classification"))
    from model.bst import BST_CG_AP
    from preparing_data.badmintonDB_dataset import create_bones, get_bone_pairs
    from preparing_data.shuttleset_dataset import get_merged_stroke_types

    joints = np.load(args.collated / "J_only.npy")
    data = {
        "joints": joints,
        "pos": np.load(args.collated / "pos.npy"),
        "shuttle": np.load(args.collated / "shuttle.npy"),
        "lengths": np.load(args.collated / "videos_len.npy"),
    }
    pairs = get_bone_pairs("coco")
    data["make_pose"] = lambda j: np.concatenate(
        [j, np.stack([create_bones(sample, pairs) for sample in j])], axis=-2
    ).astype(np.float32)
    stored = np.load(args.collated / "JnB_bone.npy", mmap_mode="r")
    if not np.allclose(data["make_pose"](joints[:64]), stored[:64], atol=1e-6):
        raise RuntimeError("Rebuilt JnB_bone does not match the collated array")

    sample_ids = []
    for class_dir in args.raw_test.iterdir():
        sample_ids.extend(p.name.removesuffix("_pos.npy") for p in sorted(class_dir.glob("*_pos.npy")))
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        manifest = {row["sample_id"]: row for row in csv.DictReader(stream)}
    common, truth = [], []
    for index, sample_id in enumerate(sample_ids):
        row = manifest.get(sample_id)
        if row and row["split"] == "test" and row["valid_coarse_label"] == "True" and row["valid_stroke_side"] == "True":
            common.append(index)
            truth.append(COARSE_CLASSES.index(row["coarse_label"]))
    to_coarse = np.full(25, -1)
    for index, name in enumerate(get_merged_stroke_types()):
        raw = name.removeprefix("Top_").removeprefix("Bottom_")
        if raw in RAW_TO_COARSE:
            to_coarse[index] = COARSE_CLASSES.index(RAW_TO_COARSE[raw])
    data.update(
        common=np.asarray(common), truth=np.asarray(truth),
        sample_ids=np.asarray(sample_ids)[common], to_coarse=to_coarse,
        model_class=BST_CG_AP,
    )
    return data


def degrade_bst(data: dict, condition: str, rate: float, rng: np.random.Generator):
    joints, shuttle = data["joints"].copy(), data["shuttle"].copy()
    valid = np.arange(joints.shape[1])[None, :] < data["lengths"][:, None]
    if rate > 0 and condition in ("pose", "both"):
        drop = (rng.random(joints.shape[:4]) < rate) & valid[:, :, None, None]
        joints[drop] = 0.0
    if rate > 0 and condition in ("shuttle", "both"):
        drop = (rng.random(shuttle.shape[:2]) < rate) & valid
        shuttle[drop] = 0.0
    if rate > 0 and condition == "player":
        drop = (rng.random(joints.shape[:3]) < rate) & valid[:, :, None]
        joints[drop] = 0.0
    if rate > 0 and condition == "burst":
        lengths = np.rint(rate * data["lengths"]).astype(int)
        starts = (rng.random(len(lengths)) * (data["lengths"] - lengths + 1)).astype(int)
        steps = np.arange(joints.shape[1])[None, :]
        drop = (steps >= starts[:, None]) & (steps < (starts + lengths)[:, None])
        joints[drop] = 0.0
        shuttle[drop] = 0.0
    return data["make_pose"](joints), shuttle


@torch.inference_mode()
def bst_predict(model, data: dict, pose: np.ndarray, shuttle: np.ndarray, device: torch.device) -> np.ndarray:
    logits = []
    for start in range(0, len(pose), 128):
        sl = slice(start, start + 128)
        human = torch.from_numpy(pose[sl]).to(device)
        human = human.view(*human.shape[:-2], -1)
        logits.append(model(
            human, torch.from_numpy(shuttle[sl]).to(device),
            torch.from_numpy(data["pos"][sl]).to(device),
            torch.from_numpy(data["lengths"][sl]).to(device),
        ).cpu())
    probability25 = torch.cat(logits)[data["common"]].softmax(1).numpy()
    probability8 = np.zeros((len(data["common"]), len(COARSE_CLASSES)), dtype=np.float32)
    for source, target in enumerate(data["to_coarse"]):
        if target >= 0:
            probability8[:, target] += probability25[:, source]
    return probability8


def gate_response(model, tensors: dict, device: torch.device) -> dict:
    """Mean shuttle gate for events with little vs almost total shuttle loss (mixed-rate masks)."""
    rng = np.random.default_rng(0)
    rates = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=len(tensors["rgb"]))
    shuttle = tensors["shuttle"].masked_fill(
        torch.from_numpy(rng.random(tensors["shuttle"].shape[:2]) < rates[:, None]).unsqueeze(-1), 0.0)
    degraded = {**tensors, "shuttle": shuttle,
                "quality": tensors["quality"] + VideoDomainAugmentor._quality(tensors["pose"], shuttle)
                - VideoDomainAugmentor._quality(tensors["pose"], tensors["shuttle"])}
    names = ("rgb", "pose", "court", "shuttle", "contact_dist", "quality")
    gates = []
    with torch.inference_mode():
        for start in range(0, len(shuttle), 512):
            gates.append(model(*[degraded[n][start:start + 512].to(device) for n in names])[2].cpu())
    gates = torch.cat(gates).numpy()
    missing = (shuttle.abs().sum(-1) < 1e-8).float().mean(1).numpy()
    return {
        "g_sh_low_loss": float(gates[missing < 0.1, 3].mean()),
        "g_sh_high_loss": float(gates[missing > 0.9, 3].mean()),
        "g_sh_corr": float(np.corrcoef(missing, gates[:, 3])[0, 1]),
        "g_pose_min": float(gates[:, 1].min()), "g_pose_max": float(gates[:, 1].max()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict = {"rates": list(RATES), "conditions": list(CONDITIONS), "kap": {}, "bst": {}, "gate": {}}
    clean_predictions: dict[str, np.ndarray] = {}
    # Keys evaluated per model: every (condition, rate) plus total shuttle loss.
    keys = [(c, r) for c in CONDITIONS for r in RATES] + [("shuttle", 1.0), ("pose", 1.0), ("both", 1.0)]

    tensors, kap_ids, kap_truth = kap_inputs(args.features)
    order = np.argsort(kap_ids)
    for run in sorted(args.runs.glob("*_seed*/kap_best.pth")):
        kwargs = json.loads((run.parent / "model_kwargs.json").read_text(encoding="utf-8"))
        model = KAPAblationModel(**kwargs)
        model.load_state_dict(torch.load(run, map_location="cpu", weights_only=False)["model"])
        model.to(device).eval()
        name = run.parent.name
        table = {}
        for condition, rate in keys:
            degraded = degrade_kap(tensors, condition, rate, mask_rng(args.mask_seed, condition, rate, "kap"))
            probability = kap_predict(model, degraded, device)
            table[f"{condition}@{rate:.1f}"] = scores(kap_truth, probability.argmax(1))
            if rate == 0 and condition == "shuttle":
                clean_predictions[f"kap/{name}"] = probability[order]
        results["kap"][name] = table
        if kwargs["gate"] == "sigmoid":
            results["gate"][name] = gate_response(model, tensors, device)
        print(name, {k: round(v["accuracy"] * 100, 2) for k, v in table.items() if k.startswith("both")}, flush=True)

    data = bst_setup(args)
    bst_order = np.argsort(data["sample_ids"])
    if not np.array_equal(data["sample_ids"][bst_order], kap_ids[order]):
        raise RuntimeError("BST and KAP test events differ")
    for checkpoint in args.bst_checkpoints:
        model = data["model_class"](in_dim=72, seq_len=100, n_class=25, depth_tem=2, depth_inter=1)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.to(device).eval()
        table = {}
        for condition, rate in keys:
            pose, shuttle = degrade_bst(data, condition, rate, mask_rng(args.mask_seed, condition, rate, "bst"))
            probability = bst_predict(model, data, pose, shuttle, device)
            table[f"{condition}@{rate:.1f}"] = scores(data["truth"], probability.argmax(1))
            if rate == 0 and condition == "shuttle":
                clean_predictions[f"bst/{checkpoint.stem}"] = probability[bst_order]
        results["bst"][checkpoint.stem] = table
        print(checkpoint.stem, {k: round(v["accuracy"] * 100, 2) for k, v in table.items() if k.startswith("both")}, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.output.with_name("clean_predictions.npz"),
        sample_id=kap_ids[order], truth=kap_truth[order],
        **{key.replace("/", "__"): value for key, value in clean_predictions.items()},
    )


if __name__ == "__main__":
    main()
