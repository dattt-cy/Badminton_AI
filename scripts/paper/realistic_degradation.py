"""Degrade the ShuttleSet test events with the error profile measured on the deployed pipeline.

The profile comes from the per-hit caches of the untrimmed test clips (see
pipeline_errors.py), not from chosen rates:
  * shuttle: every test event receives the 46-frame lost-shuttle mask of a randomly
    drawn pipeline hit, stretched to the model's own sequence (32 steps for KAP,
    the event's valid length for BST), so gap positions and coverage are kept;
  * joints: each (frame, player, joint) is removed with the per-joint rate the
    pipeline shows (mostly face keypoints below the 0.20 confidence threshold);
  * players: a whole skeleton is removed per (frame, player) with the pipeline rate.

Two versions are scored: ``pipeline`` (the profile as measured) and ``pipeline_x2``
(shuttle masks drawn only from pipeline hits that lost shuttle frames, joint and
player rates doubled), as a harder but still observed-shape condition.
Each condition is repeated over ``--draws`` mask seeds; KAP and BST see masks from
the same draw.  The accuracy difference gets a paired bootstrap over events.

Example:
    python scripts/paper/realistic_degradation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "training", REPO_ROOT / "scripts" / "paper"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import degradation_eval as de  # noqa: E402
from pipeline_errors import TEST_CLIPS  # noqa: E402
from train_kap_ablation import KAPAblationModel  # noqa: E402
from train_kap_scratch import VideoDomainAugmentor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "work_dirs/kap_ab_8clips/feature_cache")
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "work_dirs/paper_final")
    parser.add_argument("--families", nargs="+", default=["main", "no_rgb"])
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "work_dirs/paper_lan4/realistic_degradation.json")
    return parser.parse_args()


def pipeline_profile(cache: Path) -> dict:
    masks, joint_zero, player_zero = [], [], []
    for clip in TEST_CLIPS:
        cached = torch.load(cache / f"{clip}.pt", map_location="cpu", weights_only=False)
        for item in cached["features"]:
            context = item["feature"]["analytics_context"]
            masks.append(np.abs(np.asarray(context["shuttle_image"])).sum(-1) == 0)
            pose = item["feature"]["s_j"][0].numpy().reshape(-1, 2, 17, 2)
            zero = np.abs(pose).sum(-1) == 0
            whole = zero.all(-1)
            joint_zero.append((zero & ~whole[..., None]).reshape(-1, 17))
            player_zero.append(whole.reshape(-1))
    masks = np.stack(masks)
    joint_rate = np.concatenate(joint_zero).mean(0)
    # Remove a joint given that the player is present: rate among present player-frames.
    return {"masks": masks, "joint_rate": joint_rate, "player_rate": float(np.concatenate(player_zero).mean())}


def stretch(mask: np.ndarray, length: int) -> np.ndarray:
    """Nearest-neighbour resampling of a 1-D mask to ``length`` steps."""
    index = np.minimum((np.arange(length) + 0.5) * len(mask) / length, len(mask) - 1).astype(int)
    return mask[index]


def draw(profile: dict, count: int, hard: bool, rng: np.random.Generator) -> dict:
    pool = profile["masks"]
    if hard:
        pool = pool[pool.any(1)]
    scale = 2.0 if hard else 1.0
    return {"shuttle": pool[rng.integers(0, len(pool), count)],
            "joint_rate": np.minimum(profile["joint_rate"] * scale, 1.0),
            "player_rate": min(profile["player_rate"] * scale, 1.0)}


def degrade_kap(tensors: dict, plan: dict, rng: np.random.Generator) -> dict:
    pose, shuttle = tensors["pose"].clone(), tensors["shuttle"].clone()
    count, frames = pose.shape[:2]
    lost = np.stack([stretch(m, frames) for m in plan["shuttle"]])
    shuttle = shuttle.masked_fill(torch.from_numpy(lost).unsqueeze(-1), 0.0)
    joints = pose.reshape(count, frames, 2, 17, 2)
    drop = rng.random((count, frames, 2, 17)) < plan["joint_rate"]
    drop |= (rng.random((count, frames, 2)) < plan["player_rate"])[..., None]
    pose = joints.masked_fill(torch.from_numpy(drop).unsqueeze(-1), 0.0).reshape_as(pose)
    before = VideoDomainAugmentor._quality(tensors["pose"], tensors["shuttle"])
    after = VideoDomainAugmentor._quality(pose, shuttle)
    return {**tensors, "pose": pose, "shuttle": shuttle, "quality": tensors["quality"] + after - before}


def degrade_bst(data: dict, plan: dict, order: np.ndarray, rng: np.random.Generator):
    """``plan`` is indexed like the sorted common events; BST arrays hold every collated sample."""
    joints, shuttle = data["joints"].copy(), data["shuttle"].copy()
    frames = joints.shape[1]
    rows = data["common"][order]
    for event, row in enumerate(rows):
        n = int(data["lengths"][row])
        shuttle[row, :n][stretch(plan["shuttle"][event], n)] = 0.0
    valid = np.arange(frames)[None, :] < data["lengths"][:, None]
    drop = rng.random(joints.shape[:4]) < plan["joint_rate"]
    drop |= (rng.random(joints.shape[:3]) < plan["player_rate"])[..., None]
    joints[drop & valid[:, :, None, None]] = 0.0
    return data["make_pose"](joints), shuttle


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profile = pipeline_profile(args.cache)
    tensors, kap_ids, kap_truth = de.kap_inputs(REPO_ROOT / "work_dirs/context_window_pilot/features_w15_30.npz")
    kap_order = np.argsort(kap_ids)
    sys.argv = sys.argv[:1]
    data = de.bst_setup(de.parse_args())
    bst_order = np.argsort(data["sample_ids"])
    if not np.array_equal(data["sample_ids"][bst_order], kap_ids[kap_order]):
        raise RuntimeError("BST and KAP test events differ")
    truth = kap_truth[kap_order]
    count = len(truth)

    kap_models = {}
    for family in args.families:
        for run in sorted(args.runs.glob(f"{family}_seed*/kap_best.pth")):
            kwargs = json.loads((run.parent / "model_kwargs.json").read_text(encoding="utf-8"))
            model = KAPAblationModel(**kwargs)
            model.load_state_dict(torch.load(run, map_location="cpu", weights_only=False)["model"])
            kap_models.setdefault(family, []).append(model.to(device).eval())
    bst_models = []
    for checkpoint in de.parse_args().bst_checkpoints:
        model = data["model_class"](in_dim=72, seq_len=100, n_class=25, depth_tem=2, depth_inter=1)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        bst_models.append(model.to(device).eval())

    result = {"profile": {"pipeline_hits": int(len(profile["masks"])),
                          "shuttle_frames_lost": float(profile["masks"].mean()),
                          "joint_rate": [float(v) for v in profile["joint_rate"]],
                          "player_rate": profile["player_rate"]},
              "families": {f: len(m) for f, m in kap_models.items()}, "bst_checkpoints": len(bst_models),
              "draws": args.draws, "conditions": {}}
    rng_boot = np.random.default_rng(20261109)
    for condition, hard in (("clean", None), ("pipeline", False), ("pipeline_x2", True)):
        correct = {name: [] for name in [*kap_models, "bst"]}
        for d in range(1 if hard is None else args.draws):
            rng = np.random.default_rng([20261109, d, int(bool(hard))])
            if hard is None:
                kap_in, (bst_pose, bst_shuttle) = tensors, (data["make_pose"](data["joints"]), data["shuttle"])
            else:
                plan = draw(profile, count, hard, rng)
                # KAP tensors are in file order; map the sorted plan back to it.
                inverse = np.empty_like(kap_order); inverse[kap_order] = np.arange(count)
                kap_plan = {**plan, "shuttle": plan["shuttle"][inverse]}
                kap_in = degrade_kap(tensors, kap_plan, np.random.default_rng([20261109, d, 7]))
                bst_pose, bst_shuttle = degrade_bst(data, plan, bst_order, np.random.default_rng([20261109, d, 8]))
            for family, models in kap_models.items():
                for model in models:
                    p = de.kap_predict(model, kap_in, device)[kap_order]
                    correct[family].append(p.argmax(1) == truth)
            for model in bst_models:
                p = de.bst_predict(model, data, bst_pose, bst_shuttle, device)[bst_order]
                correct["bst"].append(p.argmax(1) == truth)
        table = {name: float(np.mean(v) * 100) for name, v in correct.items()}
        per_event = {name: np.mean(v, axis=0) for name, v in correct.items()}
        for family in kap_models:
            diff = per_event[family] - per_event["bst"]
            boot = diff[rng_boot.integers(0, count, (args.bootstrap, count))].mean(1)
            table[f"{family}-bst"] = float(diff.mean() * 100)
            table[f"{family}-bst_ci95"] = [float(np.quantile(boot, 0.025) * 100), float(np.quantile(boot, 0.975) * 100)]
        table["std_over_runs"] = {name: float(np.std([c.mean() for c in v]) * 100) for name, v in correct.items()}
        result["conditions"][condition] = table
        print(condition, json.dumps(table), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
