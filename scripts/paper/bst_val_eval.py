"""Score the official BST-CG-AP checkpoints on validation Match 41 (8-class view).

BST's own validation split (matches 35-38, 41) contains Match 41, which is the
KAP model-selection set, so KAP configurations can be compared with BST without
looking at the test matches.  Samples are collated with the BST repository's
own functions (pad/subsample to 100 steps, bones from joints); the collation is
checked against the released collated test arrays first.

Example:
    python scripts/paper/bst_val_eval.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BST_ROOT = REPO_ROOT / "external/BST_official"
NPY_ROOT = Path(r"C:\Users\ADMIN\Downloads\dataset_npy_between_2_hits_with_max_limits\dataset_npy_between_2_hits_with_max_limits")
sys.path.insert(0, str(BST_ROOT / "stroke_classification"))
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "training", REPO_ROOT / "scripts" / "paper"):
    sys.path.insert(0, str(path))

from model.bst import BST_CG_AP  # noqa: E402
from preparing_data.shuttleset_dataset import (  # noqa: E402
    create_bones, get_bone_pairs, get_merged_stroke_types, make_seq_len_same,
)
from degradation_eval import COARSE_CLASSES, RAW_TO_COARSE, parse_args as degradation_args, scores  # noqa: E402

SEQ_LEN = 100
PAIRS = get_bone_pairs("coco")


def collate(branch: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Same steps as prepare_train_on_shuttleset_merged.pad_and_augment_one_npy_video."""
    joints = np.load(f"{branch}_joints.npy").astype(np.float32)
    pos = np.load(f"{branch}_pos.npy").astype(np.float32)
    shuttle = np.load(f"{branch}_shuttle.npy").astype(np.float32)
    joints, pos, shuttle, length = make_seq_len_same(SEQ_LEN, joints, pos, shuttle)
    pose = np.concatenate((joints, create_bones(joints, PAIRS)), axis=-2)
    return pose, pos, shuttle, length


def branches(split: str) -> list[Path]:
    out = []
    for class_dir in (NPY_ROOT / split).iterdir():
        out.extend(Path(str(p).removesuffix("_pos.npy")) for p in sorted(class_dir.glob("*_pos.npy")))
    return out


def main() -> None:
    # 1. Check the collation against the released collated test arrays.
    collated = BST_ROOT / "downloads/shuttleset_test_collated_seq100/test"
    released_pose = np.load(collated / "JnB_bone.npy", mmap_mode="r")
    released_len = np.load(collated / "videos_len.npy")
    test_branches = branches("test")
    for index in range(0, len(test_branches), 700):
        pose, _, _, length = collate(test_branches[index])
        if length != released_len[index] or not np.allclose(pose, released_pose[index], atol=1e-5):
            raise RuntimeError(f"collation mismatch at test index {index}")
    print("collation matches released test arrays", flush=True)

    # 2. Collate Match 41 events with a valid 8-class label.
    with (REPO_ROOT / "data/manifests/shuttleset_npy.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest = {row["sample_id"]: row for row in csv.DictReader(stream)}
    samples, truth, ids = [], [], []
    for branch in branches("val"):
        sample_id = branch.name
        row = manifest.get(sample_id)
        if not row or row["match_id"] != "41":
            continue
        if row["valid_coarse_label"] != "True" or row["valid_stroke_side"] != "True":
            continue
        samples.append(collate(branch))
        truth.append(COARSE_CLASSES.index(row["coarse_label"]))
        ids.append(sample_id)
    truth = np.asarray(truth)
    pose = np.stack([s[0] for s in samples]); pos = np.stack([s[1] for s in samples])
    shuttle = np.stack([s[2] for s in samples]); lengths = np.asarray([s[3] for s in samples])
    print(f"Match 41 events: {len(truth)}", flush=True)

    to_coarse = np.full(25, -1)
    for index, name in enumerate(get_merged_stroke_types()):
        raw = name.removeprefix("Top_").removeprefix("Bottom_")
        if raw in RAW_TO_COARSE:
            to_coarse[index] = COARSE_CLASSES.index(RAW_TO_COARSE[raw])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for checkpoint in degradation_args().bst_checkpoints:
        model = BST_CG_AP(in_dim=72, seq_len=SEQ_LEN, n_class=25, depth_tem=2, depth_inter=1)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.to(device).eval()
        logits = []
        with torch.inference_mode():
            for start in range(0, len(truth), 128):
                sl = slice(start, start + 128)
                human = torch.from_numpy(pose[sl]).to(device)
                logits.append(model(
                    human.view(*human.shape[:-2], -1), torch.from_numpy(shuttle[sl]).to(device),
                    torch.from_numpy(pos[sl]).to(device), torch.from_numpy(lengths[sl]).to(device),
                ).cpu())
        probability25 = torch.cat(logits).softmax(1).numpy()
        probability8 = np.zeros((len(truth), 8), dtype=np.float32)
        for source, target in enumerate(to_coarse):
            if target >= 0:
                probability8[:, target] += probability25[:, source]
        results[checkpoint.stem] = scores(truth, probability8.argmax(1))
        print(checkpoint.stem, {k: round(v * 100, 2) for k, v in results[checkpoint.stem].items()}, flush=True)
    out = REPO_ROOT / "work_dirs/paper_ablation/bst_val_match41.json"
    out.write_text(json.dumps({"events": len(truth), "results": results}, indent=2) + "\n", encoding="utf-8")
    print("mean acc", np.mean([r["accuracy"] for r in results.values()]) * 100,
          "mean f1", np.mean([r["macro_f1"] for r in results.values()]) * 100)


if __name__ == "__main__":
    main()
