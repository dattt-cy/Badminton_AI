"""Rebuild the canonical full-ShuttleSet feature cache after artifact loss."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from ai_classifier.models import SIDE_CLASSES, STROKE_CLASSES
from scripts.evaluation.evaluate_kap_full_shuttleset_test import RGB, extract_sample_sequences, norm_clip


def main() -> None:
    manifest = ROOT / "data/manifests/shuttleset_npy.csv"
    rgb_cache = Path(r"C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_v3_full\frames_16")
    npy_root = Path(r"C:\Users\ADMIN\Downloads\dataset_npy_between_2_hits_with_max_limits\dataset_npy_between_2_hits_with_max_limits")
    rgb_checkpoint = ROOT / "src/ai_classifier/models/checkpoints/visual_backbone.pth"
    output = ROOT / "work_dirs/recovery_full_kap/sequence_features.npz"
    output.parent.mkdir(parents=True, exist_ok=True)

    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    rows = [
        row for row in source
        if row["valid_coarse_label"] == "True"
        and row["valid_stroke_side"] == "True"
        and (rgb_cache / f"{row['sample_id']}.npy").is_file()
        and (npy_root / row["joints_path"]).is_file()
        and (npy_root / row["position_path"]).is_file()
        and (npy_root / row["shuttle_path"]).is_file()
    ]
    print(f"[data] usable={len(rows)} manifest={len(source)}", flush=True)

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in source:
        groups[(row["match_id"], row["set_id"], row["rally"])].append(row)
    targets: dict[str, int] = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda item: int(item["ball_round"]))
        for index, row in enumerate(ordered):
            hit = int(float(row["hit_frame"]))
            previous = int(float(ordered[index - 1]["hit_frame"])) if index else hit - 15
            targets[row["sample_id"]] = hit - max(previous, hit - 45)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_model = RGB()
    rgb_model.load_state_dict(torch.load(rgb_checkpoint, map_location="cpu", weights_only=False)["model"])
    rgb_model.to(device).eval()
    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            clips = torch.stack([norm_clip(rgb_cache / f"{row['sample_id']}.npy") for row in batch]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                embeddings.extend(rgb_model.backbone(clips).float().cpu().numpy())
            if start % 1024 == 0:
                print(f"[rgb] {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)

    pose, court, shuttle, contact, quality = [], [], [], [], []
    for index, row in enumerate(rows, 1):
        features = extract_sample_sequences(
            np.load(npy_root / row["joints_path"]),
            np.load(npy_root / row["position_path"]),
            np.load(npy_root / row["shuttle_path"]),
            targets[row["sample_id"]],
        )
        pose.append(features[0]); court.append(features[1]); shuttle.append(features[2])
        contact.append(features[3]); quality.append(features[4])
        if index % 5000 == 0:
            print(f"[structured] {index}/{len(rows)}", flush=True)

    np.savez_compressed(
        output,
        rgb=np.asarray(embeddings, dtype=np.float32),
        pose=np.asarray(pose, dtype=np.float32),
        court=np.asarray(court, dtype=np.float32),
        shuttle=np.asarray(shuttle, dtype=np.float32),
        contact_dist=np.asarray(contact, dtype=np.float32),
        quality=np.asarray(quality, dtype=np.float32),
        stroke=np.asarray([STROKE_CLASSES.index(row["coarse_label"]) for row in rows], dtype=np.int64),
        side=np.asarray([SIDE_CLASSES.index(row["stroke_side"]) for row in rows], dtype=np.int64),
        split=np.asarray([row["split"] for row in rows]),
        sample_id=np.asarray([row["sample_id"] for row in rows]),
        valid=np.ones(len(rows), dtype=np.bool_),
    )
    print(f"[done] {output} ({output.stat().st_size / 2**20:.1f} MiB)", flush=True)


if __name__ == "__main__":
    main()
