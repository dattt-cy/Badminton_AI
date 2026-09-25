"""Train a lightweight ShuttleSet hit-frame candidate classifier.

This baseline classifies a candidate frame as no-hit, upper-player hit, or
lower-player hit from coarse motion around that frame. Matches remain disjoint
between train, validation, and test.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


CLASSES = ["no_hit", "upper_hit", "lower_hit"]
SIDE_TO_LABEL = {"top": 1, "upper": 1, "bottom": 2, "lower": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-match-ids", nargs="+", required=True)
    parser.add_argument("--val-match-ids", nargs="+", required=True)
    parser.add_argument("--test-match-ids", nargs="+", required=True)
    parser.add_argument("--positives-per-match", type=int, default=200)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--negative-exclusion", type=int, default=12)
    parser.add_argument("--frame-radius", type=int, default=4)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=54)
    parser.add_argument("--grid-cols", type=int, default=6)
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def load_matches(path: Path) -> dict[str, dict]:
    matches: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            match_id = row["match_id"]
            item = matches.setdefault(
                match_id,
                {"video_path": Path(row["video_path"]), "hits": []},
            )
            side = row["player_side"].strip().lower()
            if side in SIDE_TO_LABEL:
                item["hits"].append((int(float(row["hit_frame"])), SIDE_TO_LABEL[side]))
    for item in matches.values():
        item["hits"] = sorted(set(item["hits"]))
    return matches


def sample_centers(
    hits: list[tuple[int, int]], count: int, negative_ratio: float,
    exclusion: int, rng: random.Random,
) -> list[tuple[int, int]]:
    positives = hits.copy()
    rng.shuffle(positives)
    positives = positives[: min(count, len(positives))]
    all_hit_frames = np.asarray([frame for frame, _ in hits], dtype=np.int64)
    low, high = int(all_hit_frames.min()), int(all_hit_frames.max())
    negative_count = int(round(len(positives) * negative_ratio))
    negatives: set[int] = set()
    attempts = 0
    while len(negatives) < negative_count and attempts < negative_count * 100:
        attempts += 1
        frame = rng.randint(low, high)
        nearest = int(np.min(np.abs(all_hit_frames - frame)))
        if nearest > exclusion:
            negatives.add(frame)
    samples = positives + [(frame, 0) for frame in negatives]
    rng.shuffle(samples)
    return samples


def read_gray(cap: cv2.VideoCapture, frame_index: int, width: int, height: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    ok, frame = cap.read()
    if not ok:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def grid_stats(image: np.ndarray, rows: int, cols: int) -> list[float]:
    features: list[float] = []
    for row in np.array_split(image, rows, axis=0):
        for cell in np.array_split(row, cols, axis=1):
            features.extend((float(cell.mean()), float(cell.std()), float(np.quantile(cell, 0.9))))
    return features


def extract_feature(
    cap: cv2.VideoCapture, center: int, radius: int, width: int, height: int,
    grid_rows: int, grid_cols: int,
) -> np.ndarray | None:
    before = read_gray(cap, center - radius, width, height)
    middle = read_gray(cap, center, width, height)
    after = read_gray(cap, center + radius, width, height)
    if before is None or middle is None or after is None:
        return None
    pre_motion = np.abs(middle - before)
    post_motion = np.abs(after - middle)
    span_motion = np.abs(after - before)
    features: list[float] = []
    for motion in (pre_motion, post_motion, span_motion):
        features.extend(grid_stats(motion, grid_rows, grid_cols))
    top = span_motion[: height // 2]
    bottom = span_motion[height // 2 :]
    features.extend(
        [
            float(pre_motion.mean()), float(post_motion.mean()), float(span_motion.mean()),
            float(top.mean()), float(bottom.mean()),
            float(top.mean() - bottom.mean()), float(middle.mean()), float(middle.std()),
        ]
    )
    return np.asarray(features, dtype=np.float32)


def build_split(
    matches: dict[str, dict], match_ids: list[str], args: argparse.Namespace,
    rng: random.Random, split_name: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    features, labels, rows = [], [], []
    for match_id in match_ids:
        if match_id not in matches:
            raise ValueError(f"Unknown match_id {match_id!r}")
        item = matches[match_id]
        cap = cv2.VideoCapture(str(item["video_path"]))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open {item['video_path']}")
        samples = sample_centers(
            item["hits"], args.positives_per_match, args.negative_ratio,
            args.negative_exclusion, rng,
        )
        kept = 0
        for center, label in samples:
            feature = extract_feature(
                cap, center, args.frame_radius, args.width, args.height,
                args.grid_rows, args.grid_cols,
            )
            if feature is None:
                continue
            features.append(feature)
            labels.append(label)
            rows.append({"match_id": match_id, "frame": center, "label": CLASSES[label]})
            kept += 1
        cap.release()
        print(f"split={split_name} match={match_id} samples={kept}", flush=True)
    return np.stack(features), np.asarray(labels, dtype=np.int64), rows


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(3)).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=range(3), target_names=CLASSES,
            output_dict=True, zero_division=0,
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matches = load_matches(args.manifest)
    rng = random.Random(args.seed)
    x_train, y_train, _ = build_split(matches, args.train_match_ids, args, rng, "train")
    x_val, y_val, _ = build_split(matches, args.val_match_ids, args, rng, "val")
    x_test, y_test, test_rows = build_split(matches, args.test_match_ids, args, rng, "test")

    model = RandomForestClassifier(
        n_estimators=args.trees,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(x_train, y_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    val_metrics = metrics(y_val, val_pred)
    test_metrics = metrics(y_test, test_pred)
    for row, prediction, probabilities in zip(test_rows, test_pred, model.predict_proba(x_test)):
        row["prediction"] = CLASSES[int(prediction)]
        row["probabilities"] = {name: float(value) for name, value in zip(CLASSES, probabilities)}

    artifact = {
        "classes": CLASSES,
        "train_match_ids": args.train_match_ids,
        "val_match_ids": args.val_match_ids,
        "test_match_ids": args.test_match_ids,
        "feature_config": {
            "frame_radius": args.frame_radius, "width": args.width, "height": args.height,
            "grid_cols": args.grid_cols, "grid_rows": args.grid_rows,
        },
        "val": val_metrics,
        "test": test_metrics,
        "test_predictions": test_rows,
    }
    joblib.dump(model, args.output_dir / "hit_detector.joblib")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"val_macro_f1={val_metrics['macro_f1']:.4f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
