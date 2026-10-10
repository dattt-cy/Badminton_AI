"""Train a robust Rally Boundary Classifier (same_rally vs new_rally) for Level-2 Rally Analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
STROKES = ["clear", "drive", "drop", "lift", "net_attack", "net_shot", "serve", "smash"]
STROKE2IDX = {s: i for i, s in enumerate(STROKES)}


def extract_features_and_labels(
    df: pd.DataFrame, noise_rate: float = 0.15, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    X, y = [], []
    for (mid, sid), group in df.groupby(["match_id", "set_id"], sort=False):
        prev_f = None
        prev_side = None
        prev_stroke = None
        prev_rally = None
        fps = 30.0
        for _, row in group.iterrows():
            curr_stroke = row["coarse_label"]
            if noise_rate > 0 and np.random.rand() < noise_rate:
                curr_stroke = np.random.choice(STROKES)

            if prev_f is not None:
                gap = max((row["hit_frame"] - prev_f) / fps, 0.0)
                same_side = 1.0 if row["player_side"] == prev_side else 0.0
                curr_is_serve = 1.0 if curr_stroke == "serve" else 0.0
                prev_is_serve = 1.0 if prev_stroke == "serve" else 0.0
                curr_s_idx = STROKE2IDX.get(curr_stroke, -1)
                prev_s_idx = STROKE2IDX.get(prev_stroke, -1)

                feat = [
                    gap,
                    float(np.log(max(gap, 0.01))),
                    same_side,
                    curr_is_serve,
                    prev_is_serve,
                    float(curr_s_idx),
                    float(prev_s_idx),
                ]
                is_new = 1 if row["rally"] != prev_rally else 0
                X.append(feat)
                y.append(is_new)
            prev_f = row["hit_frame"]
            prev_side = row["player_side"]
            prev_stroke = curr_stroke
            prev_rally = row["rally"]
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data" / "manifests" / "shuttleset_rgb.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "work_dirs" / "rally_boundary_detector.joblib",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    df = df.sort_values(by=["match_id", "set_id", "hit_frame"]).reset_index(drop=True)

    train_matches = [str(i) for i in range(1, 35)]
    val_matches = ["35", "36", "37", "38", "41"]
    test_matches = ["39", "40", "42", "43", "44", "16"]

    train_df = df[df["match_id"].astype(str).isin(train_matches)]
    val_df = df[df["match_id"].astype(str).isin(val_matches)]
    test_df = df[df["match_id"].astype(str).isin(test_matches)]

    print(f"Extracting features: Train ({len(train_df)} hits), Val ({len(val_df)} hits), Test ({len(test_df)} hits)...")
    X_train, y_train = extract_features_and_labels(train_df, noise_rate=0.15)
    X_val, y_val = extract_features_and_labels(val_df, noise_rate=0.20)
    X_test, y_test = extract_features_and_labels(test_df, noise_rate=0.20)

    clf = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=150,
        learning_rate=0.08,
        min_samples_leaf=20,
    )
    print("Training HistGradientBoostingClassifier...")
    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, pos_label=1)
    print("\n--- VALIDATION SET (Matches 35-38, 41) ---")
    print(classification_report(y_val, val_preds, target_names=["same_rally", "new_rally"], digits=4))
    print(f"Validation Rally-Boundary F1: {val_f1 * 100:.2f}%")

    test_preds = clf.predict(X_test)
    test_f1 = f1_score(y_test, test_preds, pos_label=1)
    print("\n--- TEST HELD-OUT SET (Matches 39, 40, 42, 43, 44, 16) ---")
    print(classification_report(y_test, test_preds, target_names=["same_rally", "new_rally"], digits=4))
    print(f"Test Rally-Boundary F1: {test_f1 * 100:.2f}%")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": clf,
        "features": [
            "gap",
            "log_gap",
            "same_side",
            "curr_is_serve",
            "prev_is_serve",
            "curr_s_idx",
            "prev_s_idx",
        ],
        "stroke2idx": STROKE2IDX,
        "val_f1": float(val_f1),
        "test_f1": float(test_f1),
    }
    joblib.dump(payload, args.output)
    print(f"\n[DONE] Saved Rally Boundary Detector to: {args.output}")


if __name__ == "__main__":
    main()

