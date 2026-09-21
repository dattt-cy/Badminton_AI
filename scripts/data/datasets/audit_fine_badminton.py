"""Audit Fine-Badminton and build a leakage-safe virtual-clip manifest."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml


EXPECTED_COLUMNS = {"start", "end", "label", "location"}
LOCATION_MAP = {
    "Upper Half": "upper",
    "Lower Half": "lower",
    "上半区": "upper",
    "下半区": "lower",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("configs/taxonomy/fine_badminton_8class.yaml"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fine_badminton_8class.csv"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("data/manifests/fine_badminton_audit.json"),
    )
    parser.add_argument(
        "--split-manifest-dir",
        type=Path,
        default=Path("data/manifests/fine_badminton_splits"),
    )
    return parser.parse_args()


def load_taxonomy(path: Path) -> tuple[dict[str, tuple[str, str]], list[str]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    classes = [str(item) for item in config["coarse_classes"]]
    aliases: dict[str, tuple[str, str]] = {}
    for canonical, definition in config["canonical_labels"].items():
        coarse = str(definition["coarse"])
        if coarse not in classes:
            raise ValueError(f"Unknown coarse class {coarse!r} for {canonical!r}")
        for alias in definition["aliases"]:
            alias = str(alias).strip()
            if alias in aliases:
                raise ValueError(f"Duplicate taxonomy alias: {alias!r}")
            aliases[alias] = (str(canonical), coarse)
    return aliases, classes


def video_metadata(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid video metadata for {path}: frames={frames}, fps={fps}, "
            f"size={width}x{height}"
        )
    return {
        "frame_count": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": frames / fps,
        "size_bytes": path.stat().st_size,
    }


def annotation_key(path: Path) -> str:
    suffix = "_dataframe"
    if not path.stem.endswith(suffix):
        raise ValueError(f"Unexpected annotation filename: {path.name}")
    return path.stem[: -len(suffix)]


def choose_group_split(
    matches: list[str], counts: dict[str, Counter[str]], classes: list[str]
) -> dict[str, str]:
    """Choose a deterministic 6/2/2 split with similar class distributions."""
    global_counts = Counter()
    for match in matches:
        global_counts.update(counts[match])
    total = sum(global_counts.values())

    best_score: float | None = None
    best: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None = None
    for test in itertools.combinations(matches, 2):
        remaining = [item for item in matches if item not in test]
        for val in itertools.combinations(remaining, 2):
            train = tuple(item for item in remaining if item not in val)
            groups = {"train": train, "val": val, "test": test}
            group_counts: dict[str, Counter[str]] = {}
            missing_penalty = 0.0
            score = 0.0
            for split, members in groups.items():
                aggregate = Counter()
                for member in members:
                    aggregate.update(counts[member])
                group_counts[split] = aggregate
                missing_penalty += 1000.0 * sum(
                    1 for class_name in classes if aggregate[class_name] == 0
                )

            targets = {"train": 0.60, "val": 0.20, "test": 0.20}
            for split, target in targets.items():
                aggregate = group_counts[split]
                score += 3.0 * abs(sum(aggregate.values()) / total - target)
                for class_name in classes:
                    score += abs(
                        aggregate[class_name] / global_counts[class_name] - target
                    )
                # Prefer useful evaluation support, not just non-zero support.
                score += sum(
                    max(0, 20 - aggregate[class_name]) / 20
                    for class_name in classes
                )
            score += missing_penalty
            candidate = (tuple(train), tuple(val), tuple(test))
            if best_score is None or score < best_score or (
                score == best_score and candidate < best
            ):
                best_score = score
                best = candidate

    if best is None:
        raise RuntimeError("Could not create a grouped split")
    train, val, test = best
    return {
        match: split
        for split, members in (("train", train), ("val", val), ("test", test))
        for match in members
    }


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    annotations_dir = root / "annotations"
    videos_dir = root / "videos"
    if not annotations_dir.is_dir() or not videos_dir.is_dir():
        raise FileNotFoundError(
            f"Expected annotations/ and videos/ under {root}"
        )

    aliases, classes = load_taxonomy(args.taxonomy)
    annotation_paths = sorted(annotations_dir.glob("*.csv"), key=lambda p: p.name)
    video_paths = sorted(videos_dir.glob("*.mp4"), key=lambda p: p.name)
    video_by_key = {path.stem: path for path in video_paths}
    if len(annotation_paths) != 10 or len(video_paths) != 10:
        raise ValueError(
            f"Expected 10 annotations and 10 videos, got "
            f"{len(annotation_paths)} and {len(video_paths)}"
        )

    records: list[dict[str, object]] = []
    match_summaries: dict[str, dict[str, object]] = {}
    raw_counts: Counter[str] = Counter()
    canonical_counts: Counter[str] = Counter()
    coarse_counts: Counter[str] = Counter()
    match_counts: dict[str, Counter[str]] = defaultdict(Counter)
    issues: list[dict[str, object]] = []

    for match_index, annotation_path in enumerate(annotation_paths, start=1):
        match_key = annotation_key(annotation_path)
        video_path = video_by_key.get(match_key)
        if video_path is None:
            raise FileNotFoundError(
                f"No MP4 matches annotation {annotation_path.name}"
            )
        metadata = video_metadata(video_path)
        previous_end = -1
        overlap_count = 0
        shared_boundary_count = 0
        match_records = 0
        with annotation_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or set(reader.fieldnames) != EXPECTED_COLUMNS:
                raise ValueError(
                    f"Unexpected columns in {annotation_path}: {reader.fieldnames}"
                )
            for row_index, row in enumerate(reader, start=1):
                raw_label = row["label"].strip()
                if raw_label not in aliases:
                    raise ValueError(
                        f"Unmapped label {raw_label!r} in {annotation_path.name}"
                    )
                canonical, coarse = aliases[raw_label]
                location_raw = row["location"].strip()
                if location_raw not in LOCATION_MAP:
                    raise ValueError(
                        f"Unknown location {location_raw!r} in {annotation_path.name}"
                    )
                start = int(float(row["start"]))
                end = int(float(row["end"]))
                valid = 0 <= start <= end < int(metadata["frame_count"])
                if not valid:
                    issues.append(
                        {
                            "type": "out_of_bounds",
                            "match_id": match_key,
                            "row": row_index,
                            "start": start,
                            "end": end,
                            "frame_count": metadata["frame_count"],
                        }
                    )
                if start == previous_end:
                    # The dataset frequently uses the same transition/contact
                    # frame as the end of one stroke and start of the next.
                    shared_boundary_count += 1
                elif start < previous_end:
                    # Complete stroke motions of alternating players may
                    # overlap (follow-through versus next preparation). TAL
                    # supports this, so count it but do not invalidate it.
                    overlap_count += 1
                previous_end = max(previous_end, end)
                sample_id = f"fine_badminton_m{match_index:02d}_{row_index:05d}"
                records.append(
                    {
                        "sample_id": sample_id,
                        "dataset": "fine_badminton",
                        "match_id": match_key,
                        "video_relpath": video_path.relative_to(root).as_posix(),
                        "annotation_relpath": annotation_path.relative_to(root).as_posix(),
                        "start_frame": start,
                        "hit_frame": "",
                        "end_frame": end,
                        "duration_frames": end - start + 1,
                        "player_side": LOCATION_MAP[location_raw],
                        "raw_label": raw_label,
                        "canonical_label": canonical,
                        "coarse_label": coarse,
                        "fps": metadata["fps"],
                        "frame_width": metadata["width"],
                        "frame_height": metadata["height"],
                        "split": "",
                        "annotation_source": "Fine-Badminton-v1.0",
                        "valid": valid,
                    }
                )
                raw_counts[raw_label] += 1
                canonical_counts[canonical] += 1
                coarse_counts[coarse] += 1
                match_counts[match_key][coarse] += 1
                match_records += 1

        match_summaries[match_key] = {
            **metadata,
            "annotation_file": annotation_path.name,
            "video_file": video_path.name,
            "actions": match_records,
            "overlaps": overlap_count,
            "shared_boundaries": shared_boundary_count,
            "class_counts": dict(sorted(match_counts[match_key].items())),
        }

    split_by_match = choose_group_split(
        sorted(match_summaries), match_counts, classes
    )
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_totals: Counter[str] = Counter()
    for record in records:
        split = split_by_match[str(record["match_id"])]
        record["split"] = split
        split_counts[split][str(record["coarse_label"])] += 1
        split_totals[split] += 1

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with args.manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    args.split_manifest_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        split_records = [record for record in records if record["split"] == split]
        split_path = args.split_manifest_dir / f"{split}.csv"
        with split_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_records)
        match_path = args.split_manifest_dir / f"{split}_matches.txt"
        match_path.write_text(
            "\n".join(
                sorted(
                    match
                    for match, assigned_split in split_by_match.items()
                    if assigned_split == split
                )
            )
            + "\n",
            encoding="utf-8",
        )

    duration_values = sorted(int(record["duration_frames"]) for record in records)
    audit = {
        "dataset": "Fine-Badminton",
        "dataset_root_used_for_audit": str(root),
        "taxonomy": str(args.taxonomy),
        "manifest": str(args.manifest),
        "split_manifest_dir": str(args.split_manifest_dir),
        "videos": len(video_paths),
        "annotations": len(annotation_paths),
        "actions": len(records),
        "valid_actions": sum(bool(record["valid"]) for record in records),
        "raw_label_count": len(raw_counts),
        "canonical_label_count": len(canonical_counts),
        "coarse_label_count": len(coarse_counts),
        "raw_counts": dict(raw_counts.most_common()),
        "canonical_counts": dict(canonical_counts.most_common()),
        "coarse_counts": dict(coarse_counts.most_common()),
        "duration_frames": {
            "min": duration_values[0],
            "p05": duration_values[int(len(duration_values) * 0.05)],
            "median": duration_values[len(duration_values) // 2],
            "p95": duration_values[int(len(duration_values) * 0.95)],
            "max": duration_values[-1],
        },
        "split_by_match": split_by_match,
        "split_totals": dict(split_totals),
        "split_class_counts": {
            split: {name: split_counts[split][name] for name in classes}
            for split in ("train", "val", "test")
        },
        "matches": match_summaries,
        "issues": issues,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Manifest: {args.manifest} ({len(records)} actions)")
    print(f"Audit: {args.audit_output}")
    print(f"Labels: raw={len(raw_counts)}, canonical={len(canonical_counts)}, coarse={len(coarse_counts)}")
    print("Split totals: " + ", ".join(f"{key}={split_totals[key]}" for key in ("train", "val", "test")))
    for split in ("train", "val", "test"):
        print(split + ": " + ", ".join(f"{name}={split_counts[split][name]}" for name in classes))
    print(f"Issues: {len(issues)}")


if __name__ == "__main__":
    main()
