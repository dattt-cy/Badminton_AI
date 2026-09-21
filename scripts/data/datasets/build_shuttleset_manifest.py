"""Join ShuttleSet BST arrays to annotations and build a video download list."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


TRAIN_MATCHES = set(range(1, 9)) | {11} | set(range(13, 27)) | set(range(28, 35))
VAL_MATCHES = set(range(35, 39)) | {41}
TEST_MATCHES = {39, 40, 42, 43, 44}
MODALITIES = ("joints", "pos", "shuttle")
SAMPLE_PATTERN = re.compile(
    r"^(?P<match>\d+)_(?P<set>\d+)_(?P<rally>\d+)_(?P<ball_round>\d+)_joints$"
)
RAW_TO_COARSE = {
    "切球": "drop",
    "勾球": "net_shot",
    "平球": "drive",
    "挑球": "lift",
    "推球": "net_attack",
    "撲球": "net_attack",
    "擋小球": "net_shot",
    "放小球": "net_shot",
    "殺球": "smash",
    "點扣": "smash",
    "發短球": "serve",
    "發長球": "serve",
    "長球": "clear",
    "過渡切球": "drop",
    "後場抽平球": "drive",
    "防守回抽": "drive",
    "防守回挑": "lift",
    "小平球": "drive",
    "未知球種": "unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npy_root", type=Path)
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path(
            "external/BST-Badminton-Stroke-type-Transformer/ShuttleSet/set"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/shuttleset_npy.csv"),
    )
    parser.add_argument(
        "--download-list",
        type=Path,
        default=Path("data/manifests/shuttleset_video_downloads.csv"),
    )
    parser.add_argument(
        "--download-checklist",
        type=Path,
        default=Path("docs/shuttleset_video_download_checklist.md"),
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/raw/shuttleset/videos"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("data/manifests/shuttleset_npy_audit.json"),
    )
    return parser.parse_args()


def split_for_match(match_id: int) -> str:
    if match_id in TRAIN_MATCHES:
        return "train"
    if match_id in VAL_MATCHES:
        return "val"
    if match_id in TEST_MATCHES:
        return "test"
    return "excluded"


def normalized_flag(value: str | None) -> bool:
    return str(value or "").strip() in {"1", "1.0", "true", "True"}


def stroke_side(row: dict[str, str]) -> tuple[str, str]:
    backhand = normalized_flag(row.get("backhand"))
    aroundhead = normalized_flag(row.get("aroundhead"))
    if backhand and aroundhead:
        return "unknown", "conflicting_flags"
    if backhand:
        return "backhand", "backhand=1.0"
    if aroundhead:
        return "aroundhead", "aroundhead=1.0"
    return "forehand", "both_flags_blank"


def int_field(row: dict[str, str], name: str) -> int:
    return int(float(row[name]))


def load_matches(annotation_root: Path) -> dict[int, dict[str, str]]:
    match_path = annotation_root / "match.csv"
    with match_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["id"]): row for row in rows}


def load_annotations(
    annotation_root: Path, matches: dict[int, dict[str, str]]
) -> dict[tuple[int, int, int, int], dict[str, str]]:
    annotations: dict[tuple[int, int, int, int], dict[str, str]] = {}
    for match_id, match in matches.items():
        match_dir = annotation_root / match["video"]
        for set_path in sorted(match_dir.glob("set*.csv")):
            set_id = int(set_path.stem.removeprefix("set"))
            with set_path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    key = (
                        match_id,
                        set_id,
                        int_field(row, "rally"),
                        int_field(row, "ball_round"),
                    )
                    if key in annotations:
                        raise ValueError(f"Duplicate ShuttleSet annotation key: {key}")
                    row["annotation_path"] = set_path.relative_to(annotation_root).as_posix()
                    annotations[key] = row
    return annotations


def parse_class_dir(name: str) -> tuple[str, str]:
    for prefix, side in (("Top_", "top"), ("Bottom_", "bottom")):
        if name.startswith(prefix):
            return side, name[len(prefix) :]
    return "unknown", name


def build_manifest(
    npy_root: Path,
    annotations: dict[tuple[int, int, int, int], dict[str, str]],
    matches: dict[int, dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object], Counter[int]]:
    records: list[dict[str, object]] = []
    issues: Counter[str] = Counter()
    match_samples: Counter[int] = Counter()
    for split_dir in (npy_root / name for name in ("train", "val", "test")):
        if not split_dir.is_dir():
            raise FileNotFoundError(split_dir)
        for joints_path in sorted(split_dir.glob("*/*_joints.npy")):
            match = SAMPLE_PATTERN.fullmatch(joints_path.stem)
            if match is None:
                issues["invalid_sample_filename"] += 1
                continue
            ids = tuple(int(match.group(name)) for name in ("match", "set", "rally", "ball_round"))
            match_id, set_id, rally, ball_round = ids
            expected_split = split_for_match(match_id)
            actual_split = split_dir.name
            if expected_split != actual_split:
                issues["split_mismatch"] += 1

            paths = {
                modality: joints_path.with_name(
                    joints_path.name.replace("_joints.npy", f"_{modality}.npy")
                )
                for modality in MODALITIES
            }
            missing = [name for name, path in paths.items() if not path.is_file()]
            annotation = annotations.get(ids)
            player_side, folder_raw_label = parse_class_dir(joints_path.parent.name)
            if annotation is None:
                issues["missing_annotation"] += 1
                side_label, side_source = "unknown", "missing_annotation"
                annotation_raw_label = ""
                flaw = ""
                annotation_path = ""
            else:
                side_label, side_source = stroke_side(annotation)
                annotation_raw_label = annotation["type"].strip()
                flaw = annotation.get("flaw", "").strip()
                annotation_path = annotation["annotation_path"]
                if folder_raw_label != annotation_raw_label:
                    issues["raw_label_mismatch"] += 1
                if flaw:
                    issues["annotation_flaw"] += 1
                if side_label == "unknown":
                    issues["unknown_stroke_side"] += 1
            if missing:
                issues["missing_modality"] += 1

            raw_label = annotation_raw_label or folder_raw_label
            coarse_label = RAW_TO_COARSE.get(raw_label, "unknown")
            if coarse_label == "unknown":
                issues["unknown_coarse_label"] += 1
            sample_id = "_".join(str(value) for value in ids)
            source_match = matches[match_id]
            valid_stroke_side = (
                not missing
                and annotation is not None
                and not flaw
                and side_label != "unknown"
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "split": actual_split,
                    "match_id": match_id,
                    "set_id": set_id,
                    "rally": rally,
                    "ball_round": ball_round,
                    "hit_frame": int_field(annotation, "frame_num") if annotation else "",
                    "player_side": player_side,
                    "raw_label": raw_label,
                    "coarse_label": coarse_label,
                    "stroke_side": side_label,
                    "stroke_side_source": side_source,
                    "flaw": flaw,
                    "valid_stroke_side": valid_stroke_side,
                    "valid_coarse_label": valid_stroke_side and coarse_label != "unknown",
                    "joints_path": paths["joints"].relative_to(npy_root).as_posix(),
                    "position_path": paths["pos"].relative_to(npy_root).as_posix(),
                    "shuttle_path": paths["shuttle"].relative_to(npy_root).as_posix(),
                    "annotation_path": annotation_path,
                    "video_name": source_match["video"],
                    "video_url": source_match["url"],
                }
            )
            match_samples[match_id] += 1

    records.sort(key=lambda item: (str(item["split"]), int(item["match_id"]), str(item["sample_id"])))
    valid_side_records = [
        record for record in records if bool(record["valid_stroke_side"])
    ]
    valid_coarse_records = [
        record for record in records if bool(record["valid_coarse_label"])
    ]
    audit = {
        "npy_root": str(npy_root.resolve()),
        "total_samples": len(records),
        "valid_stroke_side_samples": len(valid_side_records),
        "valid_coarse_label_samples": len(valid_coarse_records),
        "invalid_stroke_side_samples": len(records) - len(valid_side_records),
        "split_counts": dict(Counter(str(row["split"]) for row in records)),
        "valid_stroke_side_split_counts": dict(
            Counter(str(row["split"]) for row in valid_side_records)
        ),
        "stroke_side_counts": dict(
            Counter(str(row["stroke_side"]) for row in valid_side_records)
        ),
        "coarse_label_counts": dict(
            Counter(str(row["coarse_label"]) for row in valid_coarse_records)
        ),
        "issues": dict(issues),
    }
    return records, audit, match_samples


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def build_download_records(
    matches: dict[int, dict[str, str]],
    match_samples: Counter[int],
    video_dir: Path,
) -> list[dict[str, object]]:
    records = []
    for match_id, match in sorted(matches.items()):
        split = split_for_match(match_id)
        existing = sorted(video_dir.glob(f"{match_id} *")) if video_dir.is_dir() else []
        records.append(
            {
                "match_id": match_id,
                "split": split,
                "npy_samples": match_samples[match_id],
                "required_for_bst_split": split != "excluded",
                "downloaded": bool(existing),
                "local_video_path": str(existing[0]) if existing else "",
                "expected_filename": f"{match_id} {match['video']}.mp4",
                "video_name": match["video"],
                "tournament": match["tournament"],
                "round": match["round"],
                "duration_minutes": match["duration"],
                "winner": match["winner"],
                "loser": match["loser"],
                "url": match["url"],
            }
        )
    return records


def write_checklist(path: Path, records: list[dict[str, object]]) -> None:
    lines = [
        "# ShuttleSet video download checklist",
        "",
        "Save each downloaded video under `data/raw/shuttleset/videos/` using the exact",
        "filename shown below. Matches marked `excluded` are not used by the published",
        "BST split and can be downloaded last.",
        "",
    ]
    for split in ("train", "val", "test", "excluded"):
        lines.extend((f"## {split}", ""))
        for row in records:
            if row["split"] != split:
                continue
            lines.append(
                f"- [{'x' if row['downloaded'] else ' '}] `{row['expected_filename']}` "
                f"([YouTube]({row['url']}), {row['npy_samples']} NPY samples)"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    matches = load_matches(args.annotation_root)
    annotations = load_annotations(args.annotation_root, matches)
    records, audit, match_samples = build_manifest(args.npy_root, annotations, matches)
    write_csv(args.manifest, records)
    downloads = build_download_records(matches, match_samples, args.video_dir)
    write_csv(args.download_list, downloads)
    write_checklist(args.download_checklist, downloads)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
