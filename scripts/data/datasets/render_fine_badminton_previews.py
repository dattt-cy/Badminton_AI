"""Render motion contact sheets for a balanced Fine-Badminton label audit."""

from __future__ import annotations

import argparse
import csv
import html
import random
from collections import defaultdict
from pathlib import Path

import cv2

from ai_classifier.datasets import decode_virtual_clip, load_fine_badminton_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fine_badminton_8class.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fine_badminton_preview_audit"),
    )
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--frames-per-sheet", type=int, default=6)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_class <= 0 or args.frames_per_sheet <= 0:
        raise ValueError("Preview counts must be positive")
    records = load_fine_badminton_manifest(args.manifest)
    grouped = defaultdict(list)
    for record in records:
        grouped[(record.split, record.coarse_label)].append(record)

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    cards: list[str] = []
    for split in ("train", "val", "test"):
        class_names = sorted(name for candidate_split, name in grouped if candidate_split == split)
        for class_name in class_names:
            candidates = sorted(grouped[(split, class_name)], key=lambda item: item.sample_id)
            chosen = rng.sample(candidates, min(args.samples_per_class, len(candidates)))
            destination = args.output_dir / split / class_name
            destination.mkdir(parents=True, exist_ok=True)
            for record in chosen:
                frames = decode_virtual_clip(
                    record, args.dataset_root, frame_count=args.frames_per_sheet
                )
                tiles = []
                for frame_number, rgb in enumerate(frames):
                    scale = args.tile_width / rgb.shape[1]
                    tile = cv2.resize(
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        (args.tile_width, max(1, round(rgb.shape[0] * scale))),
                    )
                    cv2.putText(
                        tile,
                        f"{frame_number + 1}/{len(frames)}",
                        (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                    )
                    tiles.append(tile)
                sheet = cv2.hconcat(tiles)
                filename = f"{record.sample_id}.jpg"
                output_path = destination / filename
                if not cv2.imwrite(str(output_path), sheet):
                    raise RuntimeError(f"Could not write {output_path}")
                relative = output_path.relative_to(args.output_dir).as_posix()
                row = {
                    "sample_id": record.sample_id,
                    "split": split,
                    "coarse_label": class_name,
                    "canonical_label": record.canonical_label,
                    "raw_label": record.raw_label,
                    "player_side": record.player_side,
                    "start_frame": record.start_frame,
                    "end_frame": record.end_frame,
                    "preview": relative,
                    "review_status": "pending",
                    "review_note": "",
                }
                rows.append(row)
                cards.append(
                    "<article><img src='{}' loading='lazy'><p><b>{}</b> — {} / {} "
                    "— {} — frames {}–{}</p></article>".format(
                        html.escape(relative),
                        html.escape(record.sample_id),
                        html.escape(split),
                        html.escape(class_name),
                        html.escape(record.canonical_label),
                        record.start_frame,
                        record.end_frame,
                    )
                )

    review_path = args.output_dir / "review.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Fine-Badminton audit</title>"
        "<style>body{font-family:sans-serif;background:#111;color:#eee}"
        "article{margin:20px 0;padding:12px;background:#222}img{max-width:100%}"
        "p{overflow-wrap:anywhere}</style><h1>Fine-Badminton preview audit</h1>"
        + "".join(cards),
        encoding="utf-8",
    )
    print(f"Rendered {len(rows)} previews to {args.output_dir}")
    print(f"Review sheet: {review_path}")


if __name__ == "__main__":
    main()
