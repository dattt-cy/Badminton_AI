"""Extract labeled KAP records from ShuttleSet22 broadcast videos (BWF TV uploads).

ShuttleSet22 (https://github.com/wywyWang/CoachAI-Projects, IJCAI-23) labels every stroke
of 2021-2022 singles matches with the same 18 shot types as ShuttleSet. Records are taken
at the labeled hit frames (not at hit-detector events, whose recall on these venues is
low); contact refinement still runs as in the pipeline. The hitter's court half comes
from the labeled player location projected against the annotated court.

    python scripts/data/extract_shuttleset22_records.py \
        --root C:/Users/ADMIN/Downloads/shuttleset22 --output-dir work_dirs/shuttleset22_records
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

from ai_classifier.models import SIDE_CLASSES, STROKE_CLASSES  # noqa: E402
from scripts.inference.auto_court_detection import detect_court_line_corners  # noqa: E402
from scripts.inference.robust_aligned_video_pipeline import AlignedVideoPipeline  # noqa: E402

CKPTS = REPO_ROOT / "src/ai_classifier/models/checkpoints"
KEYS = ("emb", "s_j", "s_p", "s_s", "s_c", "s_q")
# ShuttleSet raw label -> coarse class (same mapping as data/manifests/shuttleset_npy.csv);
# 過度切球 is a ShuttleSet22 spelling of 過渡切球 (passive drop).
COARSE = {
    "發短球": "serve", "發長球": "serve", "長球": "clear", "殺球": "smash", "點扣": "smash",
    "切球": "drop", "過渡切球": "drop", "過度切球": "drop", "放小球": "net_shot", "擋小球": "net_shot",
    "勾球": "net_shot", "挑球": "lift", "防守回挑": "lift", "推球": "net_attack", "撲球": "net_attack",
    "平球": "drive", "防守回抽": "drive", "後場抽平球": "drive", "小平球": "drive",
}
# Labeled frame_num is 4 frames earlier than the BWF TV upload (measured on a 6-minute
# segment by matching hit-detector events).
FRAME_OFFSET = 4


class ForwardCapture:
    """cv2.VideoCapture stand-in for frame requests in increasing order.

    The pipeline seeks before every stroke; on long YouTube uploads with sparse keyframes
    each seek re-decodes hundreds of frames, which made extraction decode-bound. This
    decodes forward once and serves recent frames from memory, seeking only to go back
    further than the buffer.
    """

    def __init__(self, path: Path, buffer_frames: int = 300) -> None:
        self._cap = cv2.VideoCapture(str(path))
        self._buffer: dict[int, np.ndarray] = {}
        self._buffer_frames = buffer_frames
        self._decoded = -1  # index of the last frame read from the stream
        self._position = 0

    def get(self, prop: int) -> float:
        return self._cap.get(prop)

    def set(self, prop: int, value: float) -> bool:
        if prop != cv2.CAP_PROP_POS_FRAMES:
            return self._cap.set(prop, value)
        self._position = int(value)
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame_index = self._position
        if frame_index not in self._buffer:
            if frame_index <= self._decoded or frame_index > self._decoded + self._buffer_frames:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                self._decoded = frame_index - 1
                self._buffer.clear()
            while self._decoded < frame_index:
                ok, frame = self._cap.read()
                if not ok:
                    return False, None
                self._decoded += 1
                self._buffer[self._decoded] = frame
                self._buffer.pop(self._decoded - self._buffer_frames, None)
        self._position += 1
        return True, self._buffer[frame_index]

    def release(self) -> None:
        self._cap.release()


def court_corners(root: Path, video: str) -> np.ndarray:
    for row in csv.DictReader((root / "set/homography.csv").open(encoding="utf-8-sig")):
        if row["video"] == video:
            f = lambda key: float(row[key])  # noqa: E731
            return np.float32([[f("downleft_x"), f("downleft_y")], [f("downright_x"), f("downright_y")],
                               [f("upright_x"), f("upright_y")], [f("upleft_x"), f("upleft_y")]])
    raise KeyError(video)


def labeled_strokes(root: Path, video: str, annotated: np.ndarray) -> list[dict]:
    """Labeled strokes with coarse class, stroke side and the hitter's court half."""
    to_image = cv2.getPerspectiveTransform(np.float32([[0, 1], [1, 1], [1, 0], [0, 0]]), annotated)
    net_y = float(cv2.perspectiveTransform(np.float32([[[0.5, 0.5]]]), to_image)[0, 0, 1])
    strokes = []
    for path in sorted((root / "set" / video).glob("set*.csv")):
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            coarse = COARSE.get(row["type"])
            if coarse is None or not row["frame_num"] or not row["player_location_y"]:
                continue
            side = ("aroundhead" if row["aroundhead"] == "1.0"
                    else "backhand" if row["backhand"] == "1.0" else "forehand")
            strokes.append({
                "hit_frame": int(float(row["frame_num"])) + FRAME_OFFSET,
                "side": "upper" if float(row["player_location_y"]) < net_y else "lower",
                "gt_stroke_idx": STROKE_CLASSES.index(coarse),
                "gt_side_idx": SIDE_CLASSES.index(side),
                "set": path.stem, "rally": int(row["rally"]), "ball_round": int(row["ball_round"]),
            })
    return sorted(strokes, key=lambda stroke: stroke["hit_frame"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="ShuttleSet22 folder with set/ and videos/.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "work_dirs/shuttleset22_records")
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "work_dirs/kap_cross_v2_liftboost_seed20260926/kap_best.pth")
    parser.add_argument("--contact-refine-radius", type=int, default=6)
    parser.add_argument("--shard", default="0/1", help="i/n: process every n-th match starting at i (parallel runs).")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard, shards = (int(v) for v in args.shard.split("/"))
    downloads = json.loads((args.root / "download_list.json").read_text(encoding="utf-8"))[shard::shards]
    for item in downloads:
        video_path = args.root / "videos" / f"{item['id']}.mp4"
        out = args.output_dir / f"{item['video']}.pt"
        if out.exists() or not video_path.exists():
            print(f"[skip] {item['video']} ({'done' if out.exists() else 'no video'})", flush=True)
            continue
        strokes = labeled_strokes(args.root, item["video"], court_corners(args.root, item["video"]))
        cap = ForwardCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        strokes = [s for s in strokes if 45 <= s["hit_frame"] < total - 60]
        corners = detect_court_line_corners(str(video_path))
        pipeline = AlignedVideoPipeline(
            ckpt_path=args.checkpoint, corners=corners,
            orig_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), orig_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            pose_model_path=CKPTS / "yolov8n-pose.pt",
        )
        pipeline.set_event_frames([s["hit_frame"] for s in strokes])
        records = []
        # Pose is cached one rally at a time so its frames are still buffered when the
        # strokes of that rally are extracted (each frame is decoded once).
        rallies: dict[tuple[str, int], list[dict]] = {}
        for stroke in strokes:
            rallies.setdefault((stroke["set"], stroke["rally"]), []).append(stroke)
        for rally in sorted(rallies.values(), key=lambda group: group[0]["hit_frame"]):
            pipeline.prepare_event_cache(cap, [s["hit_frame"] for s in rally], total,
                                         contact_refine_radius=args.contact_refine_radius)
            for stroke in rally:
                feat = pipeline.extract_features(cap, stroke["hit_frame"], total, hitter_side=stroke["side"],
                                                 contact_refine_radius=args.contact_refine_radius)
                if feat is not None:
                    records.append({**stroke, "video": item["video"],
                                     **{key: feat[key].detach().cpu() for key in KEYS}})
        cap.release()
        torch.save({"court_corners": corners.tolist(), "records": records}, out)
        print(f"[+] {item['video']}: {len(records)}/{len(strokes)} records", flush=True)


if __name__ == "__main__":
    main()
