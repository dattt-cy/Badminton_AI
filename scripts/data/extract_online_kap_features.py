"""Extract raw KAP inputs at hit-detector events for online-domain training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.inference.auto_court_detection import detect_court_corners  # noqa: E402
from scripts.inference.robust_aligned_video_pipeline import AlignedVideoPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path,
                        default=Path("models/checkpoints/pose/yolov8n-pose.pt"))
    args = parser.parse_args()
    events = json.loads(args.scan.read_text(encoding="utf-8"))["events"]
    cap = cv2.VideoCapture(str(args.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pipeline = AlignedVideoPipeline(corners=detect_court_corners(str(args.video)),
                                    orig_w=width, orig_h=height,
                                    pose_model_path=args.pose_model)
    pipeline.prepare_event_cache(
        cap, [int(event["frame"]) for event in events], total
    )
    records = []
    for index, event in enumerate(events, 1):
        feature = pipeline.extract_features(cap, int(event["frame"]), total, hitter_side=event["side"])
        if feature is not None:
            records.append({"hit_frame": int(event["frame"]), "side": event["side"],
                            **{key: feature[key].detach().cpu()
                               for key in ("emb", "s_j", "s_p", "s_s", "s_c", "s_q")}})
        print(f"event={index}/{len(events)} saved={len(records)}", flush=True)
    cap.release(); args.output.parent.mkdir(parents=True, exist_ok=True); torch.save(records, args.output)
    print(f"saved={len(records)} output={args.output}")


if __name__ == "__main__":
    main()
