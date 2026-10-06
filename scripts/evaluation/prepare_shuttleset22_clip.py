"""Build a 5-minute ShuttleSet22 evaluation manifest from the official Git tree."""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
from pathlib import Path


RAW_TO_COARSE = {
    "切球": "drop", "勾球": "net_shot", "平球": "drive", "挑球": "lift",
    "推球": "net_attack", "撲球": "net_attack", "擋小球": "net_shot",
    "放小球": "net_shot", "殺球": "smash", "點扣": "smash",
    "發短球": "serve", "發長球": "serve", "長球": "clear",
    "過渡切球": "drop", "過度切球": "drop", "後場抽平球": "drive",
    "防守回抽": "drive", "防守回挑": "lift", "小平球": "drive",
}
MATCH = "Viktor_AXELSEN_Anthony_Sinisuka_GINTING_DAIHATSU_INDONESIA_MASTERS_2022_Semifinals"


def git_text(repo: Path, object_path: str) -> str:
    return subprocess.check_output(
        ["git", f"--git-dir={repo}", "show", f"HEAD:{object_path}"], text=True,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=17046)
    parser.add_argument("--frames", type=int, default=9000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = f"CoachAI-Challenge-IJCAI2023/ShuttleSet22/set/{MATCH}"
    source = []
    for set_id in (1, 2):
        rows = csv.DictReader(io.StringIO(git_text(args.git_dir, f"{root}/set{set_id}.csv")))
        for row in rows:
            frame = int(row["frame_num"])
            label = RAW_TO_COARSE.get(row["type"])
            if label and args.offset <= frame < args.offset + args.frames:
                source.append((set_id, row, frame, label))
    fields = ["sample_id", "dataset", "split", "match_id", "set_id", "rally",
              "ball_round", "video_path", "start_frame", "hit_frame", "end_frame",
              "fps", "player_side", "raw_label", "coarse_label", "stroke_side",
              "annotation_path"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for set_id, row, frame, label in sorted(source, key=lambda item: item[2]):
            player_y = float(row["player_location_y"] or 0)
            opponent_y = float(row["opponent_location_y"] or 0)
            writer.writerow({
                "sample_id": f"shuttleset22_m16_{set_id}_{row['rally']}_{row['ball_round']}",
                "dataset": "shuttleset22", "split": "external_test", "match_id": "shuttleset22_m16",
                "set_id": set_id, "rally": row["rally"], "ball_round": row["ball_round"],
                "video_path": str(args.video.resolve()), "start_frame": max(0, frame - 30),
                "hit_frame": frame, "end_frame": frame + 30, "fps": 30.0,
                "player_side": "bottom" if player_y > opponent_y else "top",
                "raw_label": row["type"], "coarse_label": label,
                "stroke_side": "backhand" if row["backhand"] else "forehand",
                "annotation_path": f"{MATCH}/set{set_id}.csv",
            })
    print(f"saved={len(source)} output={args.output}")


if __name__ == "__main__":
    main()
