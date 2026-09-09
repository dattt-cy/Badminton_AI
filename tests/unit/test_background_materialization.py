import csv

from scripts.data.materialize_3class_background import main


FIELDS = (
    "pose", "video", "valid_ratio", "median_torso_scale_px", "scale_cv",
    "wrist_speed_spike_ratio", "wrist_speed_range", "wrist_excursion",
    "arm_angle_range", "phases_valid",
)


def test_background_gate_keeps_only_stationary_clips(tmp_path, monkeypatch):
    source = tmp_path / "source" / "Sub08"
    source.mkdir(parents=True)
    rows = []
    for name, speed in (("still", "0.5"), ("moving", "4.0")):
        pose = source / f"{name}.npz"
        video = source / f"{name}.mp4"
        pose.touch()
        video.touch()
        rows.append({
            "pose": str(pose), "video": str(video), "valid_ratio": "0.95",
            "median_torso_scale_px": "60", "scale_cv": "0.1",
            "wrist_speed_spike_ratio": "1.2", "wrist_speed_range": speed,
            "wrist_excursion": "0.1", "arm_angle_range": "10",
            "phases_valid": "False",
        })
    report = tmp_path / "quality.csv"
    with report.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "output"
    monkeypatch.setattr("sys.argv", ["materialize", str(report), str(output)])
    main()

    saved = list((output / "background").rglob("*.npz"))
    assert [path.name for path in saved] == ["still_background.npz"]
