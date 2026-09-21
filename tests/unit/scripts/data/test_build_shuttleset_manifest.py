from scripts.data.datasets.build_shuttleset_manifest import (
    build_download_records,
    parse_class_dir,
    split_for_match,
    stroke_side,
)


def test_split_for_match_matches_published_bst_partition():
    assert split_for_match(1) == "train"
    assert split_for_match(35) == "val"
    assert split_for_match(39) == "test"
    assert split_for_match(9) == "excluded"


def test_stroke_side_parses_blank_and_one_point_zero_flags():
    assert stroke_side({"backhand": "", "aroundhead": ""})[0] == "forehand"
    assert stroke_side({"backhand": "1.0", "aroundhead": ""})[0] == "backhand"
    assert stroke_side({"backhand": "", "aroundhead": "1.0"})[0] == "aroundhead"
    assert stroke_side({"backhand": "1.0", "aroundhead": "1.0"})[0] == "unknown"


def test_parse_class_dir_keeps_player_side_separate_from_shot_type():
    assert parse_class_dir("Top_挑球") == ("top", "挑球")
    assert parse_class_dir("Bottom_殺球") == ("bottom", "殺球")
    assert parse_class_dir("未知球種") == ("unknown", "未知球種")


def test_download_records_mark_a_matching_local_video(tmp_path):
    (tmp_path / "1 example.webm").touch()
    records = build_download_records(
        {
            1: {
                "video": "example",
                "tournament": "event",
                "round": "Finals",
                "duration": "10",
                "winner": "A",
                "loser": "B",
                "url": "https://example.test/video",
            }
        },
        {1: 5},
        tmp_path,
    )
    assert records[0]["downloaded"] is True
    assert records[0]["local_video_path"].endswith("1 example.webm")
