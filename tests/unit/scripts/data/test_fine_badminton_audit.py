from collections import Counter
from pathlib import Path

from scripts.data.datasets.audit_fine_badminton import (
    choose_group_split,
    load_taxonomy,
)


TAXONOMY = Path("configs/taxonomy/fine_badminton_8class.yaml")


def test_fine_badminton_taxonomy_covers_all_raw_labels():
    aliases, classes = load_taxonomy(TAXONOMY)

    assert len(aliases) == 60
    assert len({canonical for canonical, _ in aliases.values()}) == 29
    assert classes == [
        "serve",
        "clear",
        "smash",
        "drop",
        "net_shot",
        "lift",
        "drive",
        "net_attack",
    ]


def test_group_split_keeps_whole_matches_and_all_classes():
    classes = ["serve", "clear", "smash", "drop", "net_shot"]
    matches = [f"match_{index:02d}" for index in range(10)]
    counts = {
        match: Counter({name: 20 + index for name in classes})
        for index, match in enumerate(matches)
    }

    split = choose_group_split(matches, counts, classes)

    assert set(split) == set(matches)
    assert Counter(split.values()) == {"train": 6, "val": 2, "test": 2}
    for split_name in ("train", "val", "test"):
        members = [match for match, assigned in split.items() if assigned == split_name]
        aggregate = Counter()
        for member in members:
            aggregate.update(counts[member])
        assert all(aggregate[name] > 0 for name in classes)
