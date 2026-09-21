from scripts.evaluation.mine_hit_selector_hard_examples import merged_ranges


def test_merged_ranges_combines_hits_in_the_same_rally_window():
    assert merged_ranges([100, 140, 400], 30) == [(70, 171), (370, 431)]
