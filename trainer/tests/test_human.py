import numpy as np

from eval.human import human_inter_rater_baseline


def test_human_baseline_matches_physical_labels_by_event_id():
    rows = [
        {"event_id": "b", "rater_a_level": "4", "rater_b_level": "5"},
        {"event_id": "a", "rater_a_level": "3", "rater_b_level": "3"},
    ]
    result = human_inter_rater_baseline(rows, ["a", "b"], np.array([0, 2]))
    assert result["matched_to_physical_test"] == 2
    assert result["rater_vs_physical_test"]["rater_a"]["dangerous_underclassification_fnr"] == 0.5
