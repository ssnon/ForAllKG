
from scripts.discovery.run_s230_conceptual_resolution_closure import (
    conceptual_disposition,
)


def test_gap_at_l2_is_explicit():
    assert conceptual_disposition(
        [
            ("L1_BROAD", "RELATION_BACKED"),
            ("L2_INTERMEDIATE", "SEARCH_BOUNDED_GAP"),
            ("L3_EXACT", "SEARCH_BOUNDED_GAP"),
        ]
    ) == "GAP_AT_L2"


def test_no_gap_observed_is_distinct():
    assert conceptual_disposition(
        [
            ("L1_BROAD", "RELATION_BACKED"),
            ("L2_INTERMEDIATE", "PARTIAL_RELATION_BACKED"),
            ("L3_EXACT", "PARTIAL_RELATION_BACKED"),
        ]
    ) == "NO_GAP_OBSERVED"


def test_insufficient_coverage_stays_indeterminate():
    assert conceptual_disposition(
        [
            ("L1_BROAD", "RELATION_BACKED"),
            ("L2_INTERMEDIATE", "INSUFFICIENT_COVERAGE"),
            ("L3_EXACT", "SEARCH_BOUNDED_GAP"),
        ]
    ) == "INDETERMINATE_COVERAGE"


def test_conflict_is_separate():
    assert conceptual_disposition(
        [
            ("L1_BROAD", "RELATION_BACKED"),
            ("L2_INTERMEDIATE", "CONFLICTING_RELATION"),
            ("L3_EXACT", "PARTIAL_RELATION_BACKED"),
        ]
    ) == "CONFLICTING_EVIDENCE"
