from scripts.discovery.run_s231_external_novelty_status_attribution_audit import (
    attribution_class,
    transition_direction,
)


def sig(status, relation, gap, unresolved, abstracts):
    return {
        "card_status": status,
        "relation_backed_core_count": relation,
        "gap_like_core_count": gap,
        "unresolved_core_count": unresolved,
        "abstract_work_count": abstracts,
    }


def test_more_relation_backed_but_more_novel_is_flagged():
    before = sig(
        "LITERATURE_SUPPORTED_EXTENSION",
        1, 0, 1, 5,
    )
    after = sig(
        "NEW_COMBINATION_OF_KNOWN_EFFECTS",
        2, 0, 0, 7,
    )
    assert (
        attribution_class(before, after)
        == "UPWARD_NOVELTY_DESPITE_MORE_RELATION_BACKED_CORE"
    )


def test_resolution_upward_is_audit_flag_not_hard_failure():
    before = sig(
        "NEW_COMBINATION_OF_KNOWN_EFFECTS",
        1, 1, 1, 5,
    )
    after = sig(
        "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
        1, 2, 0, 7,
    )
    assert (
        attribution_class(before, after)
        == "UPWARD_NOVELTY_AFTER_RESOLUTION"
    )


def test_less_novel_with_more_prior_art_is_expected_direction():
    before = sig(
        "NEW_COMBINATION_OF_KNOWN_EFFECTS",
        1, 1, 0, 5,
    )
    after = sig(
        "LITERATURE_SUPPORTED_EXTENSION",
        2, 0, 0, 7,
    )
    assert (
        attribution_class(before, after)
        == "DOWNWARD_NOVELTY_WITH_STRONGER_PRIOR_ART"
    )


def test_insufficient_is_incomparable():
    assert (
        transition_direction(
            "INSUFFICIENT_SEARCH_EVIDENCE",
            "NEW_COMBINATION_OF_KNOWN_EFFECTS",
        )
        == "INCOMPARABLE"
    )
