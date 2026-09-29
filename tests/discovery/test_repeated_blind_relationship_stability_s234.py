from scripts.discovery.run_s234_repeated_blind_relationship_stability import (
    stability_class,
)


def test_unanimous_matching_s233():
    assert stability_class(
        original_relationship="PARTIAL_PRIOR_ART",
        replicate_relationships=[
            "PARTIAL_PRIOR_ART",
            "PARTIAL_PRIOR_ART",
            "PARTIAL_PRIOR_ART",
        ],
        all_spans_valid=True,
    ) == "UNANIMOUS_AND_MATCHES_S233"


def test_unanimous_but_differs_from_s233():
    assert stability_class(
        original_relationship="PARTIAL_PRIOR_ART",
        replicate_relationships=[
            "COMPONENT_ONLY",
            "COMPONENT_ONLY",
            "COMPONENT_ONLY",
        ],
        all_spans_valid=True,
    ) == "UNANIMOUS_BUT_DIFFERS_FROM_S233"


def test_majority_matching_s233():
    assert stability_class(
        original_relationship="COMPONENT_ONLY",
        replicate_relationships=[
            "COMPONENT_ONLY",
            "COMPONENT_ONLY",
            "PARTIAL_PRIOR_ART",
        ],
        all_spans_valid=True,
    ) == "MAJORITY_MATCHES_S233"


def test_invalid_span_dominates():
    assert stability_class(
        original_relationship="COMPONENT_ONLY",
        replicate_relationships=[
            "COMPONENT_ONLY",
            "COMPONENT_ONLY",
            "COMPONENT_ONLY",
        ],
        all_spans_valid=False,
    ) == "INVALID_EVIDENCE_SPAN_PRESENT"
