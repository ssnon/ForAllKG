from scripts.discovery.summarize_adaptive_yield_case import classify_candidate


def test_identification_failure_dominates_novelty():
    assert classify_candidate(
        "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
        ["CONTROL_VARIABLE_ADEQUACY"],
        [],
    ) == "SCIENTIFIC_IDENTIFICATION_RISK"


def test_prior_art_extension_is_not_promoted():
    assert classify_candidate(
        "LITERATURE_SUPPORTED_EXTENSION",
        [],
        [],
    ) == "PRIOR_ART_SATURATED_OR_EXTENSION"
