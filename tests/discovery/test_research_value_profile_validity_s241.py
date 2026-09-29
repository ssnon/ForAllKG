from scripts.discovery.run_s241_research_value_profile_validity_audit import (
    factor_purity,
    lexical_similarity,
)


def test_factor_purity_detects_template_mapping():
    rows = [
        {"hypothesis_type": "a", "profile_key": "p1"},
        {"hypothesis_type": "a", "profile_key": "p1"},
        {"hypothesis_type": "b", "profile_key": "p2"},
    ]
    purity, _ = factor_purity(rows, "hypothesis_type")
    assert purity == 1.0


def test_factor_purity_detects_non_template_variation():
    rows = [
        {"hypothesis_type": "a", "profile_key": "p1"},
        {"hypothesis_type": "a", "profile_key": "p2"},
        {"hypothesis_type": "b", "profile_key": "p2"},
    ]
    purity, _ = factor_purity(rows, "hypothesis_type")
    assert purity < 1.0


def test_lexical_similarity_handles_contained_observable_names():
    assert lexical_similarity(
        "hydrogen adsorption energy",
        "computed hydrogen adsorption free energy",
    ) >= 0.5
