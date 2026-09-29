from scripts.discovery.run_s239_research_value_raw_feature_shadow import (
    feature_vector,
)


def _card():
    return {
        "hypothesis_type": "mechanistic_extension",
        "predicted_observations": [
            {"observable": "mode ratio"},
            {"observable": "intensity"},
        ],
        "falsification_criteria": [
            {"observable": "mode ratio"},
        ],
    }


def _spec():
    return {
        "validation_strategy": "mechanism_validation",
        "requires_candidate_concretization": False,
        "primary_observables": ["mode ratio"],
        "required_comparisons": [
            "compare proposed mechanism against alternative",
            "matched control",
        ],
        "success_patterns": ["shift", "increase"],
        "falsification_patterns": ["no shift"],
    }


def _experimental():
    return {
        "disposition": "conditionally_plausible",
        "relative_cost_burden": "moderate",
        "relative_effort_burden": "high",
    }


def test_feature_vector_exposes_nonbinary_structure():
    row = feature_vector(
        card=_card(),
        spec=_spec(),
        experimental=_experimental(),
    )
    assert row["prediction_observable_count"] == 2
    assert row["falsifier_observable_count"] == 1
    assert row["shared_observable_count"] == 1
    assert row["observable_union_count"] == 2
    assert row["observable_overlap_ratio"] == 0.5
    assert row["shared_primary_coverage_ratio"] == 1.0
    assert row["prediction_primary_coverage_ratio"] == 0.5
    assert row["required_comparison_count"] == 2
    assert row["explicit_contrast_comparison_count"] == 2
    assert row["success_pattern_count"] == 2
    assert row["falsification_pattern_count"] == 1
    assert row["two_sided_pattern_balance_ratio"] == 0.5


def test_resource_fields_remain_descriptive():
    row = feature_vector(
        card=_card(),
        spec=_spec(),
        experimental=_experimental(),
    )
    assert row["relative_cost_burden"] == "moderate"
    assert row["relative_effort_burden"] == "high"
