from scripts.discovery.run_s240_research_value_profile_v2_shadow import (
    build_profile,
)


def test_single_channel_profile():
    profile = build_profile(
        {
            "distinct_comparison_count": 1,
            "explicit_contrast_comparison_count": 1,
            "predicted_observation_count": 1,
            "falsification_criterion_count": 1,
            "success_pattern_count": 1,
            "falsification_pattern_count": 1,
            "two_sided_pattern_balance_ratio": 1.0,
            "shared_observable_count": 1,
            "observable_union_count": 1,
            "shared_primary_coverage_ratio": 1.0,
            "prediction_primary_coverage_ratio": 1.0,
            "falsifier_primary_coverage_ratio": 1.0,
            "experimental_disposition": "experimentally_plausible",
            "requires_candidate_concretization": False,
            "relative_cost_burden": "moderate",
            "relative_effort_burden": "moderate",
        }
    )

    assert profile["comparison_structure"] == "SINGLE_EXPLICIT_CONTRAST"
    assert profile["outcome_structure"] == "SINGLE_TWO_SIDED_OUTCOME"
    assert profile["observable_structure"] == "SINGLE_SHARED_OBSERVABLE"
    assert profile["primary_alignment"] == "FULL_PRIMARY_ALIGNMENT"


def test_multi_channel_profile_differs():
    profile = build_profile(
        {
            "distinct_comparison_count": 2,
            "explicit_contrast_comparison_count": 2,
            "predicted_observation_count": 2,
            "falsification_criterion_count": 2,
            "success_pattern_count": 2,
            "falsification_pattern_count": 2,
            "two_sided_pattern_balance_ratio": 1.0,
            "shared_observable_count": 2,
            "observable_union_count": 2,
            "shared_primary_coverage_ratio": 0.5,
            "prediction_primary_coverage_ratio": 0.5,
            "falsifier_primary_coverage_ratio": 0.5,
            "experimental_disposition": "conditionally_plausible",
            "requires_candidate_concretization": True,
            "relative_cost_burden": "high",
            "relative_effort_burden": "high",
        }
    )

    assert profile["comparison_structure"] == "MULTIPLE_EXPLICIT_CONTRASTS"
    assert profile["outcome_structure"] == "MULTI_OUTCOME_BALANCED"
    assert profile["observable_structure"] == "MULTIPLE_SHARED_OBSERVABLES"
    assert profile["primary_alignment"] == "PARTIAL_PRIMARY_ALIGNMENT"
