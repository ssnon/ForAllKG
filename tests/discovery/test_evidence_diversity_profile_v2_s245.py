from scripts.discovery.run_s245_evidence_diversity_profile_v2 import (
    build_profile,
)


def test_profile_exposes_shared_core_and_unique_support():
    evidence = {
        "hypothesis_count": 3,
        "eligible_statement_count": 5,
        "used_statement_count": 4,
        "eligible_statement_coverage": 0.8,
        "unused_eligible_statement_ids": ["s5"],
        "shared_core_statement_count": 2,
        "distinct_premise_set_count": 2,
        "exact_premise_set_duplicate_group_count": 1,
        "exact_premise_set_groups": [
            {
                "hypothesis_ids": ["h1", "h2"],
            }
        ],
        "mean_pairwise_statement_jaccard": 0.7,
        "max_pairwise_statement_jaccard": 1.0,
        "multi_paper_used_statement_count": 2,
        "mean_papers_per_used_statement": 1.5,
        "cards": [
            {"portfolio_unique_premise_count": 0},
            {"portfolio_unique_premise_count": 0},
            {"portfolio_unique_premise_count": 1},
        ],
    }

    row = build_profile(evidence)

    assert row["shared_core_fraction_of_used"] == 0.5
    assert row["zero_unique_support_hypothesis_count"] == 2
    assert row["zero_unique_support_hypothesis_fraction"] == 0.666667
    assert row["exact_duplicate_hypothesis_count"] == 2
    assert row["exact_duplicate_hypothesis_fraction"] == 0.666667
    assert row["distinct_premise_set_fraction"] == 0.666667
    assert row["unused_eligible_statement_count"] == 1
