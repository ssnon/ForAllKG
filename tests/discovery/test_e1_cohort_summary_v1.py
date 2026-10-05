from scripts.discovery.summarize_e1_cohort import summarize_payloads


def case(case_id, fail_count, risk_count):
    return {
        "case_id": case_id,
        "candidates": [
            {"validity": {"warn_dimensions": ["CAUSAL_MEDIATION"]}}
            for _ in range(5)
        ],
        "fresh_search_novelty_status_counts": {
            "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP": 3,
            "NEW_COMBINATION_OF_KNOWN_EFFECTS": 2,
        },
        "validity_fail_dimension_counts": {
            "COMPARISON_CONTEXT_COMPATIBILITY": fail_count,
        },
        "synthesis_class_counts": {
            "SCIENTIFIC_IDENTIFICATION_RISK": risk_count,
            "RELATIONAL_GAP_OR_NEW_COMBINATION": 5 - risk_count,
        },
        "stages": [
            {
                "condition": "GRAPH_ADAPTIVE_777",
                "cumulative_effective_count": 1,
                "marginal_effective_count": 1,
                "unresolved_count": 0,
                "graph_retraversal_count": 2,
                "new_context_count": 2,
                "structurally_new_premise_count": 5,
                "search_attempt_unique_count": 4,
            }
        ],
    }


def test_cohort_aggregates_cases_and_risk():
    body = summarize_payloads(
        [
            ("q1.json", case("q1", 4, 4)),
            ("q2.json", case("q2", 2, 2)),
            ("q3.json", case("q3", 3, 3)),
        ]
    )
    assert body["case_count"] == 3
    assert body["candidate_count"] == 15
    assert body["scientific_identification_risk_count"] == 9
    assert body["aggregate_validity_fail_dimension_counts"][
        "COMPARISON_CONTEXT_COMPATIBILITY"
    ] == 9
    assert body["aggregate_stage_totals"]["GRAPH_ADAPTIVE_777"][
        "graph_retraversal_count"
    ] == 6
