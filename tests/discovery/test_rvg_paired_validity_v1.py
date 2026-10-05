from scripts.discovery.summarize_rvg_paired_validity import (
    build_paired_summary,
)


def audit(alias, comparison, controls):
    return {
        "candidate_alias": alias,
        "summary": alias,
        "dimensions": [
            {
                "dimension": "COMPARISON_CONTEXT_COMPATIBILITY",
                "verdict": comparison,
                "rationale": "r",
            },
            {
                "dimension": "CONTROL_VARIABLE_ADEQUACY",
                "verdict": controls,
                "rationale": "r",
            },
        ],
    }


def test_paired_summary_tracks_improvement_and_abstention():
    control = {
        "candidates": [
            audit("V01", "FAIL", "FAIL"),
            audit("V02", "FAIL", "WARN"),
        ]
    }
    treatment = {
        "candidates": [
            audit("V01", "WARN", "PASS"),
        ]
    }
    mapping = {
        "control_case_id": "c",
        "treatment_case_id": "t",
        "control_candidate_count": 2,
        "records": [
            {
                "control_candidate_alias": "V01",
                "control_hypothesis_id": "h1",
                "treatment_status": "GENERATED_TREATMENT",
                "identification_assessment": "PARTIALLY_IDENTIFIED",
                "treatment_hypothesis_id": "t1",
                "treatment_audit_alias": "V01",
                "exact_premise_identity": True,
                "exact_gap_identity": True,
                "exact_hypothesis_type_identity": True,
            },
            {
                "control_candidate_alias": "V02",
                "control_hypothesis_id": "h2",
                "treatment_status": "ABSTAINED_NOT_IDENTIFIABLE",
                "identification_assessment": "NOT_IDENTIFIABLE",
                "treatment_hypothesis_id": None,
                "treatment_audit_alias": None,
                "exact_premise_identity": None,
                "exact_gap_identity": None,
                "exact_hypothesis_type_identity": None,
            },
        ],
    }

    body = build_paired_summary(
        control_audit=control,
        treatment_audit=treatment,
        mapping=mapping,
    )

    assert body["generated_pair_count"] == 1
    assert body["safe_abstention_count"] == 1
    assert body["dimension_transition_counts"]["IMPROVED"] == 2
    assert body["control_fail_dimension_counts"][
        "COMPARISON_CONTEXT_COMPATIBILITY"
    ] == 2
    assert body["treatment_fail_dimension_counts"] == {}


def test_paired_summary_distinguishes_resolved_fail_from_unknown_downgrade():
    control = {
        "candidates": [
            {
                "candidate_alias": "V01",
                "summary": "control",
                "dimensions": [
                    {
                        "dimension": "COMPARISON_CONTEXT_COMPATIBILITY",
                        "verdict": "FAIL",
                        "rationale": "r",
                    },
                    {
                        "dimension": "VARIANCE_SEMANTICS",
                        "verdict": "PASS",
                        "rationale": "r",
                    },
                ],
            }
        ]
    }
    treatment = {
        "candidates": [
            {
                "candidate_alias": "V01",
                "summary": "treatment",
                "dimensions": [
                    {
                        "dimension": "COMPARISON_CONTEXT_COMPATIBILITY",
                        "verdict": "WARN",
                        "rationale": "r",
                    },
                    {
                        "dimension": "VARIANCE_SEMANTICS",
                        "verdict": "UNKNOWN",
                        "rationale": "r",
                    },
                ],
            }
        ]
    }
    mapping = {
        "control_case_id": "c",
        "treatment_case_id": "t",
        "control_candidate_count": 1,
        "records": [
            {
                "control_candidate_alias": "V01",
                "control_hypothesis_id": "h1",
                "treatment_status": "GENERATED_TREATMENT",
                "identification_assessment": "PARTIALLY_IDENTIFIED",
                "treatment_hypothesis_id": "t1",
                "treatment_audit_alias": "V01",
                "exact_premise_identity": True,
                "exact_gap_identity": True,
                "exact_hypothesis_type_identity": True,
            }
        ],
    }

    body = build_paired_summary(
        control_audit=control,
        treatment_audit=treatment,
        mapping=mapping,
    )

    row = body["paired_candidates"][0]
    assert (
        row["paired_outcome"]
        == "GENERATED_WITH_HARD_VALIDITY_IMPROVEMENT_AND_UNCERTAINTY_DOWNGRADE"
    )
    assert row["introduced_failures"] == []
    assert row["resolved_failures"] == [
        "COMPARISON_CONTEXT_COMPATIBILITY"
    ]
    assert row["uncertainty_downgrades"] == [
        "VARIANCE_SEMANTICS"
    ]
