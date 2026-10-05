from scripts.discovery.summarize_rvg_paired_novelty import (
    build_summary,
)


def synth(alias, status):
    return {
        "candidates": [
            {
                "validity": {"candidate_alias": alias},
                "fresh_search_novelty": {"status": status},
            }
        ]
    }


def test_paired_novelty_is_categorical_and_maps_generated_treatment():
    body = build_summary(
        control_synthesis=synth(
            "V01", "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
        ),
        treatment_synthesis=synth(
            "V01", "NEW_COMBINATION_OF_KNOWN_EFFECTS"
        ),
        mapping={
            "control_case_id": "c",
            "treatment_case_id": "t",
            "records": [
                {
                    "control_candidate_alias": "V01",
                    "control_hypothesis_id": "h1",
                    "treatment_status": "GENERATED_TREATMENT",
                    "identification_assessment": "PARTIALLY_IDENTIFIED",
                    "treatment_hypothesis_id": "t1",
                    "treatment_audit_alias": "V01",
                    "exact_premise_identity": True,
                }
            ],
        },
    )

    assert body["novelty_status_is_ordinal"] is False
    assert body["transition_counts"] == {
        "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP -> "
        "NEW_COMBINATION_OF_KNOWN_EFFECTS": 1
    }
