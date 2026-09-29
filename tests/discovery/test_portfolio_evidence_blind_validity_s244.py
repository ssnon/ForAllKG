from scripts.discovery.run_s244_portfolio_evidence_blind_validity import (
    build_blind_payload,
    expected_disposition,
)


def test_recommendation_mapping():
    assert (
        expected_disposition("REDUNDANCY_REVIEW")
        == "REDUNDANCY_REVIEW_WARRANTED"
    )
    assert (
        expected_disposition("NO_ACTION")
        == "NO_MATERIAL_REDUNDANCY_CONCERN"
    )


def test_blind_payload_reuses_anonymous_evidence_labels():
    evidence = {
        "cards": [
            {
                "title": "H1",
                "premise_statement_ids": ["s1", "s2"],
            },
            {
                "title": "H2",
                "premise_statement_ids": ["s1"],
            },
        ]
    }
    context = {
        "evidence_statements": [
            {
                "statement_id": "s1",
                "text": "Evidence one.",
                "epistemic_role": "reported",
                "claim_kind": "observation",
                "paper_ids": ["p1"],
            },
            {
                "statement_id": "s2",
                "text": "Evidence two.",
                "epistemic_role": "reported",
                "claim_kind": "mechanism",
                "paper_ids": ["p2"],
            },
        ]
    }

    payload, labels = build_blind_payload(
        evidence=evidence,
        context=context,
    )

    assert labels == {"s1": "E1", "s2": "E2"}
    assert payload["hypotheses"][0]["positive_premises"] == ["E1", "E2"]
    assert payload["hypotheses"][1]["positive_premises"] == ["E1"]
    assert "statement_id" not in payload["evidence_catalog"]["E1"]
