import pytest

from domains.sers.context_contracts import (
    SERSContextFact,
    SERSContextFinding,
    SERSContextProvenance,
    SERSContextReview,
    SERSContextSignature,
    expected_context_finding_severity,
    expected_context_review_status,
)
from domains.sers.hypothesis_action_adapter import (
    SERSActionBindingError,
    SERSContextFindingActionAdapter,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    PredictedObservation,
)


SOURCE_H = "hypothesis:source"
TARGET_H = "hypothesis:target"


CENTRAL_TEXT = (
    "Copper-conditioned context moderates "
    "the Au-Ag nanogap response."
)

BRIDGE_TEXT = (
    "A source-context relation is proposed "
    "as a bounded moderator."
)

PREDICTION_TEXT = (
    "Observable: SERS response\n"
    "Expected direction: shift\n"
    "Rationale: copper-conditioned context "
    "may alter the gap response"
)

ASSUMPTION_TEXT = (
    "Measurement conditions remain comparable."
)


SOURCE_ASSERTIONS = {
    f"central:{SOURCE_H}":
        CENTRAL_TEXT,

    f"bridge:{SOURCE_H}":
        BRIDGE_TEXT,

    "prediction:source-p1":
        PREDICTION_TEXT,

    f"assumption:{SOURCE_H}:0":
        ASSUMPTION_TEXT,
}


def _target_card(
    *,
    bridge_text=BRIDGE_TEXT,
):
    return HypothesisCard.model_construct(
        hypothesis_id=TARGET_H,
        domain_profile_id="sers_au_ag",
        hypothesis_statement=
            CENTRAL_TEXT,
        inferential_bridge=
            bridge_text,
        predicted_observations=[
            PredictedObservation(
                observation_id=
                    "prediction:target-p1",

                observable=
                    "SERS response",

                expected_direction=
                    "shift",

                rationale=(
                    "copper-conditioned context "
                    "may alter the gap response"
                ),
            )
        ],
        assumptions=[
            ASSUMPTION_TEXT
        ],
    )


def _review(
    *,
    status,
    assertion_ids,
):
    source_fact = SERSContextFact(
        fact_id="fact:source",
        dimension="environment",
        scientific_role="environment",
        knowledge_state="explicit",
        value="copper substrate",
        normalized_value="copper substrate",
        provenance=[
            SERSContextProvenance(
                kind="grounded_support_node",
                node_ids=[
                    "node:source"
                ],
            )
        ],
    )

    hypothesis_facts = []

    for index, assertion_id in enumerate(
        assertion_ids
    ):
        hypothesis_facts.append(
            SERSContextFact(
                fact_id=
                    f"fact:h:{index}",

                dimension=
                    "environment",

                scientific_role=
                    "environment",

                knowledge_state=
                    "explicit",

                value=
                    "copper-conditioned context",

                normalized_value=
                    "copper conditioned context",

                provenance=[
                    SERSContextProvenance(
                        kind=
                            "hypothesis_assertion",

                        hypothesis_ids=[
                            SOURCE_H
                        ],

                        excerpt=
                            SOURCE_ASSERTIONS[
                                assertion_id
                            ],
                    )
                ],

                tags=[
                    "assertion:"
                    + assertion_id,
                ],
            )
        )

    source_signature = (
        SERSContextSignature(
            signature_id=
                "signature:source",

            domain_profile_id=
                "sers_au_ag",

            scope=
                "grounded_premise",

            source_ref_id=
                "stmt:source",

            facts=[
                source_fact
            ],
        )
    )

    hypothesis_signature = (
        SERSContextSignature(
            signature_id=
                "signature:hypothesis",

            domain_profile_id=
                "sers_au_ag",

            scope=
                "hypothesis",

            source_ref_id=
                SOURCE_H,

            facts=
                hypothesis_facts,
        )
    )

    finding = SERSContextFinding(
        finding_id=
            "finding:test",

        dimension=
            "environment",

        status=
            status,

        severity=
            expected_context_finding_severity(
                status
            ),

        left_signature_id=
            source_signature
            .signature_id,

        right_signature_id=
            hypothesis_signature
            .signature_id,

        left_fact_ids=[
            source_fact.fact_id
        ],

        right_fact_ids=[
            row.fact_id
            for row
            in hypothesis_facts
        ],

        rationale=
            "synthetic context finding",

        tags=[
            "assertion:"
            + assertion_id
            for assertion_id
            in assertion_ids
        ],
    )

    return SERSContextReview(
        review_id=
            "review:test",

        hypothesis_id=
            SOURCE_H,

        signatures=[
            source_signature,
            hypothesis_signature,
        ],

        findings=[
            finding
        ],

        status=
            expected_context_review_status(
                [finding]
            ),
    )


def _normalize(
    review,
    *,
    target_card=None,
    lineage=None,
):
    return (
        SERSContextFindingActionAdapter()
        .normalize(
            review,
            target_card=(
                target_card
                or _target_card()
            ),
            target_portfolio_id=
                "portfolio:target",
            source_portfolio_id=
                "portfolio:source",
            source_artifact_id=
                review.review_id,
            lineage_ref_ids=(
                ["lineage:test"]
                if lineage is None
                else lineage
            ),
        )
    )


def test_s1_status_authority_mapping_informational():
    review = _review(
        status=
            "compatible_extension",
        assertion_ids=[
            f"central:{SOURCE_H}"
        ],
    )

    rows = _normalize(
        review
    )

    assert len(rows) == 1
    assert (
        rows[0].authority
        == "informational"
    )


def test_s1_unknown_maps_to_advisory_not_terminal():
    review = _review(
        status="unknown",
        assertion_ids=[
            f"central:{SOURCE_H}"
        ],
    )

    rows = _normalize(
        review
    )

    assert len(rows) == 1
    assert (
        rows[0].authority
        == "advisory"
    )


def test_s1_context_conflation_maps_to_actionable():
    review = _review(
        status=
            "context_conflation",
        assertion_ids=[
            f"central:{SOURCE_H}"
        ],
    )

    rows = _normalize(
        review
    )

    assert len(rows) == 1
    assert (
        rows[0].authority
        == "actionable"
    )


def test_multiscope_s1_finding_explodes_to_local_g1_scopes():
    review = _review(
        status=
            "role_mismatch",

        assertion_ids=[
            f"central:{SOURCE_H}",
            f"bridge:{SOURCE_H}",
            "prediction:source-p1",
            f"assumption:{SOURCE_H}:0",
        ],
    )

    rows = _normalize(
        review
    )

    assert {
        row.source_scope.kind
        for row in rows
    } == {
        "central",
        "bridge",
        "prediction",
        "assumption",
    }

    assert {
        row.target_scope.kind
        for row in rows
    } == {
        "central",
        "bridge",
        "prediction",
        "assumption",
    }

    by_kind = {
        row.target_scope.kind:
            row
        for row in rows
    }

    assert (
        by_kind[
            "central"
        ].target_scope.assertion_ids
        == [
            f"central:{TARGET_H}"
        ]
    )

    assert (
        by_kind[
            "bridge"
        ].target_scope.assertion_ids
        == [
            f"bridge:{TARGET_H}"
        ]
    )

    assert (
        by_kind[
            "prediction"
        ].target_scope.assertion_ids
        == [
            "prediction:target-p1"
        ]
    )

    assert (
        by_kind[
            "assumption"
        ].target_scope.assertion_ids
        == [
            f"assumption:{TARGET_H}:0"
        ]
    )

    assert all(
        row.authority
        == "actionable"
        for row in rows
    )


def test_cross_generation_binding_requires_explicit_lineage():
    review = _review(
        status="unknown",
        assertion_ids=[
            f"central:{SOURCE_H}"
        ],
    )

    with pytest.raises(
        SERSActionBindingError,
        match="requires explicit",
    ):
        _normalize(
            review,
            lineage=[],
        )


def test_changed_scientific_assertion_is_not_silently_rebound():
    review = _review(
        status=
            "role_mismatch",
        assertion_ids=[
            f"bridge:{SOURCE_H}"
        ],
    )

    with pytest.raises(
        SERSActionBindingError,
        match="S1 re-review is required",
    ):
        _normalize(
            review,
            target_card=
                _target_card(
                    bridge_text=(
                        "Scientifically changed "
                        "refined bridge."
                    )
                ),
        )
