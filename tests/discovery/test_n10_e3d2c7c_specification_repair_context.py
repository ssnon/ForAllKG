from copy import deepcopy

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQueryPlan,
    NoveltyClaim,
)
from pipeline_core.discovery.n10_specification_repair_context import (
    N10SpecificationRepairContext,
    build_n10_specification_repair_context,
)


H = "hypothesis:synthetic"


def _claim(
    *,
    claim_id: str,
    rank: int,
    role: str,
    text: str,
) -> NoveltyClaim:
    return NoveltyClaim(
        claim_id=claim_id,
        hypothesis_id=H,
        claim_rank=rank,
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role=role,
        text=text,
        rationale="synthetic rationale",
    )


def _plan() -> LiteratureQueryPlan:
    return LiteratureQueryPlan(
        plan_id="plan:synthetic",
        plan_sha256="plan-sha",
        source_portfolio_id="portfolio:synthetic",
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id=H,
                title="Synthetic hypothesis",
                claims=[
                    _claim(
                        claim_id="claim:novel",
                        rank=1,
                        role="NOVELTY_BEARING",
                        text=(
                            "Factor M moderates relation X to Y."
                        ),
                    ),
                    _claim(
                        claim_id="claim:test",
                        rank=2,
                        role="TESTING_PREDICTION",
                        text=(
                            "Measured Y differs across M."
                        ),
                    ),
                ],
            )
        ],
    )


def _decision(
    *,
    claim_id: str,
    claim_text: str,
    missing: list[str],
) -> dict:
    return {
        "claim": {
            "hypothesis_id": H,
            "claim_id": claim_id,
            "claim_text": claim_text,
        },
        "specification": {
            "status":
                "NEEDS_REFINEMENT",
            "missing_fields":
                list(missing),
            "reason_codes": [
                "atomic_residue_under_specified",
                *[
                    "missing_" + field
                    for field in missing
                ],
            ],
        },
        "shadow_state":
            "NEEDS_REFINEMENT",
        "next_action":
            "REFINE_HYPOTHESIS_SPECIFICATION",
    }


def _intake() -> dict:
    return {
        "schema_version":
            "nonobviousness-shadow-v1",
        "shadow_only":
            True,
        "scientific_selection_changed":
            False,
        "source_query_plan_id":
            "plan:synthetic",
        "source_query_plan_sha256":
            "plan-sha",
        "source_external_report_id":
            "external-report:synthetic",
        "source_external_report_sha256":
            "external-report-sha",
        "hypotheses": [
            {
                "hypothesis_id":
                    H,
                "claims": [
                    _decision(
                        claim_id="claim:novel",
                        claim_text=(
                            "Factor M moderates relation X to Y."
                        ),
                        missing=[
                            "required_bridge",
                        ],
                    ),
                    _decision(
                        claim_id="claim:test",
                        claim_text=(
                            "Measured Y differs across M."
                        ),
                        missing=[
                            "falsification_condition",
                        ],
                    ),
                ],
            }
        ],
    }


def _gate() -> dict:
    return {
        "schema_version":
            "scientific-novelty-fallback-gate-v2",
        "production_authority":
            True,
        "authority_scope":
            "alpha6_post_generation_candidate",
        "conditional_is_positive":
            False,
        "absence_is_novelty":
            False,
        "candidate_semantics_preserved":
            True,
        "gates": [
            {
                "hypothesis_id":
                    H,
                "selection_class":
                    "CONDITIONAL",
                "base_aggregation_action":
                    "REFINE_NOVELTY_BEARING_SPECIFICATION",
                "positive_nonobviousness_authority":
                    False,
                "fallback_allowed":
                    False,
            }
        ],
    }


def _build():
    return build_n10_specification_repair_context(
        source_hypothesis_id=H,
        query_plan=_plan(),
        intake_shadow=_intake(),
        post_generation_gate=_gate(),
    )


def test_builds_deterministically_and_is_non_authoritative():
    a = _build()
    b = _build()

    assert a == b
    assert a.context_id == b.context_id
    assert a.context_sha256 == b.context_sha256

    assert a.diagnostic_only is True
    assert a.production_authority is False
    assert a.scientific_evidence_authority is False
    assert (
        a.external_prior_art_can_be_positive_premise
        is False
    )
    assert a.absence_is_novelty is False


def test_only_novelty_bearing_needs_refinement_is_emitted():
    context = _build()

    assert [
        row.claim_id
        for row in context.claim_diagnostics
    ] == [
        "claim:novel"
    ]

    assert (
        context.claim_diagnostics[0]
        .missing_fields
        == ["required_bridge"]
    )


def test_non_conditional_gate_fails_closed():
    gate = _gate()
    gate["gates"][0][
        "selection_class"
    ] = "ELIGIBLE"

    with pytest.raises(
        ValueError,
        match="requires CONDITIONAL",
    ):
        build_n10_specification_repair_context(
            source_hypothesis_id=H,
            query_plan=_plan(),
            intake_shadow=_intake(),
            post_generation_gate=gate,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "positive_nonobviousness_authority",
            True,
        ),
        (
            "fallback_allowed",
            True,
        ),
    ],
)
def test_authoritative_or_fallback_gate_fails_closed(
    field,
    value,
):
    gate = _gate()
    gate["gates"][0][
        field
    ] = value

    with pytest.raises(
        ValueError,
    ):
        build_n10_specification_repair_context(
            source_hypothesis_id=H,
            query_plan=_plan(),
            intake_shadow=_intake(),
            post_generation_gate=gate,
        )


def test_wrong_repair_action_fails_closed():
    gate = _gate()

    gate["gates"][0][
        "base_aggregation_action"
    ] = (
        "RESOLVE_NOVELTY_BEARING_EVIDENCE"
    )

    with pytest.raises(
        ValueError,
        match="frozen specification-repair action",
    ):
        build_n10_specification_repair_context(
            source_hypothesis_id=H,
            query_plan=_plan(),
            intake_shadow=_intake(),
            post_generation_gate=gate,
        )


def test_query_plan_provenance_drift_fails_closed():
    intake = _intake()

    intake[
        "source_query_plan_sha256"
    ] = "wrong-sha"

    with pytest.raises(
        ValueError,
        match="query-plan SHA mismatch",
    ):
        build_n10_specification_repair_context(
            source_hypothesis_id=H,
            query_plan=_plan(),
            intake_shadow=intake,
            post_generation_gate=_gate(),
        )


def test_claim_text_drift_fails_closed():
    intake = _intake()

    intake[
        "hypotheses"
    ][0][
        "claims"
    ][0][
        "claim"
    ][
        "claim_text"
    ] = "A different scientific claim."

    with pytest.raises(
        ValueError,
        match="claim text drift",
    ):
        build_n10_specification_repair_context(
            source_hypothesis_id=H,
            query_plan=_plan(),
            intake_shadow=intake,
            post_generation_gate=_gate(),
        )


def test_unknown_missing_field_fails_closed():
    intake = _intake()

    spec = (
        intake[
            "hypotheses"
        ][0][
            "claims"
        ][0][
            "specification"
        ]
    )

    spec[
        "missing_fields"
    ] = [
        "invent_a_new_mechanism"
    ]

    spec[
        "reason_codes"
    ] = [
        "atomic_residue_under_specified",
        "missing_invent_a_new_mechanism",
    ]

    with pytest.raises(
        ValueError,
        match="unsupported missing specification fields",
    ):
        build_n10_specification_repair_context(
            source_hypothesis_id=H,
            query_plan=_plan(),
            intake_shadow=intake,
            post_generation_gate=_gate(),
        )


def test_missing_field_requires_matching_reason_code():
    intake = _intake()

    spec = (
        intake[
            "hypotheses"
        ][0][
            "claims"
        ][0][
            "specification"
        ]
    )

    spec[
        "reason_codes"
    ] = [
        "atomic_residue_under_specified",
    ]

    with pytest.raises(
        ValueError,
        match="matching reason code",
    ):
        build_n10_specification_repair_context(
            source_hypothesis_id=H,
            query_plan=_plan(),
            intake_shadow=intake,
            post_generation_gate=_gate(),
        )


def test_no_novelty_bearing_diagnostic_fails_closed():
    intake = _intake()

    # Make the novelty-bearing branch READY rather than repairable.
    decision = (
        intake[
            "hypotheses"
        ][0][
            "claims"
        ][0]
    )

    decision[
        "shadow_state"
    ] = "READY_FOR_CLOSURE"

    decision[
        "next_action"
    ] = "TARGETED_CLOSURE_REQUIRED"

    decision[
        "specification"
    ] = {
        "status":
            "READY_FOR_CLOSURE",
        "missing_fields":
            [],
        "reason_codes": [
            "branch_bridge_prediction_falsifier_present"
        ],
    }

    with pytest.raises(
        ValueError,
        match="no matching N9 diagnostic",
    ):
        build_n10_specification_repair_context(
            source_hypothesis_id=H,
            query_plan=_plan(),
            intake_shadow=intake,
            post_generation_gate=_gate(),
        )


def test_contract_forbids_authority_escalation_and_extra_fields():
    context = _build()

    payload = context.model_dump(
        mode="json"
    )

    payload[
        "production_authority"
    ] = True

    with pytest.raises(
        ValidationError,
    ):
        N10SpecificationRepairContext.model_validate(
            payload
        )

    payload = context.model_dump(
        mode="json"
    )

    payload[
        "prior_art_titles"
    ] = [
        "Forbidden external literature"
    ]

    with pytest.raises(
        ValidationError,
    ):
        N10SpecificationRepairContext.model_validate(
            payload
        )


def test_artifact_contains_no_prior_art_match_payload():
    payload = _build().model_dump(
        mode="json"
    )

    forbidden_keys = {
        "work_id",
        "work_ids",
        "prior_art_titles",
        "matches",
        "match_rationales",
        "reviewer_free_text",
    }

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                assert key not in forbidden_keys
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
