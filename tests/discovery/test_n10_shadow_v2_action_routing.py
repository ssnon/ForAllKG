from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQueryPlan,
    NoveltyClaim,
)
from pipeline_core.discovery.hypothesis_selection_shadow_v2 import (
    build_hypothesis_selection_shadow_v2,
)


def _claim(
    claim_id,
    role,
    *,
    kind="mediator",
    importance="supporting",
    basis=None,
    components=None,
):
    return NoveltyClaim(
        claim_id=claim_id,
        hypothesis_id="hypothesis:1",
        claim_rank=1,
        kind=kind,
        importance=importance,
        novelty_selection_role=role,
        text=claim_id,
        rationale="Synthetic.",
        higher_order_relation_basis=list(
            basis or []
        ),
        higher_order_component_claim_ids=list(
            components or []
        ),
    )


def _plan(claims):
    return LiteratureQueryPlan(
        plan_id="plan:1",
        plan_sha256="plansha",
        source_portfolio_id="portfolio:1",
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="hypothesis:1",
                title="Synthetic",
                claims=claims,
            )
        ],
    )


def test_partial_prior_art_changes_action_not_selection():
    plan = _plan(
        [
            _claim(
                "known",
                "REQUIRED_ENABLING_RELATION",
            ),
            _claim(
                "composite",
                "NOVELTY_BEARING",
                kind="composite",
                importance="core",
                basis=[
                    "A is linked to C through B."
                ],
                components=["known"],
            ),
        ]
    )

    result = build_hypothesis_selection_shadow_v2(
        query_plan=plan,
        atomic_outcomes={
            "known": {
                "nonobviousness_outcome":
                    "SATURATED_PRIOR_ART",
                "reason_codes": [
                    "atomic_claim_saturated_by_prior_art",
                ],
            },
            "composite": {
                "nonobviousness_outcome":
                    "NEEDS_REFINEMENT",
                "reason_codes": [
                    "partial_prior_art_requires_resolution",
                ],
            },
        },
    )

    row = result["hypotheses"][0]

    assert (
        row["selection_class"]
        == "CONDITIONAL"
    )

    assert (
        row["action"]
        == "RESOLVE_NOVELTY_BEARING_"
           "PRIOR_ART_RELATION"
    )

    assert (
        row["base_aggregation_action"]
        == "REFINE_NOVELTY_BEARING_"
           "SPECIFICATION"
    )

    assert (
        row["shadow_positive_nonobviousness_authority"]
        is False
    )

    assert row["fallback_allowed"] is False


def test_true_under_specification_keeps_refinement_action():
    plan = _plan(
        [
            _claim(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            ),
        ]
    )

    result = build_hypothesis_selection_shadow_v2(
        query_plan=plan,
        atomic_outcomes={
            "novel": {
                "nonobviousness_outcome":
                    "NEEDS_REFINEMENT",
                "reason_codes": [
                    "atomic_specification_incomplete",
                    "missing_specification_field:"
                    "predicted_observation",
                ],
            },
        },
    )

    row = result["hypotheses"][0]

    assert (
        row["selection_class"]
        == "CONDITIONAL"
    )

    assert (
        row["action"]
        == "REFINE_NOVELTY_BEARING_"
           "SPECIFICATION"
    )


def test_insufficient_evidence_keeps_evidence_resolution():
    plan = _plan(
        [
            _claim(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            ),
        ]
    )

    result = build_hypothesis_selection_shadow_v2(
        query_plan=plan,
        atomic_outcomes={
            "novel": {
                "nonobviousness_outcome":
                    "INSUFFICIENT_FOR_JUDGMENT",
                "reason_codes": [
                    "candidate_not_ready_for_adjudication",
                ],
            },
        },
    )

    row = result["hypotheses"][0]

    assert (
        row["action"]
        == "RESOLVE_NOVELTY_BEARING_EVIDENCE"
    )


def test_routing_never_grants_production_authority():
    plan = _plan(
        [
            _claim(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            ),
        ]
    )

    result = build_hypothesis_selection_shadow_v2(
        query_plan=plan,
        atomic_outcomes={
            "novel": {
                "nonobviousness_outcome":
                    "POTENTIALLY_NON_OBVIOUS",
                "reason_codes": [],
            },
        },
    )

    row = result["hypotheses"][0]

    assert (
        row["selection_class"]
        == "ELIGIBLE"
    )

    assert row["fallback_allowed"] is False

    assert result["production_authority"] is False

    assert (
        result[
            "alpha6_original_fallback_authority"
        ]
        is False
    )
