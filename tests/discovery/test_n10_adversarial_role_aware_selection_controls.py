import pytest

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
    basis=(),
    components=(),
):
    return NoveltyClaim(
        claim_id=claim_id,
        hypothesis_id="hypothesis:synthetic",
        claim_rank=1,
        kind=kind,
        importance=importance,
        novelty_selection_role=role,
        text=claim_id,
        rationale="Synthetic adversarial control.",
        higher_order_relation_basis=list(
            basis
        ),
        higher_order_component_claim_ids=list(
            components
        ),
    )


def _plan(claims):
    return LiteratureQueryPlan(
        plan_id="plan:synthetic",
        plan_sha256="synthetic-sha",
        source_portfolio_id="portfolio:synthetic",
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="hypothesis:synthetic",
                title="Synthetic control",
                claims=claims,
            )
        ],
    )


def _compile(
    claims,
    outcomes,
):
    result = build_hypothesis_selection_shadow_v2(
        query_plan=_plan(claims),
        atomic_outcomes=outcomes,
    )

    assert result["shadow_only"] is True
    assert result["production_authority"] is False

    assert (
        result[
            "alpha6_original_fallback_authority"
        ]
        is False
    )

    row = result["hypotheses"][0]

    # Shadow-v2 NEVER directly grants Alpha6 fallback,
    # even for ELIGIBLE scientific-selection states.
    assert row["fallback_allowed"] is False

    return row


def _outcome(
    value,
    *reasons,
):
    return {
        "nonobviousness_outcome": value,
        "reason_codes": list(reasons),
    }


def test_known_enabling_components_do_not_destroy_pno_composite():
    row = _compile(
        [
            _claim(
                "a-b",
                "REQUIRED_ENABLING_RELATION",
            ),
            _claim(
                "b-c",
                "REQUIRED_ENABLING_RELATION",
            ),
            _claim(
                "a-b-c",
                "NOVELTY_BEARING",
                kind="composite",
                importance="core",
                basis=(
                    "A is linked to C through B.",
                ),
                components=(
                    "a-b",
                    "b-c",
                ),
            ),
        ],
        {
            "a-b": _outcome(
                "SATURATED_PRIOR_ART",
                "atomic_claim_saturated_by_prior_art",
            ),
            "b-c": _outcome(
                "ROUTINE_FROM_PRIOR_ART",
            ),
            "a-b-c": _outcome(
                "POTENTIALLY_NON_OBVIOUS",
            ),
        },
    )

    assert row["selection_class"] == "ELIGIBLE"

    assert (
        row[
            "shadow_positive_nonobviousness_authority"
        ]
        is True
    )

    assert (
        row["action"]
        == "KEEP_ROLE_AWARE_NONOBVIOUS_CANDIDATE"
    )


def test_routine_nested_novelty_branch_cannot_hide_behind_composite():
    row = _compile(
        [
            _claim(
                "a-b",
                "NOVELTY_BEARING",
                importance="core",
            ),
            _claim(
                "a-b-c",
                "NOVELTY_BEARING",
                kind="composite",
                importance="core",
                basis=(
                    "A is linked to C through B.",
                ),
                components=("a-b",),
            ),
        ],
        {
            "a-b": _outcome(
                "ROUTINE_FROM_PRIOR_ART",
            ),
            "a-b-c": _outcome(
                "POTENTIALLY_NON_OBVIOUS",
            ),
        },
    )

    assert row["selection_class"] == "INELIGIBLE"

    assert row["blocking_claim_ids"] == [
        "a-b"
    ]

    assert (
        row[
            "nested_novelty_bearing_component_ids"
        ]
        == ["a-b"]
    )

    assert row["action"] == (
        "REMOVE_OR_REAXIS_ROUTINE_NOVELTY_BRANCH"
    )


def test_saturated_nested_novelty_branch_is_also_decisive():
    row = _compile(
        [
            _claim(
                "component",
                "NOVELTY_BEARING",
                importance="core",
            ),
            _claim(
                "composite",
                "NOVELTY_BEARING",
                kind="composite",
                importance="core",
                basis=(
                    "The explicit composite relation.",
                ),
                components=("component",),
            ),
        ],
        {
            "component": _outcome(
                "SATURATED_PRIOR_ART",
            ),
            "composite": _outcome(
                "POTENTIALLY_NON_OBVIOUS",
            ),
        },
    )

    assert row["selection_class"] == "INELIGIBLE"

    assert row["action"] == (
        "REMOVE_OR_REAXIS_ROUTINE_NOVELTY_BRANCH"
    )


def test_missing_higher_order_basis_never_becomes_positive():
    row = _compile(
        [
            _claim(
                "composite",
                "NOVELTY_BEARING",
                kind="composite",
                importance="core",
                basis=(),
            )
        ],
        {
            "composite": _outcome(
                "POTENTIALLY_NON_OBVIOUS",
            )
        },
    )

    assert row["selection_class"] == "CONDITIONAL"

    assert (
        row[
            "shadow_positive_nonobviousness_authority"
        ]
        is False
    )

    assert row["action"] == (
        "REFINE_HIGHER_ORDER_RELATION_SPECIFICATION"
    )

    assert (
        row["structurally_unresolved_claim_ids"]
        == ["composite"]
    )


def test_partial_prior_art_is_resolution_not_under_specification():
    row = _compile(
        [
            _claim(
                "composite",
                "NOVELTY_BEARING",
                kind="composite",
                importance="core",
                basis=(
                    "The explicit composite relation.",
                ),
            )
        ],
        {
            "composite": _outcome(
                "NEEDS_REFINEMENT",
                "partial_prior_art_requires_resolution",
            )
        },
    )

    assert row["selection_class"] == "CONDITIONAL"

    assert row["action"] == (
        "RESOLVE_NOVELTY_BEARING_PRIOR_ART_RELATION"
    )

    assert row["base_aggregation_action"] == (
        "REFINE_NOVELTY_BEARING_SPECIFICATION"
    )


def test_true_under_specification_remains_specification_refinement():
    row = _compile(
        [
            _claim(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            )
        ],
        {
            "novel": _outcome(
                "NEEDS_REFINEMENT",
                "atomic_specification_incomplete",
                "missing_specification_field:"
                "predicted_observation",
            )
        },
    )

    assert row["selection_class"] == "CONDITIONAL"

    assert row["action"] == (
        "REFINE_NOVELTY_BEARING_SPECIFICATION"
    )


def test_insufficient_evidence_remains_conditional_not_novel():
    row = _compile(
        [
            _claim(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            )
        ],
        {
            "novel": _outcome(
                "INSUFFICIENT_FOR_JUDGMENT",
                "candidate_not_ready_for_adjudication",
            )
        },
    )

    assert row["selection_class"] == "CONDITIONAL"

    assert (
        row[
            "shadow_positive_nonobviousness_authority"
        ]
        is False
    )

    assert row["action"] == (
        "RESOLVE_NOVELTY_BEARING_EVIDENCE"
    )


@pytest.mark.parametrize(
    "known_outcome",
    [
        "POTENTIALLY_NON_OBVIOUS",
        "SATURATED_PRIOR_ART",
        "ROUTINE_FROM_PRIOR_ART",
    ],
)
def test_resolved_enabling_status_does_not_change_pno_composite_selection(
    known_outcome,
):
    row = _compile(
        [
            _claim(
                "component",
                "REQUIRED_ENABLING_RELATION",
            ),
            _claim(
                "composite",
                "NOVELTY_BEARING",
                kind="composite",
                importance="core",
                basis=(
                    "The explicit composite relation.",
                ),
                components=("component",),
            ),
        ],
        {
            "component": _outcome(
                known_outcome,
            ),
            "composite": _outcome(
                "POTENTIALLY_NON_OBVIOUS",
            ),
        },
    )

    assert row["selection_class"] == "ELIGIBLE"


def test_unresolved_required_enabling_relation_keeps_candidate_conditional():
    row = _compile(
        [
            _claim(
                "component",
                "REQUIRED_ENABLING_RELATION",
            ),
            _claim(
                "composite",
                "NOVELTY_BEARING",
                kind="composite",
                importance="core",
                basis=(
                    "The explicit composite relation.",
                ),
                components=("component",),
            ),
        ],
        {
            "component": _outcome(
                "INSUFFICIENT_FOR_JUDGMENT",
            ),
            "composite": _outcome(
                "POTENTIALLY_NON_OBVIOUS",
            ),
        },
    )

    assert row["selection_class"] == "CONDITIONAL"

    assert row["action"] == (
        "RESOLVE_REQUIRED_ENABLING_RELATION"
    )


def test_under_specified_required_relation_is_role_scoped():
    row = _compile(
        [
            _claim(
                "component",
                "REQUIRED_ENABLING_RELATION",
            ),
            _claim(
                "composite",
                "NOVELTY_BEARING",
                kind="composite",
                importance="core",
                basis=(
                    "The explicit composite relation.",
                ),
                components=("component",),
            ),
        ],
        {
            "component": _outcome(
                "NEEDS_REFINEMENT",
                "atomic_specification_incomplete",
            ),
            "composite": _outcome(
                "POTENTIALLY_NON_OBVIOUS",
            ),
        },
    )

    assert row["selection_class"] == "CONDITIONAL"

    assert row["action"] == (
        "REFINE_REQUIRED_ENABLING_"
        "RELATION_SPECIFICATION"
    )


def test_saturated_testing_prediction_does_not_block_novelty():
    row = _compile(
        [
            _claim(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            ),
            _claim(
                "prediction",
                "TESTING_PREDICTION",
            ),
        ],
        {
            "novel": _outcome(
                "POTENTIALLY_NON_OBVIOUS",
            ),
            "prediction": _outcome(
                "SATURATED_PRIOR_ART",
            ),
        },
    )

    assert row["selection_class"] == "ELIGIBLE"


def test_unresolved_auxiliary_claim_does_not_block_established_novelty():
    row = _compile(
        [
            _claim(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            ),
            _claim(
                "aux",
                "AUXILIARY",
            ),
        ],
        {
            "novel": _outcome(
                "POTENTIALLY_NON_OBVIOUS",
            ),
            "aux": _outcome(
                "INSUFFICIENT_FOR_JUDGMENT",
            ),
        },
    )

    assert row["selection_class"] == "ELIGIBLE"


def test_no_novelty_bearing_claim_fails_closed():
    row = _compile(
        [
            _claim(
                "component",
                "REQUIRED_ENABLING_RELATION",
            ),
            _claim(
                "prediction",
                "TESTING_PREDICTION",
            ),
        ],
        {
            "component": _outcome(
                "POTENTIALLY_NON_OBVIOUS",
            ),
            "prediction": _outcome(
                "POTENTIALLY_NON_OBVIOUS",
            ),
        },
    )

    assert row["selection_class"] == "INELIGIBLE"

    assert row["action"] == (
        "REFINE_NOVELTY_SELECTION_ROLE_SPECIFICATION"
    )


def test_unknown_role_downgrades_otherwise_eligible_candidate():
    row = _compile(
        [
            _claim(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            ),
            _claim(
                "unresolved-role",
                None,
            ),
        ],
        {
            "novel": _outcome(
                "POTENTIALLY_NON_OBVIOUS",
            ),
            "unresolved-role": _outcome(
                "POTENTIALLY_NON_OBVIOUS",
            ),
        },
    )

    assert row["selection_class"] == "CONDITIONAL"

    assert row["action"] == (
        "REFINE_NOVELTY_SELECTION_ROLE_SPECIFICATION"
    )

    assert (
        row[
            "unresolved_selection_role_claim_ids"
        ]
        == ["unresolved-role"]
    )


def test_unknown_role_cannot_rescue_routine_novelty_branch():
    row = _compile(
        [
            _claim(
                "routine",
                "NOVELTY_BEARING",
                importance="core",
            ),
            _claim(
                "unresolved-role",
                None,
            ),
        ],
        {
            "routine": _outcome(
                "ROUTINE_FROM_PRIOR_ART",
            ),
            "unresolved-role": _outcome(
                "POTENTIALLY_NON_OBVIOUS",
            ),
        },
    )

    assert row["selection_class"] == "INELIGIBLE"

    assert row["action"] == (
        "REMOVE_OR_REAXIS_ROUTINE_NOVELTY_BRANCH"
    )


def test_topology_does_not_demote_independent_novelty_component():
    row = _compile(
        [
            _claim(
                "component",
                "NOVELTY_BEARING",
                importance="core",
            ),
            _claim(
                "composite",
                "NOVELTY_BEARING",
                kind="composite",
                importance="core",
                basis=(
                    "The explicit composite relation.",
                ),
                components=("component",),
            ),
        ],
        {
            "component": _outcome(
                "POTENTIALLY_NON_OBVIOUS",
            ),
            "composite": _outcome(
                "POTENTIALLY_NON_OBVIOUS",
            ),
        },
    )

    assert row["selection_class"] == "ELIGIBLE"

    assert row["novelty_bearing_claim_ids"] == [
        "component",
        "composite",
    ]

    assert (
        row[
            "nested_novelty_bearing_component_ids"
        ]
        == ["component"]
    )


def test_degrading_novelty_from_positive_to_insufficient_cannot_improve_selection():
    positive = _compile(
        [
            _claim(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            )
        ],
        {
            "novel": _outcome(
                "POTENTIALLY_NON_OBVIOUS",
            )
        },
    )

    insufficient = _compile(
        [
            _claim(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            )
        ],
        {
            "novel": _outcome(
                "INSUFFICIENT_FOR_JUDGMENT",
            )
        },
    )

    assert (
        positive["selection_class"]
        == "ELIGIBLE"
    )

    assert (
        insufficient["selection_class"]
        == "CONDITIONAL"
    )


def test_degrading_novelty_from_positive_to_routine_becomes_ineligible():
    positive = _compile(
        [
            _claim(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            )
        ],
        {
            "novel": _outcome(
                "POTENTIALLY_NON_OBVIOUS",
            )
        },
    )

    routine = _compile(
        [
            _claim(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            )
        ],
        {
            "novel": _outcome(
                "ROUTINE_FROM_PRIOR_ART",
            )
        },
    )

    assert positive["selection_class"] == "ELIGIBLE"
    assert routine["selection_class"] == "INELIGIBLE"
