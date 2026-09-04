import pytest

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQueryPlan,
    NoveltyClaim,
)
from pipeline_core.discovery.hypothesis_selection_shadow_v2 import (
    build_hypothesis_selection_shadow_v2,
)
from pipeline_core.discovery.novelty_selection_topology_aggregation import (
    TopologyAwareAtomicClaim,
    aggregate_topology_aware_nonobviousness,
)


def _atomic(
    claim_id,
    role,
    outcome,
    *,
    kind="mediator",
    basis=(),
    components=(),
):
    return TopologyAwareAtomicClaim(
        claim_id=claim_id,
        claim_kind=kind,
        novelty_selection_role=role,
        nonobviousness_outcome=outcome,
        higher_order_relation_basis=tuple(
            basis
        ),
        higher_order_component_claim_ids=tuple(
            components
        ),
    )


def _canonical(
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
        rationale="Synthetic corruption control.",
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
                title="Synthetic corruption control",
                claims=claims,
            )
        ],
    )


def test_unknown_runtime_selection_role_is_rejected():
    claims = (
        _atomic(
            "novel",
            "CORRUPTED_ROLE",
            "POTENTIALLY_NON_OBVIOUS",
        ),
    )

    with pytest.raises(
        ValueError,
        match="unsupported novelty selection role",
    ):
        aggregate_topology_aware_nonobviousness(
            claims
        )


def test_unknown_runtime_outcome_is_rejected():
    claims = (
        _atomic(
            "novel",
            "NOVELTY_BEARING",
            "NO_DIRECT_MATCH_FOUND",
        ),
    )

    with pytest.raises(
        ValueError,
        match="unsupported nonobviousness outcome",
    ):
        aggregate_topology_aware_nonobviousness(
            claims
        )


def test_duplicate_claim_id_is_rejected():
    claims = (
        _atomic(
            "same",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
        ),
        _atomic(
            "same",
            "AUXILIARY",
            "POTENTIALLY_NON_OBVIOUS",
        ),
    )

    with pytest.raises(
        ValueError,
        match="duplicate topology-aware claim_id",
    ):
        aggregate_topology_aware_nonobviousness(
            claims
        )


def test_non_composite_cannot_smuggle_component_topology():
    claims = (
        _atomic(
            "ordinary",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
            components=("other",),
        ),
        _atomic(
            "other",
            "AUXILIARY",
            "POTENTIALLY_NON_OBVIOUS",
        ),
    )

    with pytest.raises(
        ValueError,
        match="non-composite",
    ):
        aggregate_topology_aware_nonobviousness(
            claims
        )


def test_duplicate_component_reference_is_rejected():
    claims = (
        _atomic(
            "component",
            "REQUIRED_ENABLING_RELATION",
            "SATURATED_PRIOR_ART",
        ),
        _atomic(
            "composite",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
            kind="composite",
            basis=(
                "Explicit higher-order relation.",
            ),
            components=(
                "component",
                "component",
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="duplicate topology component reference",
    ):
        aggregate_topology_aware_nonobviousness(
            claims
        )


def test_self_referential_composite_is_rejected():
    claims = (
        _atomic(
            "composite",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
            kind="composite",
            basis=(
                "Explicit higher-order relation.",
            ),
            components=("composite",),
        ),
    )

    with pytest.raises(
        ValueError,
        match="self-reference",
    ):
        aggregate_topology_aware_nonobviousness(
            claims
        )


def test_unknown_component_reference_is_rejected():
    claims = (
        _atomic(
            "composite",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
            kind="composite",
            basis=(
                "Explicit higher-order relation.",
            ),
            components=("missing",),
        ),
    )

    with pytest.raises(
        ValueError,
        match="unknown topology component claim_id",
    ):
        aggregate_topology_aware_nonobviousness(
            claims
        )


def test_cyclic_composite_topology_is_rejected():
    claims = (
        _atomic(
            "one",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
            kind="composite",
            basis=("Relation one.",),
            components=("two",),
        ),
        _atomic(
            "two",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
            kind="composite",
            basis=("Relation two.",),
            components=("one",),
        ),
    )

    with pytest.raises(
        ValueError,
        match="cyclic higher-order component topology",
    ):
        aggregate_topology_aware_nonobviousness(
            claims
        )


def test_empty_claim_set_fails_closed():
    result = (
        aggregate_topology_aware_nonobviousness(
            ()
        )
    )

    assert result.selection_class == "INELIGIBLE"

    assert (
        result.positive_nonobviousness_authority
        is False
    )

    assert result.action == (
        "REFINE_NOVELTY_SELECTION_ROLE_SPECIFICATION"
    )


def test_missing_atomic_outcome_is_rejected_by_shadow_compiler():
    plan = _plan(
        [
            _canonical(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="missing atomic nonobviousness outcome",
    ):
        build_hypothesis_selection_shadow_v2(
            query_plan=plan,
            atomic_outcomes={},
        )


def test_unknown_extra_outcome_claim_is_rejected():
    plan = _plan(
        [
            _canonical(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="unknown claim_id",
    ):
        build_hypothesis_selection_shadow_v2(
            query_plan=plan,
            atomic_outcomes={
                "novel":
                    "POTENTIALLY_NON_OBVIOUS",
                "injected":
                    "POTENTIALLY_NON_OBVIOUS",
            },
        )


def test_absence_label_cannot_be_used_as_atomic_novelty_outcome():
    plan = _plan(
        [
            _canonical(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="unsupported atomic",
    ):
        build_hypothesis_selection_shadow_v2(
            query_plan=plan,
            atomic_outcomes={
                "novel":
                    "NO_DIRECT_MATCH_FOUND",
            },
        )


def test_malformed_outcome_reason_codes_are_rejected():
    plan = _plan(
        [
            _canonical(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="reason_codes must be a list or tuple",
    ):
        build_hypothesis_selection_shadow_v2(
            query_plan=plan,
            atomic_outcomes={
                "novel": {
                    "nonobviousness_outcome":
                        "NEEDS_REFINEMENT",
                    "reason_codes":
                        "partial_prior_art_requires_resolution",
                },
            },
        )


def test_missing_composite_basis_cannot_hide_routine_nested_branch():
    plan = _plan(
        [
            _canonical(
                "component",
                "NOVELTY_BEARING",
                importance="core",
            ),
            _canonical(
                "composite",
                "NOVELTY_BEARING",
                kind="composite",
                importance="core",
                basis=(),
                components=("component",),
            ),
        ]
    )

    result = build_hypothesis_selection_shadow_v2(
        query_plan=plan,
        atomic_outcomes={
            "component":
                "ROUTINE_FROM_PRIOR_ART",
            "composite":
                "POTENTIALLY_NON_OBVIOUS",
        },
    )

    row = result["hypotheses"][0]

    # Structural incompleteness can never soften an already decisive
    # routine novelty-bearing branch.
    assert row["selection_class"] == "INELIGIBLE"

    assert (
        row[
            "shadow_positive_nonobviousness_authority"
        ]
        is False
    )

    assert row["blocking_claim_ids"] == [
        "component"
    ]


def test_unknown_role_cannot_create_positive_authority():
    plan = _plan(
        [
            _canonical(
                "unresolved",
                None,
                importance="core",
            ),
        ]
    )

    result = build_hypothesis_selection_shadow_v2(
        query_plan=plan,
        atomic_outcomes={
            "unresolved":
                "POTENTIALLY_NON_OBVIOUS",
        },
    )

    row = result["hypotheses"][0]

    assert (
        row[
            "shadow_positive_nonobviousness_authority"
        ]
        is False
    )

    assert row["selection_class"] == "INELIGIBLE"

    assert row["fallback_allowed"] is False


def test_shadow_compiler_never_grants_production_authority():
    plan = _plan(
        [
            _canonical(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            )
        ]
    )

    result = build_hypothesis_selection_shadow_v2(
        query_plan=plan,
        atomic_outcomes={
            "novel":
                "POTENTIALLY_NON_OBVIOUS",
        },
    )

    assert (
        result["hypotheses"][0][
            "selection_class"
        ]
        == "ELIGIBLE"
    )

    assert result["production_authority"] is False

    assert (
        result[
            "alpha6_original_fallback_authority"
        ]
        is False
    )

    assert (
        result["hypotheses"][0][
            "fallback_allowed"
        ]
        is False
    )
