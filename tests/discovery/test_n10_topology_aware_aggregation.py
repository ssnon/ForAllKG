import pytest

from pipeline_core.discovery.novelty_selection_topology_aggregation import (
    TopologyAwareAtomicClaim,
    aggregate_topology_aware_nonobviousness,
)


def _claim(
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
        higher_order_relation_basis=tuple(basis),
        higher_order_component_claim_ids=tuple(
            components
        ),
    )


def test_known_enabling_components_do_not_block_pno_composite():
    claims = (
        _claim(
            "ab",
            "REQUIRED_ENABLING_RELATION",
            "SATURATED_PRIOR_ART",
        ),
        _claim(
            "bc",
            "REQUIRED_ENABLING_RELATION",
            "ROUTINE_FROM_PRIOR_ART",
        ),
        _claim(
            "abc",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
            kind="composite",
            basis=(
                "A is linked to C through B.",
            ),
            components=(
                "ab",
                "bc",
            ),
        ),
    )

    result = (
        aggregate_topology_aware_nonobviousness(
            claims
        )
    )

    assert result.selection_class == "ELIGIBLE"
    assert (
        result.positive_nonobviousness_authority
        is True
    )

    assert result.topology_edges == (
        ("abc", "ab"),
        ("abc", "bc"),
    )


def test_nested_novelty_bearing_component_is_not_silently_demoted():
    claims = (
        _claim(
            "ab",
            "NOVELTY_BEARING",
            "SATURATED_PRIOR_ART",
        ),
        _claim(
            "bc",
            "REQUIRED_ENABLING_RELATION",
            "SATURATED_PRIOR_ART",
        ),
        _claim(
            "abc",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
            kind="composite",
            basis=(
                "A is linked to C through B.",
            ),
            components=(
                "ab",
                "bc",
            ),
        ),
    )

    result = (
        aggregate_topology_aware_nonobviousness(
            claims
        )
    )

    assert result.selection_class == "INELIGIBLE"
    assert result.blocking_claim_ids == ("ab",)

    assert (
        result.nested_novelty_bearing_component_ids
        == ("ab",)
    )

    assert (
        "nested_novelty_bearing_components_"
        "remain_selection_relevant"
        in result.reason_codes
    )


def test_unresolved_nested_novelty_bearing_component_is_conditional():
    claims = (
        _claim(
            "ab",
            "NOVELTY_BEARING",
            "INSUFFICIENT_FOR_JUDGMENT",
        ),
        _claim(
            "abc",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
            kind="composite",
            basis=(
                "A is linked to C through B.",
            ),
            components=("ab",),
        ),
    )

    result = (
        aggregate_topology_aware_nonobviousness(
            claims
        )
    )

    assert result.selection_class == "CONDITIONAL"
    assert (
        result.positive_nonobviousness_authority
        is False
    )


def test_missing_composite_basis_cannot_produce_eligible_authority():
    claims = (
        _claim(
            "ab",
            "REQUIRED_ENABLING_RELATION",
            "SATURATED_PRIOR_ART",
        ),
        _claim(
            "abc",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
            kind="composite",
            basis=(),
            components=("ab",),
        ),
    )

    result = (
        aggregate_topology_aware_nonobviousness(
            claims
        )
    )

    assert result.selection_class == "CONDITIONAL"

    assert (
        result.positive_nonobviousness_authority
        is False
    )

    assert (
        result.structurally_unresolved_claim_ids
        == ("abc",)
    )

    assert result.action == (
        "REFINE_HIGHER_ORDER_RELATION_SPECIFICATION"
    )


def test_missing_basis_never_rescues_routine_novelty_branch():
    claims = (
        _claim(
            "ab",
            "NOVELTY_BEARING",
            "SATURATED_PRIOR_ART",
        ),
        _claim(
            "abc",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
            kind="composite",
            basis=(),
            components=("ab",),
        ),
    )

    result = (
        aggregate_topology_aware_nonobviousness(
            claims
        )
    )

    assert result.selection_class == "INELIGIBLE"

    assert (
        result.positive_nonobviousness_authority
        is False
    )


def test_non_composite_cannot_have_topology_components():
    claims = (
        _claim(
            "ab",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
            components=("other",),
        ),
        _claim(
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


def test_unknown_component_reference_is_rejected():
    claims = (
        _claim(
            "abc",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
            kind="composite",
            basis=(
                "A is linked to C through B.",
            ),
            components=("missing",),
        ),
    )

    with pytest.raises(
        ValueError,
        match="unknown topology component",
    ):
        aggregate_topology_aware_nonobviousness(
            claims
        )


def test_self_reference_is_rejected():
    claims = (
        _claim(
            "abc",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
            kind="composite",
            basis=(
                "A is linked to C through B.",
            ),
            components=("abc",),
        ),
    )

    with pytest.raises(
        ValueError,
        match="self-reference",
    ):
        aggregate_topology_aware_nonobviousness(
            claims
        )


def test_cyclic_composite_topology_is_rejected():
    claims = (
        _claim(
            "one",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
            kind="composite",
            basis=("Relation one.",),
            components=("two",),
        ),
        _claim(
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


def test_topology_does_not_promote_required_relation():
    claims = (
        _claim(
            "ab",
            "REQUIRED_ENABLING_RELATION",
            "POTENTIALLY_NON_OBVIOUS",
        ),
        _claim(
            "abc",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
            kind="composite",
            basis=(
                "A is linked to C through B.",
            ),
            components=("ab",),
        ),
    )

    result = (
        aggregate_topology_aware_nonobviousness(
            claims
        )
    )

    assert result.novelty_bearing_claim_ids == (
        "abc",
    )

    assert result.required_enabling_claim_ids == (
        "ab",
    )


def test_topology_does_not_demote_novelty_bearing_component():
    claims = (
        _claim(
            "ab",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
        ),
        _claim(
            "abc",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
            kind="composite",
            basis=(
                "A is linked to C through B.",
            ),
            components=("ab",),
        ),
    )

    result = (
        aggregate_topology_aware_nonobviousness(
            claims
        )
    )

    assert result.novelty_bearing_claim_ids == (
        "ab",
        "abc",
    )

    assert result.selection_class == "ELIGIBLE"
