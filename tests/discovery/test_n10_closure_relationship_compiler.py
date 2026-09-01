from pipeline_core.discovery.novelty_closure_relationships import (
    ClosureRelationshipAssessmentDraft,
    compile_closure_relationship_assessment,
)


def _review(
    slot,
    *,
    state="ESTABLISHED",
    work_ids=(),
):
    return {
        "slot": slot,
        "evidence_state": state,
        "positive_work_ids": list(
            work_ids
        ),
    }


def _complete_reviews():
    return [
        _review(
            "BASE_RELATION",
            work_ids=("work:base",),
        ),
        _review(
            "DISTINGUISHING_FACTOR_EFFECT",
            work_ids=("work:factor",),
        ),
        _review(
            "BRIDGE_RELATION",
            work_ids=("work:bridge",),
        ),
        _review(
            "FULL_RELATION",
            state="NOT_FOUND",
        ),
    ]


def test_traceable_mediation_and_scope_compatibility_compile():
    result = (
        compile_closure_relationship_assessment(
            reviews=_complete_reviews(),
            draft=ClosureRelationshipAssessmentDraft(
                bridge_kind="MEDIATION_CHAIN",
                scope_compatibility="COMPATIBLE",
                bridge_basis_work_ids=[
                    "work:bridge",
                ],
                scope_basis_work_ids=[
                    "work:base",
                    "work:factor",
                    "work:bridge",
                ],
                interpretation=(
                    "The established lower-order relations "
                    "occupy a compatible scientific scope."
                ),
            ),
        )
    )

    assert (
        result.bridge_kind
        == "MEDIATION_CHAIN"
    )
    assert (
        result.scope_compatible
        is True
    )
    assert result.reason_codes == ()


def test_interaction_compatible_requires_established_bridge():
    reviews = _complete_reviews()
    reviews[2] = _review(
        "BRIDGE_RELATION",
        state="UNASSESSED",
    )

    result = (
        compile_closure_relationship_assessment(
            reviews=reviews,
            draft=ClosureRelationshipAssessmentDraft(
                bridge_kind="INTERACTION_COMPATIBLE",
                scope_compatibility="UNASSESSED",
                bridge_basis_work_ids=[
                    "work:bridge",
                ],
                interpretation=(
                    "Proposed interaction-compatible bridge."
                ),
            ),
        )
    )

    assert result.bridge_kind == "NONE"
    assert result.scope_compatible is False
    assert (
        "bridge_kind_requires_established_bridge_relation"
        in result.reason_codes
    )


def test_unknown_retrieved_work_cannot_support_bridge_or_scope():
    result = (
        compile_closure_relationship_assessment(
            reviews=_complete_reviews(),
            draft=ClosureRelationshipAssessmentDraft(
                bridge_kind="MEDIATION_CHAIN",
                scope_compatibility="COMPATIBLE",
                bridge_basis_work_ids=[
                    "work:not-positive",
                ],
                scope_basis_work_ids=[
                    "work:base",
                    "work:factor",
                    "work:not-positive",
                ],
                interpretation=(
                    "Contains an unsupported work ID."
                ),
            ),
        )
    )

    assert result.bridge_kind == "NONE"
    assert result.scope_compatible is False

    assert (
        "bridge_basis_contains_nonpositive_work"
        in result.reason_codes
    )
    assert (
        "scope_basis_contains_nonpositive_work"
        in result.reason_codes
    )


def test_scope_compatible_requires_basis_from_every_lower_order_slot():
    result = (
        compile_closure_relationship_assessment(
            reviews=_complete_reviews(),
            draft=ClosureRelationshipAssessmentDraft(
                bridge_kind="NONE",
                scope_compatibility="COMPATIBLE",
                scope_basis_work_ids=[
                    "work:base",
                    "work:bridge",
                ],
                interpretation=(
                    "Factor evidence was not cited."
                ),
            ),
        )
    )

    assert result.scope_compatible is False

    assert (
        "scope_basis_does_not_cover_all_lower_order_slots"
        in result.reason_codes
    )


def test_unassessed_relationships_fail_closed():
    result = (
        compile_closure_relationship_assessment(
            reviews=_complete_reviews(),
            draft=ClosureRelationshipAssessmentDraft(
                interpretation=(
                    "Insufficient evidence for cross-slot semantics."
                ),
            ),
        )
    )

    assert result.bridge_kind == "NONE"
    assert result.scope_compatible is False

    assert (
        "bridge_kind_unassessed_fail_closed"
        in result.reason_codes
    )
    assert (
        "scope_compatibility_unassessed_fail_closed"
        in result.reason_codes
    )


def test_partial_or_negative_slot_cannot_supply_positive_scope_basis():
    reviews = _complete_reviews()
    reviews[1] = _review(
        "DISTINGUISHING_FACTOR_EFFECT",
        state="NOT_FOUND",
    )

    result = (
        compile_closure_relationship_assessment(
            reviews=reviews,
            draft=ClosureRelationshipAssessmentDraft(
                bridge_kind="MEDIATION_CHAIN",
                scope_compatibility="COMPATIBLE",
                bridge_basis_work_ids=[
                    "work:bridge",
                ],
                scope_basis_work_ids=[
                    "work:base",
                    "work:factor",
                    "work:bridge",
                ],
                interpretation=(
                    "Negative factor slot must not become "
                    "positive compatibility evidence."
                ),
            ),
        )
    )

    assert (
        result.bridge_kind
        == "MEDIATION_CHAIN"
    )

    # Bridge type can still be classified from its own positive slot,
    # but cross-slot scope compatibility cannot be asserted.
    assert result.scope_compatible is False

    assert (
        "scope_compatibility_requires_all_lower_order_slots_established"
        in result.reason_codes
    )
