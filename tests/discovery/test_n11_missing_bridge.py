from __future__ import annotations

import pytest

from pipeline_core.discovery.nonobviousness_missing_bridge import (
    compile_missing_bridge_opportunity,
)


def execution_plan():
    return {
        "plan_id": "closure_plan:test",
        "source_portfolio_id": "portfolio:test",
        "source_hypothesis_id": "hypothesis:test",
        "source_claim_id": "claim:test",
        "targets": [
            {
                "slot": "BASE_RELATION",
                "search_terms": [
                    "SERS response",
                    "electromagnetic enhancement",
                    "chemical enhancement",
                    "relative contribution",
                ],
                "identity_anchor_terms": [],
                "source_text": (
                    "SERS response | electromagnetic enhancement "
                    "| chemical enhancement | relative contribution"
                ),
            },
            {
                "slot": (
                    "DISTINGUISHING_FACTOR_EFFECT"
                ),
                "search_terms": [
                    "interparticle spacing",
                    "SERS response",
                    "electromagnetic enhancement",
                    "chemical enhancement",
                    "relative contribution",
                ],
                "identity_anchor_terms": [
                    "interparticle spacing"
                ],
                "source_text": (
                    "identity=interparticle spacing; "
                    "relation_context=SERS response"
                ),
            },
            {
                "slot": "BRIDGE_RELATION",
                "search_terms": [
                    "interparticle spacing",
                    "SERS response",
                    "electromagnetic enhancement",
                    "chemical enhancement",
                    "relative contribution",
                    "charge transfer enhancement",
                ],
                "identity_anchor_terms": [
                    "interparticle spacing"
                ],
                "source_text": (
                    "Existing evidence grounds components. "
                    "The new inference is that interparticle "
                    "spacing could alter relative mechanistic "
                    "contribution."
                ),
            },
            {
                "slot": "FULL_RELATION",
                "search_terms": [
                    "interparticle spacing",
                    "relative contribution",
                ],
                "identity_anchor_terms": [
                    "interparticle spacing"
                ],
                "source_text": (
                    "Interparticle spacing changes the relative "
                    "electromagnetic and chemical contributions."
                ),
            },
        ],
    }


def reviews(
    *,
    bridge_state="NOT_FOUND",
    full_state="NOT_FOUND",
):
    bridge_positive = (
        ["work:bridge"]
        if bridge_state == "ESTABLISHED"
        else []
    )

    full_positive = (
        ["work:full"]
        if full_state == "ESTABLISHED"
        else []
    )

    return [
        {
            "slot": "BASE_RELATION",
            "evidence_state": "ESTABLISHED",
            "positive_work_ids": [
                "work:base:1",
                "work:base:2",
            ],
        },
        {
            "slot": (
                "DISTINGUISHING_FACTOR_EFFECT"
            ),
            "evidence_state": "ESTABLISHED",
            "positive_work_ids": [
                "work:factor:1",
            ],
        },
        {
            "slot": "BRIDGE_RELATION",
            "evidence_state": bridge_state,
            "positive_work_ids": bridge_positive,
        },
        {
            "slot": "FULL_RELATION",
            "evidence_state": full_state,
            "positive_work_ids": full_positive,
        },
    ]


def relationships(
    *,
    scope="UNASSESSED",
    review_performed=False,
):
    return {
        "compiled": {
            "bridge_kind": "NONE",
            "scope_compatible": False,
            "bridge_basis_work_ids": [],
            "scope_basis_work_ids": [],
            "reason_codes": [
                "bridge_kind_unassessed_fail_closed",
                "scope_compatibility_unassessed_fail_closed",
            ],
            "interpretation": "fail closed",
        },
        "draft": {
            "bridge_kind": "UNASSESSED",
            "scope_compatibility": scope,
            "bridge_basis_work_ids": [],
            "scope_basis_work_ids": [],
            "interpretation": "not reviewed",
        },
        "review_performed": review_performed,
    }


def test_current_c4c_shape_compiles_missing_bridge():
    result = compile_missing_bridge_opportunity(
        execution_plan=execution_plan(),
        slot_reviews=reviews(),
        closure_relationships=relationships(),
    )

    assert (
        result.status
        == "ELIGIBLE_FOR_GROUNDED_BRIDGE_SEARCH"
    )

    opportunity = result.opportunity

    assert opportunity is not None

    assert opportunity.factor_identity_terms == [
        "interparticle spacing"
    ]

    assert opportunity.base_relation_terms == [
        "SERS response",
        "electromagnetic enhancement",
        "chemical enhancement",
        "relative contribution",
    ]

    assert opportunity.established_base_work_ids == [
        "work:base:1",
        "work:base:2",
    ]

    assert opportunity.established_factor_work_ids == [
        "work:factor:1",
    ]

    assert (
        opportunity.bridge_state
        == "NOT_FOUND"
    )

    assert (
        opportunity.full_state
        == "NOT_FOUND"
    )

    assert (
        opportunity.production_authority
        is False
    )


def test_common_anchor_and_navigation_are_never_sufficient():
    result = compile_missing_bridge_opportunity(
        execution_plan=execution_plan(),
        slot_reviews=reviews(),
        closure_relationships=relationships(),
    )

    req = result.opportunity.search_requirement

    assert req.allowed_path_classes == [
        "DIRECT_SCIENTIFIC_CHAIN"
    ]

    assert req.blocked_path_classes == [
        "COMMON_ANCHOR_CONTEXT",
        "NAVIGATION_ONLY",
    ]

    assert (
        req.common_anchor_context_is_sufficient
        is False
    )

    assert (
        req.navigation_only_is_sufficient
        is False
    )

    assert (
        req.opportunity_is_positive_evidence
        is False
    )


def test_bridge_unassessed_is_not_missing_bridge_opportunity():
    result = compile_missing_bridge_opportunity(
        execution_plan=execution_plan(),
        slot_reviews=reviews(
            bridge_state="UNASSESSED",
        ),
        closure_relationships=relationships(),
    )

    assert result.status == "NOT_ELIGIBLE"
    assert result.opportunity is None

    assert (
        "bridge_relation_requires_more_closure"
        in result.reason_codes
    )


def test_established_bridge_does_not_trigger_search():
    result = compile_missing_bridge_opportunity(
        execution_plan=execution_plan(),
        slot_reviews=reviews(
            bridge_state="ESTABLISHED",
        ),
        closure_relationships=relationships(),
    )

    assert result.status == "NOT_ELIGIBLE"

    assert (
        "bridge_relation_already_established"
        in result.reason_codes
    )


def test_established_full_relation_does_not_trigger_search():
    result = compile_missing_bridge_opportunity(
        execution_plan=execution_plan(),
        slot_reviews=reviews(
            full_state="ESTABLISHED",
        ),
        closure_relationships=relationships(),
    )

    assert result.status == "NOT_ELIGIBLE"

    assert (
        "full_relation_already_established"
        in result.reason_codes
    )


def test_explicit_scope_incompatibility_is_not_bridge_search():
    result = compile_missing_bridge_opportunity(
        execution_plan=execution_plan(),
        slot_reviews=reviews(),
        closure_relationships=relationships(
            scope="INCOMPATIBLE",
        ),
    )

    assert result.status == "NOT_ELIGIBLE"

    assert (
        "lower_order_scope_explicitly_incompatible"
        in result.reason_codes
    )


def test_not_found_slot_cannot_contain_positive_work():
    rows = reviews()

    next(
        row
        for row in rows
        if row["slot"] == "BRIDGE_RELATION"
    )["positive_work_ids"] = [
        "work:impossible"
    ]

    with pytest.raises(
        ValueError,
        match="NOT_FOUND but contains positive",
    ):
        compile_missing_bridge_opportunity(
            execution_plan=execution_plan(),
            slot_reviews=rows,
            closure_relationships=relationships(),
        )


def test_opportunity_id_is_deterministic():
    kwargs = {
        "execution_plan": execution_plan(),
        "slot_reviews": reviews(),
        "closure_relationships": relationships(),
    }

    first = compile_missing_bridge_opportunity(
        **kwargs
    )

    second = compile_missing_bridge_opportunity(
        **kwargs
    )

    assert (
        first.opportunity.opportunity_id
        == second.opportunity.opportunity_id
    )
