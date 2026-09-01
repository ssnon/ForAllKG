import pytest

from pipeline_core.discovery.novelty_closure_planner import (
    build_closure_retrieval_plan,
)
from pipeline_core.discovery.novelty_residue import (
    NoveltyResidueClaim,
)


def ready_claim():
    return NoveltyResidueClaim(
        hypothesis_id="hypothesis:threshold",
        claim_id="claim:threshold",
        claim_text=(
            "A critical laser power Pc separates two distinct "
            "regimes of the interparticle-spacing-to-SERS "
            "relationship."
        ),
        claim_kind="distinctive_prediction",
        prior_art_status="NO_DIRECT_MATCH_FOUND",
        disposition="RESIDUAL",
        is_residue=True,

        distinguishing_terms=(
            "critical laser power",
            "two regimes",
        ),
        prior_art_identity_terms=(
            "laser power",
        ),
        relation_nucleus_terms=(
            "interparticle spacing",
            "SERS enhancement",
            "dependence",
        ),

        required_bridge=(
            "Laser power drives a transition at Pc that changes "
            "how interparticle spacing maps to measured SERS "
            "enhancement."
        ),
        predicted_observation=(
            "Below and above Pc, the spacing-to-SERS response "
            "occupies two distinguishable regimes."
        ),
        falsification_condition=(
            "The spacing-to-SERS response varies smoothly with "
            "power and shows no reproducible regime boundary."
        ),

        direct_or_partial_work_ids=(),
        lower_order_work_ids=(
            "work:spacing-sers",
        ),
        component_work_ids=(
            "work:power",
        ),
    )


def test_ready_residue_yields_exactly_four_targets():
    plan = build_closure_retrieval_plan(
        ready_claim()
    )

    assert tuple(
        row.slot
        for row in plan.targets
    ) == (
        "BASE_RELATION",
        "DISTINGUISHING_FACTOR_EFFECT",
        "BRIDGE_RELATION",
        "FULL_RELATION",
    )

    assert all(
        row.evidence_status == "UNASSESSED"
        for row in plan.targets
    )


def test_base_target_uses_only_relation_nucleus():
    claim = ready_claim()

    plan = build_closure_retrieval_plan(
        claim
    )

    base = plan.targets[0]

    assert base.search_terms == (
        "interparticle spacing",
        "SERS enhancement",
        "dependence",
    )

    assert "laser power" not in (
        base.search_query.lower()
    )


def test_factor_target_uses_identity_plus_relation_context():
    plan = build_closure_retrieval_plan(
        ready_claim()
    )

    factor = plan.targets[1]

    assert "laser power" in factor.search_terms
    assert "interparticle spacing" in factor.search_terms
    assert "SERS enhancement" in factor.search_terms

    # It remains explicitly a retrieval target rather than a
    # newly asserted scientific proposition.
    assert (
        factor.target_basis
        == "IDENTITY_PLUS_RELATION_CONTEXT"
    )


def test_bridge_target_is_exactly_source_grounded():
    claim = ready_claim()

    plan = build_closure_retrieval_plan(
        claim
    )

    bridge = plan.targets[2]

    assert (
        bridge.source_text
        == claim.required_bridge
    )

    assert (
        bridge.search_query
        == claim.required_bridge
    )


def test_full_target_is_original_residual_claim():
    claim = ready_claim()

    plan = build_closure_retrieval_plan(
        claim
    )

    full = plan.targets[3]

    assert (
        full.source_text
        == claim.claim_text
    )


def test_under_specified_residue_cannot_enter_planner():
    claim = ready_claim()

    claim = NoveltyResidueClaim(
        **{
            **claim.__dict__,
            "required_bridge": "",
        }
    )

    with pytest.raises(
        ValueError,
        match="READY_FOR_CLOSURE",
    ):
        build_closure_retrieval_plan(
            claim
        )


def test_missing_identity_fails_closed():
    claim = ready_claim()

    claim = NoveltyResidueClaim(
        **{
            **claim.__dict__,
            "prior_art_identity_terms": (),
        }
    )

    with pytest.raises(
        ValueError,
        match="prior_art_identity_terms",
    ):
        build_closure_retrieval_plan(
            claim
        )


def test_missing_relation_nucleus_fails_closed():
    claim = ready_claim()

    claim = NoveltyResidueClaim(
        **{
            **claim.__dict__,
            "relation_nucleus_terms": (),
        }
    )

    with pytest.raises(
        ValueError,
        match="relation_nucleus_terms",
    ):
        build_closure_retrieval_plan(
            claim
        )
