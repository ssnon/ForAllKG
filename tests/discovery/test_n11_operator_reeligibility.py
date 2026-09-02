from __future__ import annotations

import pytest

from pipeline_core.discovery.nonobviousness_grounded_claim_attachment import (
    N11GroundedClaimAttachment,
    N11GroundedClaimAttachmentResult,
)
from pipeline_core.discovery.nonobviousness_missing_bridge_contracts import (
    N11MissingBridgeOpportunity,
)
from pipeline_core.discovery.nonobviousness_operator_reeligibility import (
    decide_relative_contribution_shift_reeligibility,
)


def opportunity(
    opportunity_id: str = "n11_missing_bridge:test",
):
    return N11MissingBridgeOpportunity(
        opportunity_id=opportunity_id,
        source_portfolio_id="portfolio:test",
        source_hypothesis_id="hypothesis:test",
        source_claim_id="claim:test",
        source_execution_plan_id="plan:test",
        factor_identity_terms=[
            "interparticle spacing",
        ],
        base_relation_terms=[
            "SERS response",
            "electromagnetic enhancement",
            "chemical enhancement",
            "relative contribution",
        ],
        bridge_target_text_for_audit=(
            "spacing may alter mechanism balance"
        ),
        full_relation_text_for_audit=(
            "spacing changes relative EM and "
            "chemical contribution"
        ),
        bridge_retrieval_terms_for_audit=[
            "interparticle spacing",
        ],
        established_base_work_ids=[
            "work:base",
        ],
        established_factor_work_ids=[
            "work:factor",
        ],
    )


def attachment(
    *,
    attachment_id: str,
    terms: list[str],
):
    return N11GroundedClaimAttachment(
        attachment_id=attachment_id,
        source_missing_bridge_opportunity_id=(
            "n11_missing_bridge:test"
        ),
        claim_node_id=(
            f"claim:{attachment_id}"
        ),
        factor_node_id="motif:gap",
        claim_node_type="MechanismClaim",
        claim_text=(
            "Nanogap size changes a grounded "
            "scientific mechanism."
        ),
        attachment_edge_id=(
            f"edge:{attachment_id}"
        ),
        matched_factor_features=[
            "nanogap",
        ],
        matched_base_context_terms=terms,
        source_paper_ids=[
            "paper:test",
        ],
        evidence_pointer_count=1,
    )


def result(
    candidates,
    *,
    opportunity_id="n11_missing_bridge:test",
):
    return N11GroundedClaimAttachmentResult(
        search_id="search:test",
        source_missing_bridge_opportunity_id=(
            opportunity_id
        ),
        status=(
            "FOUND_GROUNDED_CLAIM_ATTACHMENTS"
            if candidates
            else
            "ABSTAIN_NO_GROUNDED_FACTOR_RELATION_CLAIM"
        ),
        reviewed_applies_to_edges=10,
        grounded_candidate_count=len(
            candidates
        ),
        candidates=candidates,
        rejection_reason_counts={},
        reason_codes=[
            "test",
        ],
    )


def test_em_without_chemical_abstains():
    decision = (
        decide_relative_contribution_shift_reeligibility(
            opportunity=opportunity(),
            d2_result=result([
                attachment(
                    attachment_id="em",
                    terms=[
                        "electromagnetic enhancement",
                        "SERS response",
                    ],
                )
            ]),
        )
    )

    assert (
        decision.status
        == "ABSTAIN_MISSING_SUPPLEMENTAL_FACTOR_BRIDGE"
    )

    assert (
        decision.electromagnetic_factor_bridge_grounded
        is True
    )

    assert (
        decision.chemical_factor_bridge_grounded
        is False
    )

    assert (
        decision.eligible_for_fresh_regeneration
        is False
    )


def test_em_and_chemical_make_operator_reeligible():
    decision = (
        decide_relative_contribution_shift_reeligibility(
            opportunity=opportunity(),
            d2_result=result([
                attachment(
                    attachment_id="em",
                    terms=[
                        "electromagnetic enhancement",
                    ],
                ),
                attachment(
                    attachment_id="chemical",
                    terms=[
                        "chemical enhancement",
                    ],
                ),
            ]),
        )
    )

    assert (
        decision.status
        == "ELIGIBLE_FOR_FRESH_CANDIDATE_REGENERATION"
    )

    assert (
        decision.eligible_for_fresh_regeneration
        is True
    )

    assert (
        decision.old_candidate_approved
        is False
    )

    assert (
        decision.old_candidate_may_be_reused
        is False
    )


def test_grounded_target_relation_defers_generation():
    decision = (
        decide_relative_contribution_shift_reeligibility(
            opportunity=opportunity(),
            d2_result=result([
                attachment(
                    attachment_id="target",
                    terms=[
                        "relative contribution",
                    ],
                )
            ]),
        )
    )

    assert (
        decision.status
        == "DEFER_TARGET_RELATION_ALREADY_GROUNDED"
    )

    assert (
        decision.next_action
        == "REASSESS_TARGET_RELATION_PRIOR_ART"
    )

    assert (
        decision.eligible_for_fresh_regeneration
        is False
    )


def test_chemical_without_em_is_not_enough():
    decision = (
        decide_relative_contribution_shift_reeligibility(
            opportunity=opportunity(),
            d2_result=result([
                attachment(
                    attachment_id="chemical",
                    terms=[
                        "chemical enhancement",
                    ],
                )
            ]),
        )
    )

    assert (
        decision.status
        == "ABSTAIN_MISSING_SHARED_FACTOR_BRIDGE"
    )


def test_d2_result_must_match_d1_opportunity():
    with pytest.raises(
        ValueError,
        match="does not belong",
    ):
        decide_relative_contribution_shift_reeligibility(
            opportunity=opportunity(),
            d2_result=result(
                [],
                opportunity_id=(
                    "n11_missing_bridge:other"
                ),
            ),
        )
