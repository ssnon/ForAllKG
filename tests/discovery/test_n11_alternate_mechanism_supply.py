from __future__ import annotations

from domains.sers.profile import (
    SERS_AU_AG_PROFILE,
)
from pipeline_core.discovery.nonobviousness_alternate_mechanism_supply import (
    scan_alternate_mechanism_supply,
)
from pipeline_core.discovery.nonobviousness_missing_bridge_contracts import (
    N11MissingBridgeOpportunity,
)
from pipeline_core.discovery.nonobviousness_operator_reeligibility import (
    N11RelativeContributionBranchDecision,
)


def opportunity():
    return N11MissingBridgeOpportunity(
        opportunity_id="n11_missing_bridge:test",
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
            "spacing may alter mechanism"
        ),
        full_relation_text_for_audit=(
            "spacing changes SERS behavior"
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


def d3(
    *,
    status=(
        "ABSTAIN_MISSING_SUPPLEMENTAL_FACTOR_BRIDGE"
    ),
    next_action=(
        "SEARCH_ALTERNATE_SUPPLEMENTAL_MECHANISM_OR_GAP"
    ),
):
    return N11RelativeContributionBranchDecision(
        decision_id="d3:test",
        source_missing_bridge_opportunity_id=(
            "n11_missing_bridge:test"
        ),
        source_d2_search_id="d2:test",
        status=status,
        shared_factor_bridge_attachment_ids=[
            "em:test"
        ],
        supplemental_factor_bridge_attachment_ids=[],
        target_relation_attachment_ids=[],
        electromagnetic_factor_bridge_grounded=True,
        chemical_factor_bridge_grounded=False,
        relative_contribution_relation_grounded=False,
        eligible_for_fresh_regeneration=(
            status
            == "ELIGIBLE_FOR_FRESH_CANDIDATE_REGENERATION"
        ),
        next_action=next_action,
        reason_codes=[
            "test"
        ],
    )


def motif():
    return {
        "node_id": "motif:gap",
        "type": "StructuralMotif",
        "label": "Interparticle nanogap",
        "node_text": "Interparticle nanogap",
        "source_paper_id": "paper:test",
    }


def mechanism_claim(
    text=(
        "Small interparticle nanogaps alter "
        "plasmonic coupling and generate "
        "localized electromagnetic hotspots."
    ),
):
    return {
        "node_id": "claim:mechanism",
        "type": "MechanismClaim",
        "label": text,
        "node_text": text,
        "source_paper_id": "paper:test",
        "requires_verification": False,
    }


def applies_to():
    return {
        "edge_id": "edge:applies",
        "source": "claim:mechanism",
        "relation": "APPLIES_TO",
        "target": "motif:gap",
        "source_paper_id": "paper:test",
        "source_paper_ids": [
            "paper:test"
        ],
        "evidence_pointers": [
            {
                "page": 1
            }
        ],
        "requires_verification": False,
    }


def test_factor_grounded_mechanism_claim_enters_supply():
    result = scan_alternate_mechanism_supply(
        opportunity=opportunity(),
        d3_decision=d3(),
        node_rows=[
            motif(),
            mechanism_claim(),
        ],
        edge_rows=[
            applies_to(),
        ],
        profile=SERS_AU_AG_PROFILE,
    )

    assert (
        result.status
        == "FOUND_FACTOR_GROUNDED_MECHANISM_SUPPLY"
    )

    assert result.supply_candidate_count == 1

    candidate = result.candidates[0]

    assert (
        candidate.factor_connected_grounded_claim
        is True
    )

    assert (
        candidate.distinct_from_baseline_assessed
        is False
    )

    assert (
        candidate.eligible_for_semantic_review
        is True
    )

    assert (
        candidate.eligible_as_positive_hypothesis_premise
        is False
    )


def test_generic_mechanism_attached_to_gap_is_rejected():
    result = scan_alternate_mechanism_supply(
        opportunity=opportunity(),
        d3_decision=d3(),
        node_rows=[
            motif(),
            mechanism_claim(
                text=(
                    "Chemical enhancement may arise "
                    "from charge transfer between "
                    "a molecule and a metal."
                )
            ),
        ],
        edge_rows=[
            applies_to(),
        ],
        profile=SERS_AU_AG_PROFILE,
    )

    assert (
        result.status
        == "ABSTAIN_NO_FACTOR_GROUNDED_MECHANISM_SUPPLY"
    )

    assert (
        result.rejection_reason_counts[
            "claim_does_not_state_factor"
        ]
        == 1
    )


def test_observation_claim_is_not_mechanism_supply():
    row = mechanism_claim()
    row["type"] = "ObservationClaim"

    result = scan_alternate_mechanism_supply(
        opportunity=opportunity(),
        d3_decision=d3(),
        node_rows=[
            motif(),
            row,
        ],
        edge_rows=[
            applies_to(),
        ],
        profile=SERS_AU_AG_PROFILE,
    )

    assert (
        result.status
        == "ABSTAIN_NO_FACTOR_GROUNDED_MECHANISM_SUPPLY"
    )

    assert (
        result.rejection_reason_counts[
            "source_not_mechanism_claim"
        ]
        == 1
    )


def test_bridge_concept_is_not_mechanism_supply():
    row = mechanism_claim()
    row["type"] = "BridgeConcept"

    result = scan_alternate_mechanism_supply(
        opportunity=opportunity(),
        d3_decision=d3(),
        node_rows=[
            motif(),
            row,
        ],
        edge_rows=[
            applies_to(),
        ],
        profile=SERS_AU_AG_PROFILE,
    )

    assert (
        result.status
        == "ABSTAIN_NO_FACTOR_GROUNDED_MECHANISM_SUPPLY"
    )


def test_search_requires_d3_alternate_supply_authority():
    decision = d3(
        status=(
            "DEFER_TARGET_RELATION_ALREADY_GROUNDED"
        ),
        next_action=(
            "REASSESS_TARGET_RELATION_PRIOR_ART"
        ),
    )

    # Make status-consistent fixture.
    decision = (
        decision.model_copy(
            update={
                "relative_contribution_relation_grounded":
                    True,
                "target_relation_attachment_ids":
                    ["target:test"],
            }
        )
    )

    result = scan_alternate_mechanism_supply(
        opportunity=opportunity(),
        d3_decision=decision,
        node_rows=[
            motif(),
            mechanism_claim(),
        ],
        edge_rows=[
            applies_to(),
        ],
        profile=SERS_AU_AG_PROFILE,
    )

    assert (
        result.status
        == "NOT_ELIGIBLE_FROM_D3"
    )

    assert (
        result.reviewed_applies_to_edges
        == 0
    )


def test_composite_claim_does_not_leak_charge_transfer_scope():
    text = (
        "Dense nanogaps between Ag nanoparticles "
        "form hotspots and increase the electric field, "
        "while charge transfer among Ag, ZnO, and RhB "
        "further enhances the SERS signal."
    )

    result = scan_alternate_mechanism_supply(
        opportunity=opportunity(),
        d3_decision=d3(),
        node_rows=[
            motif(),
            mechanism_claim(
                text=text
            ),
        ],
        edge_rows=[
            applies_to(),
        ],
        profile=SERS_AU_AG_PROFILE,
    )

    assert result.supply_candidate_count == 1

    candidate = result.candidates[0]

    assert (
        "charge_transfer"
        not in candidate.mechanism_scope_features
    )

    assert (
        "charge_transfer"
        in candidate.whole_claim_scope_features
    )

    assert any(
        "nanogap" in segment.lower()
        for segment
        in candidate.factor_local_text_segments
    )


def test_explicit_factor_to_charge_transfer_keeps_ct_scope():
    text = (
        "Smaller interparticle nanogaps increase "
        "charge transfer between the metal and "
        "adsorbed molecule, enhancing the SERS signal."
    )

    result = scan_alternate_mechanism_supply(
        opportunity=opportunity(),
        d3_decision=d3(),
        node_rows=[
            motif(),
            mechanism_claim(
                text=text
            ),
        ],
        edge_rows=[
            applies_to(),
        ],
        profile=SERS_AU_AG_PROFILE,
    )

    assert result.supply_candidate_count == 1

    candidate = result.candidates[0]

    assert (
        "charge_transfer"
        in candidate.mechanism_scope_features
    )
