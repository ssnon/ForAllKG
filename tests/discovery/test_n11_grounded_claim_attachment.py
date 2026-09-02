from __future__ import annotations

from domains.sers.profile import (
    SERS_AU_AG_PROFILE,
)
from pipeline_core.discovery.nonobviousness_grounded_claim_attachment import (
    scan_grounded_claim_attachments,
)
from pipeline_core.discovery.nonobviousness_missing_bridge_contracts import (
    N11MissingBridgeOpportunity,
)


def opportunity() -> N11MissingBridgeOpportunity:
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
            "SERS response",
        ],
        established_base_work_ids=[
            "work:base",
        ],
        established_factor_work_ids=[
            "work:factor",
        ],
    )


def motif():
    return {
        "node_id": "motif:gap",
        "type": "StructuralMotif",
        "label": "Interior nanogap",
        "node_text": "Interior nanogap",
        "source_paper_id": "paper:test",
    }


def claim(
    *,
    node_id="claim:test",
    node_type="ObservationClaim",
    text=(
        "Increasing the interior gap size "
        "decreases electromagnetic enhancement."
    ),
):
    return {
        "node_id": node_id,
        "type": node_type,
        "label": text,
        "node_text": text,
        "source_paper_id": "paper:test",
        "requires_verification": False,
    }


def applies_to(
    *,
    source="claim:test",
    target="motif:gap",
    pointers=True,
):
    return {
        "edge_id": "edge:test",
        "source": source,
        "relation": "APPLIES_TO",
        "target": target,
        "source_paper_id": "paper:test",
        "source_paper_ids_json": '["paper:test"]',
        "evidence_pointers_json": (
            '[{"page": 1}]'
            if pointers
            else "[]"
        ),
        "requires_verification": False,
    }


def test_grounded_observation_claim_attachment_is_found():
    result = scan_grounded_claim_attachments(
        opportunity=opportunity(),
        node_rows=[
            motif(),
            claim(),
        ],
        edge_rows=[
            applies_to(),
        ],
        profile=SERS_AU_AG_PROFILE,
    )

    assert (
        result.status
        == "FOUND_GROUNDED_CLAIM_ATTACHMENTS"
    )

    assert result.grounded_candidate_count == 1

    candidate = result.candidates[0]

    assert (
        candidate.path_class
        == "GROUNDED_CLAIM_ATTACHMENT"
    )

    assert (
        "electromagnetic enhancement"
        in candidate.matched_base_context_terms
    )

    assert (
        "nanogap"
        in candidate.matched_factor_features
    )

    assert (
        candidate.production_authority
        is False
    )


def test_smaller_gap_stronger_sers_is_relation_bearing():
    text = (
        "Double-shelled Au/Ag nanoboxes with "
        "smaller interior gap sizes exhibit "
        "stronger SERS intensity."
    )

    result = scan_grounded_claim_attachments(
        opportunity=opportunity(),
        node_rows=[
            motif(),
            claim(
                text=text
            ),
        ],
        edge_rows=[
            applies_to(),
        ],
        profile=SERS_AU_AG_PROFILE,
    )

    assert result.grounded_candidate_count == 1

    assert (
        "SERS response"
        in result
        .candidates[0]
        .matched_base_context_terms
    )


def test_bridge_concept_is_not_positive_claim_evidence():
    result = scan_grounded_claim_attachments(
        opportunity=opportunity(),
        node_rows=[
            motif(),
            claim(
                node_type="BridgeConcept",
                text=(
                    "SERS enhancement varies "
                    "with nanogap width"
                ),
            ),
        ],
        edge_rows=[
            applies_to(),
        ],
        profile=SERS_AU_AG_PROFILE,
    )

    assert (
        result.status
        == "ABSTAIN_NO_GROUNDED_FACTOR_RELATION_CLAIM"
    )

    assert (
        result.rejection_reason_counts[
            "source_not_reported_claim"
        ]
        == 1
    )


def test_common_anchor_structure_is_not_claim_attachment():
    substrate = {
        "node_id": "substrate:test",
        "type": "PlasmonicSubstrate",
        "label": "Gold nanostructure",
        "node_text": "Gold nanostructure",
        "source_paper_id": "paper:test",
    }

    observation = claim(
        text=(
            "The substrate shows strong "
            "SERS enhancement."
        ),
    )

    result = scan_grounded_claim_attachments(
        opportunity=opportunity(),
        node_rows=[
            motif(),
            substrate,
            observation,
        ],
        edge_rows=[
            {
                "edge_id": "edge:motif",
                "source": "substrate:test",
                "relation": "HAS_STRUCTURAL_MOTIF",
                "target": "motif:gap",
                "source_paper_id": "paper:test",
                "evidence_pointers_json": (
                    '[{"page": 1}]'
                ),
            },
            {
                "edge_id": "edge:obs",
                "source": "substrate:test",
                "relation": "SUPPORTED_OBSERVATION",
                "target": "claim:test",
                "source_paper_id": "paper:test",
                "evidence_pointers_json": (
                    '[{"page": 1}]'
                ),
            },
        ],
        profile=SERS_AU_AG_PROFILE,
    )

    assert (
        result.status
        == "ABSTAIN_NO_GROUNDED_FACTOR_RELATION_CLAIM"
    )


def test_claim_without_relation_language_is_rejected():
    text = (
        "The calculated electromagnetic "
        "enhancement is localized within "
        "the interior nanogap."
    )

    result = scan_grounded_claim_attachments(
        opportunity=opportunity(),
        node_rows=[
            motif(),
            claim(
                text=text
            ),
        ],
        edge_rows=[
            applies_to(),
        ],
        profile=SERS_AU_AG_PROFILE,
    )

    assert (
        result.status
        == "ABSTAIN_NO_GROUNDED_FACTOR_RELATION_CLAIM"
    )

    assert (
        result.rejection_reason_counts[
            "claim_lacks_relation_language"
        ]
        == 1
    )


def test_missing_pointer_is_rejected():
    result = scan_grounded_claim_attachments(
        opportunity=opportunity(),
        node_rows=[
            motif(),
            claim(),
        ],
        edge_rows=[
            applies_to(
                pointers=False
            ),
        ],
        profile=SERS_AU_AG_PROFILE,
    )

    assert (
        result.status
        == "ABSTAIN_NO_GROUNDED_FACTOR_RELATION_CLAIM"
    )

    assert (
        result.rejection_reason_counts[
            "attachment_missing_pointer"
        ]
        == 1
    )


def test_materialized_pointer_and_paper_lists_are_supported():
    edge = applies_to()

    edge.pop(
        "evidence_pointers_json"
    )
    edge.pop(
        "source_paper_ids_json"
    )

    edge["evidence_pointers"] = [
        {
            "page": 1,
            "quote": "gap-dependent trend",
        }
    ]

    edge["source_paper_ids"] = [
        "paper:test"
    ]

    result = scan_grounded_claim_attachments(
        opportunity=opportunity(),
        node_rows=[
            motif(),
            claim(),
        ],
        edge_rows=[
            edge,
        ],
        profile=SERS_AU_AG_PROFILE,
    )

    assert (
        result.status
        == "FOUND_GROUNDED_CLAIM_ATTACHMENTS"
    )
    assert (
        result.grounded_candidate_count
        == 1
    )
    assert (
        result.candidates[0]
        .evidence_pointer_count
        == 1
    )
    assert (
        result.candidates[0]
        .source_paper_ids
        == ["paper:test"]
    )


def test_generic_chemical_claim_attached_to_gap_is_not_factor_relation():
    text = (
        "Chemical enhancement can arise from "
        "charge-transfer or resonance effects "
        "between an adsorbed molecule and a "
        "metallic nanostructure."
    )

    result = scan_grounded_claim_attachments(
        opportunity=opportunity(),
        node_rows=[
            motif(),
            claim(
                node_type="MechanismClaim",
                text=text,
            ),
        ],
        edge_rows=[
            applies_to(),
        ],
        profile=SERS_AU_AG_PROFILE,
    )

    assert (
        result.status
        == "ABSTAIN_NO_GROUNDED_FACTOR_RELATION_CLAIM"
    )

    assert (
        result.rejection_reason_counts[
            "claim_does_not_state_factor"
        ]
        == 1
    )
