from __future__ import annotations

from pipeline_core.discovery.explorer_contracts import NodeEvidence
from pipeline_core.discovery.explorer_packet import (
    _relation_pattern_payload,
)


def _accepted_row() -> dict[str, object]:
    return {
        "node_id": "paper::p::bridge::b",
        "type": "BridgeConcept",
        "label": "SERS enhancement varies with molecular orientation",
        "node_text": "SERS enhancement varies with molecular orientation",
        "retention_lane": "accepted_pattern",
        "concept_type": "RelationPattern",
        "pattern_subject": "SERS enhancement",
        "pattern_relation": "VARIES_WITH",
        "pattern_object": "molecular orientation",
        "relation_strength": "correlational",
        "evidence_scope": "paper_result",
        "pattern_support_mode": "explicit_single_span",
        "source_paper_id": "p",
        "chunk_id": "chunk:p",
        "document_id": "document:p",
    }


def test_accepted_relation_pattern_recovers_atomic_semantics():
    payload = _relation_pattern_payload(_accepted_row())

    assert payload is not None
    assert payload["subject"] == "SERS enhancement"
    assert payload["relation"] == "VARIES_WITH"
    assert payload["object"] == "molecular orientation"
    assert payload["paper_id"] == "p"

    node = NodeEvidence(
        node_id="paper::p::bridge::b",
        node_type="BridgeConcept",
        label="SERS enhancement varies with molecular orientation",
        node_text="SERS enhancement varies with molecular orientation",
        source_paper_id="p",
        relation_pattern=payload,
    )

    assert node.relation_pattern is not None
    assert node.relation_pattern.subject == "SERS enhancement"
    assert node.relation_pattern.object == "molecular orientation"


def test_candidate_relation_pattern_does_not_gain_confirmed_authority():
    row = _accepted_row()
    row["retention_lane"] = "semantic_candidate"

    assert _relation_pattern_payload(row) is None


def test_frontier_concept_does_not_gain_relation_pattern_authority():
    row = _accepted_row()
    row["retention_lane"] = "frontier"
    row["concept_type"] = "MechanisticConcept"

    assert _relation_pattern_payload(row) is None


def test_missing_atomic_field_fails_closed_instead_of_parsing_label():
    row = _accepted_row()
    row["pattern_relation"] = ""

    assert _relation_pattern_payload(row) is None


def test_legacy_node_evidence_without_relation_pattern_still_validates():
    node = NodeEvidence(
        node_id="legacy",
        node_type="ObservationClaim",
        label="legacy",
        node_text="legacy",
    )

    assert node.relation_pattern is None
