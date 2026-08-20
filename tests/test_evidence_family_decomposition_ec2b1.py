from __future__ import annotations

from dac_her.evidence_family_decomposition import (
    _child_statement,
)
from pipeline_core.discovery.evidence_family_diagnostics import (
    EvidenceFamilyProfile,
)
from pipeline_core.discovery.explorer_contracts import GraphExplorerPacket
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisEvidenceStatement,
)


def _packet() -> GraphExplorerPacket:
    return GraphExplorerPacket.model_validate(
        {
            "domain_profile_id": "dac_her",
            "packet_id": "packet:test",
            "packet_sha256": "a" * 64,
            "task": {
                "task_id": "task:test",
                "question": "test",
                "traversal_mode": "mechanism",
                "objective": "explain_connection",
            },
            "corpus": {
                "corpus_id": "corpus:test",
                "projection_mode": "mechanism",
                "papers": [],
                "substrate_version": "test",
            },
            "retrieval_summary": {
                "algorithm": "top_n",
                "returned_path_count": 0,
            },
            "direct_concept_hits": [],
            "paths": [],
            "evidence_catalog": {
                "nodes": {
                    "n1": {
                        "node_id": "n1",
                        "node_type": "CoordinationMotif",
                        "label": "motif one",
                        "node_text": "grounded coordination motif one",
                    },
                    "n2": {
                        "node_id": "n2",
                        "node_type": "CoordinationMotif",
                        "label": "motif two",
                        "node_text": "grounded coordination motif two",
                    },
                },
                "edges": {
                    "e1": {
                        "edge_id": "e1",
                        "scientific_source": "n1",
                        "relation": "HAS_MOTIF",
                        "scientific_target": "n2",
                        "source_paper_ids": ["P1", "P2"],
                    }
                },
            },
            "provenance_summary": {
                "strict_provenance": True,
                "edge_count": 1,
                "pointer_grounded_edge_count": 1,
                "pointer_recovered_from_traversal_count": 0,
                "derived_alignment_edge_count": 0,
                "missing_pointer_edge_count": 0,
                "materialized_node_count": 2,
                "suppressed_alignment_member_node_count": 0,
            },
        }
    )


def test_ec2b_child_constructor_matches_strict_hypothesis_contract():
    parent = HypothesisEvidenceStatement(
        statement_id="stmt:parent",
        text="parent synthesis",
        epistemic_role="evidence_synthesis",
        claim_kind="mechanism",
        paper_ids=["P1", "P2"],
        scientific_support_node_ids=["n1", "n2"],
        scientific_support_edge_ids=["e1"],
        support_path_ids=["path:existing"],
        requires_verification=False,
        eligible_as_premise=True,
    )
    family = EvidenceFamilyProfile(
        family_id="family:test",
        paper_ids=["P1", "P2"],
        paper_count=2,
        node_types=["CoordinationMotif"],
        edge_relations=["HAS_MOTIF"],
        direct_support_node_ids=["n1", "n2"],
        direct_support_edge_ids=["e1"],
        direct_support_unit_count=3,
        paper_direct_support_unit_counts={
            "P1": 2,
            "P2": 1,
        },
    )

    child = _child_statement(
        _packet(),
        parent=parent,
        family=family,
    )

    assert isinstance(
        child,
        HypothesisEvidenceStatement,
    )
    assert child.epistemic_role == "evidence_synthesis"
    assert child.claim_kind == "observation"
    assert child.paper_ids == ["P1", "P2"]
    assert child.scientific_support_node_ids == [
        "n1",
        "n2",
    ]
    assert child.scientific_support_edge_ids == [
        "e1",
    ]
    assert child.support_path_ids == []
    assert child.alignment_path_ids == []
    assert child.eligible_as_premise is True

    payload = child.model_dump(mode="json")
    assert "support_node_ids" not in payload
    assert "support_edge_ids" not in payload
    assert "scientific_support_node_ids" in payload
    assert "scientific_support_edge_ids" in payload
