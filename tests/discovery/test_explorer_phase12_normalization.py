from __future__ import annotations

from pipeline_core.discovery.explorer_contracts import GraphExplorerPacket
from pipeline_core.discovery.explorer_draft import ExplorationDraft
from domains.dac_her.profile import DAC_HER_PROFILE
from pipeline_core.discovery.explorer_normalization import ExplorerDraftNormalizer


def _packet() -> GraphExplorerPacket:
    return GraphExplorerPacket.model_validate(
        {
            "packet_id": "packet:test",
            "packet_sha256": "sha",
            "domain_profile_id": "dac_her",
            "task": {
                "task_id": "task:test",
                "question": "test",
                "traversal_mode": "mechanism",
                "objective": "explain_connection",
            },
            "corpus": {
                "corpus_id": "c",
                "projection_mode": "mechanism",
                "papers": [],
                "substrate_version": "test",
            },
            "retrieval_summary": {"algorithm": "top_n"},
            "direct_concept_hits": [],
            "paths": [],
            "evidence_catalog": {
                "nodes": {
                    "node:obs": {
                        "node_id": "node:obs",
                        "node_type": "Observation",
                        "label": "reported association",
                        "node_text": "reported association",
                    }
                },
                "edges": {},
            },
            "provenance_summary": {
                "strict_provenance": True,
                "edge_count": 0,
                "pointer_grounded_edge_count": 0,
                "pointer_recovered_from_traversal_count": 0,
                "derived_alignment_edge_count": 0,
                "missing_pointer_edge_count": 0,
                "materialized_node_count": 1,
                "suppressed_alignment_member_node_count": 0,
            },
        }
    )


def test_strong_causal_unsupported_mechanism_is_dropped_with_cascade():
    draft = ExplorationDraft.model_validate(
        {
            "statements": [
                {
                    "local_id": "s_bad",
                    "text": "Charge transfer controls HER activity.",
                    "epistemic_role": "reported",
                    "claim_kind": "mechanism",
                    "support_node_ids": ["node:obs"],
                },
                {
                    "local_id": "s_keep",
                    "text": "Charge transfer is associated with HER activity.",
                    "epistemic_role": "reported",
                    "claim_kind": "association",
                    "support_node_ids": ["node:obs"],
                },
            ],
            "direct_finding_local_ids": ["s_bad", "s_keep"],
            "mechanism_routes": [
                {
                    "local_id": "route_drop",
                    "path_ids": ["path:any"],
                    "statement_local_ids": ["s_bad"],
                },
                {
                    "local_id": "route_keep",
                    "path_ids": ["path:any"],
                    "statement_local_ids": ["s_bad", "s_keep"],
                },
            ],
            "recurring_mechanistic_motifs": [
                {
                    "local_id": "motif_drop",
                    "label": "unsupported motif",
                    "statement_local_ids": ["s_bad"],
                }
            ],
            "cross_paper_connections": [
                {
                    "local_id": "conn_keep",
                    "statement_local_ids": ["s_bad", "s_keep"],
                    "path_ids": ["path:any"],
                }
            ],
            "evidence_tensions": [
                {
                    "local_id": "tension_drop",
                    "statement_local_id": "s_bad",
                    "side_a_statement_local_ids": ["s_bad"],
                    "side_b_statement_local_ids": ["s_keep"],
                    "tension_type": "potential_conflict",
                }
            ],
            "unresolved_connections": [
                {
                    "local_id": "gap_drop",
                    "statement_local_id": "s_bad",
                    "reason": "missing_direct_relation_in_packet",
                }
            ],
            "reported_design_levers": [
                {
                    "local_id": "lever_keep",
                    "label": "reported lever",
                    "statement_local_ids": ["s_bad", "s_keep"],
                }
            ],
        }
    )

    result = ExplorerDraftNormalizer(domain_profile=DAC_HER_PROFILE).normalize(_packet(), draft)
    normalized = result.draft

    assert [row.local_id for row in normalized.statements] == ["s_keep"]
    assert normalized.direct_finding_local_ids == ["s_keep"]
    assert [row.local_id for row in normalized.mechanism_routes] == ["route_keep"]
    assert normalized.mechanism_routes[0].statement_local_ids == ["s_keep"]
    assert normalized.recurring_mechanistic_motifs == []
    assert normalized.cross_paper_connections[0].statement_local_ids == ["s_keep"]
    assert normalized.evidence_tensions == []
    assert normalized.unresolved_connections == []
    assert normalized.reported_design_levers[0].statement_local_ids == ["s_keep"]
    assert result.audit.applied is True
    assert any(
        row.action == "drop_unsupported_strong_causal_statement"
        for row in result.audit.actions
    )
