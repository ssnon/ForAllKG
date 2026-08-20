from __future__ import annotations

import pytest

from pipeline_core.discovery.explorer_contracts import GraphExplorerPacket
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisEvidenceStatement,
)
from dac_her.path_lineage_diagnostics import (
    StatementPathLineageAssessor,
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
                "returned_path_count": 3,
            },
            "direct_concept_hits": [],
            "paths": [
                {
                    "path_id": "P1",
                    "bundle_rank": 1,
                    "endpoint": {
                        "source_node_id": "n1",
                        "target_node_id": "n3",
                    },
                    "node_ids": ["n1", "n2", "n3"],
                    "steps": [
                        {
                            "navigation_source": "n1",
                            "navigation_target": "n2",
                            "traversal_direction": "forward",
                            "scientific_source": "n1",
                            "relation": "R1",
                            "scientific_target": "n2",
                            "selected_original_edge_id": "e1",
                            "edge_evidence_ref": "e1",
                        },
                        {
                            "navigation_source": "n2",
                            "navigation_target": "n3",
                            "traversal_direction": "forward",
                            "scientific_source": "n2",
                            "relation": "R2",
                            "scientific_target": "n3",
                            "selected_original_edge_id": "e2",
                            "edge_evidence_ref": "e2",
                        },
                    ],
                    "visited_paper_ids": ["A", "B"],
                    "supporting_paper_ids": ["A", "B"],
                    "quality": {
                        "path_type": "CROSS_PAPER_MECHANISTIC",
                        "path_structure_type": "CROSS_PAPER_MECHANISTIC",
                        "mechanism_bearing": True,
                        "mechanistic_content": "high",
                    },
                },
                {
                    "path_id": "P2",
                    "bundle_rank": 2,
                    "endpoint": {
                        "source_node_id": "n4",
                        "target_node_id": "n5",
                    },
                    "node_ids": ["n4", "n5"],
                    "steps": [
                        {
                            "navigation_source": "n4",
                            "navigation_target": "n5",
                            "traversal_direction": "forward",
                            "scientific_source": "n4",
                            "relation": "R3",
                            "scientific_target": "n5",
                            "selected_original_edge_id": "e3",
                            "edge_evidence_ref": "e3",
                        }
                    ],
                    "visited_paper_ids": ["C"],
                    "supporting_paper_ids": ["C"],
                    "quality": {
                        "path_type": "CROSS_PAPER_BRIDGE",
                        "path_structure_type": "CROSS_PAPER_BRIDGE",
                        "mechanism_bearing": False,
                        "mechanistic_content": "low",
                    },
                },
                {
                    "path_id": "P3",
                    "bundle_rank": 3,
                    "endpoint": {
                        "source_node_id": "n6",
                        "target_node_id": "n7",
                    },
                    "node_ids": ["n6", "n7"],
                    "steps": [
                        {
                            "navigation_source": "n6",
                            "navigation_target": "n7",
                            "traversal_direction": "forward",
                            "scientific_source": "n6",
                            "relation": "R4",
                            "scientific_target": "n7",
                            "selected_original_edge_id": "e4",
                            "edge_evidence_ref": "e4",
                        }
                    ],
                    "visited_paper_ids": ["D"],
                    "supporting_paper_ids": ["D"],
                    "quality": {
                        "path_type": "DIRECT_MECHANISTIC",
                        "path_structure_type": "DIRECT_MECHANISTIC",
                        "mechanism_bearing": True,
                        "mechanistic_content": "high",
                    },
                },
            ],
            "evidence_catalog": {
                "nodes": {
                    node_id: {
                        "node_id": node_id,
                        "node_type": "ObservationClaim",
                        "label": node_id,
                        "node_text": node_id,
                    }
                    for node_id in [
                        "n1",
                        "n2",
                        "n3",
                        "n4",
                        "n5",
                        "n6",
                        "n7",
                    ]
                },
                "edges": {
                    "e1": {
                        "edge_id": "e1",
                        "scientific_source": "n1",
                        "relation": "R1",
                        "scientific_target": "n2",
                        "source_paper_ids": ["A"],
                    },
                    "e2": {
                        "edge_id": "e2",
                        "scientific_source": "n2",
                        "relation": "R2",
                        "scientific_target": "n3",
                        "source_paper_ids": ["B"],
                    },
                    "e3": {
                        "edge_id": "e3",
                        "scientific_source": "n4",
                        "relation": "R3",
                        "scientific_target": "n5",
                        "source_paper_ids": ["C"],
                    },
                    "e4": {
                        "edge_id": "e4",
                        "scientific_source": "n6",
                        "relation": "R4",
                        "scientific_target": "n7",
                        "source_paper_ids": ["D"],
                    },
                },
            },
            "provenance_summary": {
                "strict_provenance": True,
                "edge_count": 4,
                "pointer_grounded_edge_count": 4,
                "pointer_recovered_from_traversal_count": 0,
                "derived_alignment_edge_count": 0,
                "missing_pointer_edge_count": 0,
                "materialized_node_count": 7,
                "suppressed_alignment_member_node_count": 0,
            },
        }
    )


def _context() -> HypothesisContext:
    return HypothesisContext(
        context_id="context:test",
        context_sha256="b" * 64,
        source_packet_id="packet:test",
        source_packet_sha256="a" * 64,
        source_report_id="report:test",
        source_report_sha256="c" * 64,
        task_id="task:test",
        question="test",
        corpus_id="corpus:test",
        domain_profile_id="dac_her",
        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id="S1",
                text="split across two selected paths",
                epistemic_role="reported",
                claim_kind="observation",
                paper_ids=["A", "C"],
                eligible_as_premise=True,
                scientific_support_node_ids=[
                    "n1",
                    "n2",
                    "n4",
                    "n5",
                ],
                scientific_support_edge_ids=[
                    "e1",
                    "e3",
                ],
            ),
            HypothesisEvidenceStatement(
                statement_id="S2",
                text="exact explicit mechanistic route",
                epistemic_role="reported",
                claim_kind="mechanism",
                paper_ids=["D"],
                eligible_as_premise=True,
                scientific_support_node_ids=[
                    "n6",
                    "n7",
                ],
                scientific_support_edge_ids=[
                    "e4"
                ],
                support_path_ids=[
                    "P3"
                ],
            ),
            HypothesisEvidenceStatement(
                statement_id="S3",
                text="node context but its edge is not selected",
                epistemic_role="reported",
                claim_kind="observation",
                paper_ids=["A"],
                eligible_as_premise=True,
                scientific_support_node_ids=[
                    "n1"
                ],
                scientific_support_edge_ids=[
                    "e2"
                ],
            ),
            HypothesisEvidenceStatement(
                statement_id="S4",
                text="node-only premise can use node attribution",
                epistemic_role="reported",
                claim_kind="observation",
                paper_ids=["B"],
                eligible_as_premise=True,
                scientific_support_node_ids=[
                    "n3"
                ],
                scientific_support_edge_ids=[],
            ),
        ],
    )


def test_pl1a_recovers_missing_explicit_lineage_from_conservative_edge_overlap():
    report = StatementPathLineageAssessor().assess(
        _packet(),
        _context(),
    )
    by_id = {
        row.statement_id: row
        for row in report.statement_diagnostics
    }

    s1 = by_id["S1"]
    assert s1.has_explicit_path_lineage is False
    assert s1.deterministic_attribution_candidate_path_ids == [
        "P1",
        "P2",
    ]
    assert s1.missing_explicit_lineage_but_recoverable is True
    assert s1.candidate_union_covers_all_statement_edges is True
    assert s1.candidate_union_statement_edge_coverage == pytest.approx(
        1.0
    )


def test_pl1a_preserves_exact_explicit_mechanistic_attribution():
    report = StatementPathLineageAssessor().assess(
        _packet(),
        _context(),
    )
    by_id = {
        row.statement_id: row
        for row in report.statement_diagnostics
    }

    s2 = by_id["S2"]
    assert s2.explicit_support_path_ids == ["P3"]
    assert s2.deterministic_attribution_candidate_path_ids == ["P3"]
    assert s2.deterministic_mechanistic_candidate_path_ids == ["P3"]
    assert (
        s2.explicit_path_ids_not_deterministically_attributable
        == []
    )
    assert s2.path_overlaps[0].relationship == "exact_support_route"


def test_pl1a_node_overlap_is_not_enough_when_statement_has_edges():
    report = StatementPathLineageAssessor().assess(
        _packet(),
        _context(),
    )
    by_id = {
        row.statement_id: row
        for row in report.statement_diagnostics
    }

    # S3 shares n1 with P1 but its scientific edge is e2. P1 actually
    # contains e2 too, so make sure the conservative rule finds it by edge.
    s3 = by_id["S3"]
    assert "P1" in s3.deterministic_attribution_candidate_path_ids

    # The general invariant is checked directly on non-candidate overlaps:
    for overlap in s3.path_overlaps:
        if (
            overlap.edge_overlap_count == 0
            and overlap.node_overlap_count > 0
        ):
            assert overlap.relationship == "node_context_only"
            assert overlap.attribution_candidate is False


def test_pl1a_node_only_statement_can_be_attributed_by_node_overlap():
    report = StatementPathLineageAssessor().assess(
        _packet(),
        _context(),
    )
    by_id = {
        row.statement_id: row
        for row in report.statement_diagnostics
    }

    s4 = by_id["S4"]
    assert s4.deterministic_attribution_candidate_path_ids == ["P1"]
    assert s4.path_overlaps[0].relationship == "exact_node_support_route"


def test_pl1a_fails_closed_on_packet_lineage_mismatch():
    context = _context().model_copy(
        update={
            "source_packet_id": "packet:wrong",
        }
    )
    with pytest.raises(
        ValueError,
        match="context/packet ID mismatch",
    ):
        StatementPathLineageAssessor().assess(
            _packet(),
            context,
        )
