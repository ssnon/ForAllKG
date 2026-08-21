from __future__ import annotations

import pytest

from domains.registry import get_domain_profile
from pipeline_core.discovery.explorer_validation import ExplorationReportValidator
from pipeline_core.discovery.evidence_compression import (
    EvidenceCompressionAssessor,
)
from pipeline_core.discovery.explorer_contracts import (
    ExplorationReport,
    GraphExplorerPacket,
)
from pipeline_core.discovery.hypothesis_context import HypothesisContextBuilder


def _packet() -> GraphExplorerPacket:
    return GraphExplorerPacket.model_validate(
        {
            "domain_profile_id": "dac_her",
            "packet_id": "packet:test",
            "packet_sha256": "a" * 64,
            "task": {
                "task_id": "task:test",
                "question": "test question",
                "traversal_mode": "mechanism",
                "objective": "explain_connection",
            },
            "corpus": {
                "corpus_id": "corpus:test",
                "projection_mode": "mechanism",
                "papers": [
                    {
                        "paper_id": paper_id,
                        "quality_status": "complete",
                        "absence_claims_allowed": True,
                    }
                    for paper_id in [
                        "P1",
                        "P2",
                        "P3",
                        "P4",
                        "P5",
                        "P6",
                    ]
                ],
                "substrate_version": "test",
            },
            "retrieval_summary": {
                "algorithm": "top_n",
                "returned_path_count": 1,
            },
            "direct_concept_hits": [],
            "paths": [
                {
                    "path_id": "path:all",
                    "bundle_rank": 1,
                    "endpoint": {
                        "source_node_id": "n1",
                        "target_node_id": "n5",
                    },
                    "node_ids": [],
                    "steps": [],
                    "visited_paper_ids": [
                        "P1",
                        "P2",
                        "P3",
                        "P4",
                        "P5",
                        "P6",
                    ],
                    "supporting_paper_ids": [
                        "P1",
                        "P2",
                        "P3",
                        "P4",
                        "P5",
                        "P6",
                    ],
                    "quality": {},
                }
            ],
            "evidence_catalog": {
                "nodes": {
                    "n1": {
                        "node_id": "n1",
                        "node_type": "Observation",
                        "label": "obs1",
                        "node_text": "observation one",
                        "source_paper_id": "P1",
                    },
                    "n2": {
                        "node_id": "n2",
                        "node_type": "Observation",
                        "label": "obs2",
                        "node_text": "observation two",
                        "source_paper_id": "P2",
                    },
                    "n3": {
                        "node_id": "n3",
                        "node_type": "Mechanism",
                        "label": "mech3",
                        "node_text": "mechanism three",
                        "source_paper_id": "P3",
                    },
                    "n4": {
                        "node_id": "n4",
                        "node_type": "Mechanism",
                        "label": "mech4",
                        "node_text": "mechanism four",
                        "source_paper_id": "P4",
                    },
                    "n5": {
                        "node_id": "n5",
                        "node_type": "Observation",
                        "label": "obs5",
                        "node_text": "observation five",
                        "source_paper_id": "P5",
                    },
                },
                "edges": {
                    "e1": {
                        "edge_id": "e1",
                        "scientific_source": "n1",
                        "relation": "REL",
                        "scientific_target": "n2",
                        "source_paper_ids": ["P1", "P2"],
                    },
                    "e2": {
                        "edge_id": "e2",
                        "scientific_source": "n4",
                        "relation": "MODULATES",
                        "scientific_target": "n5",
                        "source_paper_ids": ["P4"],
                    },
                },
            },
            "provenance_summary": {
                "strict_provenance": True,
                "edge_count": 2,
                "pointer_grounded_edge_count": 2,
                "pointer_recovered_from_traversal_count": 0,
                "derived_alignment_edge_count": 0,
                "missing_pointer_edge_count": 0,
                "materialized_node_count": 5,
                "suppressed_alignment_member_node_count": 0,
            },
        }
    )


def _report() -> ExplorationReport:
    return ExplorationReport.model_validate(
        {
            "report_id": "report:test",
            "task_id": "task:test",
            "source_packet_sha256": "a" * 64,
            "statements": [
                {
                    "statement_id": "S1",
                    "text": "P1 and P2 support one synthesized observation.",
                    "epistemic_role": "evidence_synthesis",
                    "claim_kind": "observation",
                    "support_node_ids": ["n1", "n2"],
                    "support_edge_ids": ["e1"],
                    "paper_ids": ["P1", "P2"],
                },
                {
                    "statement_id": "S2",
                    "text": "P3 reports one mechanism.",
                    "epistemic_role": "reported",
                    "claim_kind": "mechanism",
                    "support_node_ids": ["n3"],
                    "paper_ids": ["P3"],
                },
                {
                    "statement_id": "S3",
                    "text": "P4 and P5 are grouped despite structurally different support.",
                    "epistemic_role": "reported",
                    "claim_kind": "mechanism",
                    "support_node_ids": ["n4", "n5"],
                    "support_edge_ids": ["e2"],
                    "paper_ids": ["P4", "P5"],
                },
            ],
        }
    )


def _context(packet, report):
    validator = ExplorationReportValidator(
        semantics=get_domain_profile(
            packet.domain_profile_id
        ).discovery,
    )
    return HypothesisContextBuilder(
        validator=validator,
    ).build(
        packet,
        report,
        require_valid_report=False,
    )


def test_ec1_measures_paper_to_statement_compression_without_changing_science():
    packet = _packet()
    report = _report()
    context = _context(packet, report)

    result = EvidenceCompressionAssessor().assess(
        packet,
        report,
        context,
    )

    assert result.policy.diagnostic_only is True
    assert result.policy.scientific_selection_changed is False

    assert result.selected_path_paper_count == 6
    assert result.explorer_statement_paper_count == 5
    assert result.eligible_premise_paper_count == 5

    assert result.eligible_statement_count == 3
    assert result.eligible_single_paper_statement_count == 1
    assert result.eligible_multi_paper_statement_count == 2
    assert result.eligible_multi_paper_statement_fraction == pytest.approx(
        2 / 3
    )
    assert result.mean_papers_per_eligible_statement == pytest.approx(
        5 / 3
    )

    assert result.eligible_papers_with_single_paper_statement_ids == [
        "P3"
    ]
    assert result.eligible_papers_only_in_multi_paper_statements_ids == [
        "P1",
        "P2",
        "P4",
        "P5",
    ]
    assert (
        result.eligible_papers_only_in_multi_paper_statements_fraction
        == pytest.approx(4 / 5)
    )

    assert result.eligible_multi_paper_synthesis_statement_count == 1
    assert result.eligible_multi_paper_reported_statement_count == 1


def test_ec1_exposes_structural_profile_heterogeneity_and_incidence_groups():
    packet = _packet()
    report = _report()
    context = _context(packet, report)

    result = EvidenceCompressionAssessor().assess(
        packet,
        report,
        context,
    )
    by_id = {
        row.statement_id: row
        for row in result.statement_cards
    }

    # S1 has two papers with the same node-type profile plus a shared edge.
    assert by_id["S1"].multi_paper is True
    assert by_id["S1"].all_declared_papers_have_direct_scientific_support is True

    # S3 deliberately mixes Mechanism and Observation support profiles.
    assert by_id["S3"].distinct_structural_support_profile_count >= 2
    assert (
        "heterogeneous_structural_support_profiles"
        in by_id["S3"].diagnostic_flags
    )
    assert result.eligible_multi_paper_heterogeneous_profile_count >= 1

    groups = {
        tuple(row.eligible_statement_ids): row.paper_ids
        for row in result.repeated_paper_incidence_groups
    }
    assert groups[("S1",)] == ["P1", "P2"]
    assert groups[("S3",)] == ["P4", "P5"]


def test_ec1_fails_closed_on_lineage_mismatch():
    packet = _packet()
    report = _report()
    context = _context(packet, report).model_copy(
        update={
            "source_packet_id": "packet:wrong",
        }
    )

    with pytest.raises(
        ValueError,
        match="context/packet ID mismatch",
    ):
        EvidenceCompressionAssessor().assess(
            packet,
            report,
            context,
        )
