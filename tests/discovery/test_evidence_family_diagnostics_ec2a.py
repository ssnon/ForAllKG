from __future__ import annotations

from pipeline_core.discovery.evidence_compression import EvidenceCompressionReport
from pipeline_core.discovery.evidence_family_diagnostics import (
    EvidenceFamilyCandidateAssessor,
)


def _compression() -> EvidenceCompressionReport:
    return EvidenceCompressionReport.model_validate(
        {
            "report_id": "compression:test",
            "report_sha256": "a" * 64,
            "source_packet_id": "packet:test",
            "source_packet_sha256": "b" * 64,
            "source_report_id": "explorer:test",
            "source_report_sha256": "c" * 64,
            "source_context_id": "context:test",
            "source_context_sha256": "d" * 64,
            "domain_profile_id": "dac_her",
            "statement_cards": [
                {
                    "statement_id": "S1",
                    "text": "homogeneous multi-paper",
                    "epistemic_role": "reported",
                    "claim_kind": "observation",
                    "eligible_as_premise": True,
                    "paper_ids": ["P1", "P2"],
                    "paper_count": 2,
                    "multi_paper": True,
                    "scientific_support_node_ids": ["n1", "n2"],
                    "scientific_support_edge_ids": ["e1", "e2"],
                    "support_path_ids": [],
                    "scientific_support_node_count": 2,
                    "scientific_support_edge_count": 2,
                    "support_path_count": 0,
                    "direct_scientific_support_paper_ids": ["P1", "P2"],
                    "paper_contributions": [
                        {
                            "paper_id": "P1",
                            "declared_in_statement": True,
                            "direct_support_node_ids": ["n1"],
                            "direct_support_edge_ids": ["e1"],
                            "node_types": ["ObservationClaim"],
                            "edge_relations": ["SUPPORTED_OBSERVATION"],
                            "direct_support_unit_count": 2,
                            "has_direct_scientific_support": True,
                        },
                        {
                            "paper_id": "P2",
                            "declared_in_statement": True,
                            "direct_support_node_ids": ["n2"],
                            "direct_support_edge_ids": ["e2"],
                            "node_types": ["ObservationClaim"],
                            "edge_relations": ["SUPPORTED_OBSERVATION"],
                            "direct_support_unit_count": 2,
                            "has_direct_scientific_support": True,
                        },
                    ],
                    "distinct_structural_support_profile_count": 1,
                    "all_declared_papers_have_direct_scientific_support": True,
                    "diagnostic_flags": [
                        "multi_paper_reported_statement",
                        "multi_paper_statement",
                    ],
                },
                {
                    "statement_id": "S2",
                    "text": "heterogeneous multi-paper",
                    "epistemic_role": "evidence_synthesis",
                    "claim_kind": "mechanism",
                    "eligible_as_premise": True,
                    "paper_ids": ["P3", "P4", "P5"],
                    "paper_count": 3,
                    "multi_paper": True,
                    "scientific_support_node_ids": ["n3", "n4", "n5"],
                    "scientific_support_edge_ids": ["e3", "e4", "e5"],
                    "support_path_ids": [],
                    "scientific_support_node_count": 3,
                    "scientific_support_edge_count": 3,
                    "support_path_count": 0,
                    "direct_scientific_support_paper_ids": ["P3", "P4", "P5"],
                    "paper_contributions": [
                        {
                            "paper_id": "P3",
                            "declared_in_statement": True,
                            "direct_support_node_ids": ["n3"],
                            "direct_support_edge_ids": ["e3"],
                            "node_types": ["CoordinationMotif"],
                            "edge_relations": ["HAS_MOTIF"],
                            "direct_support_unit_count": 2,
                            "has_direct_scientific_support": True,
                        },
                        {
                            "paper_id": "P4",
                            "declared_in_statement": True,
                            "direct_support_node_ids": ["n4"],
                            "direct_support_edge_ids": ["e4"],
                            "node_types": ["CoordinationMotif"],
                            "edge_relations": ["HAS_MOTIF"],
                            "direct_support_unit_count": 2,
                            "has_direct_scientific_support": True,
                        },
                        {
                            "paper_id": "P5",
                            "declared_in_statement": True,
                            "direct_support_node_ids": ["n5"],
                            "direct_support_edge_ids": ["e5"],
                            "node_types": ["MechanismClaim"],
                            "edge_relations": [
                                "SUPPORTED_MECHANISM_INTERPRETATION"
                            ],
                            "direct_support_unit_count": 2,
                            "has_direct_scientific_support": True,
                        },
                    ],
                    "distinct_structural_support_profile_count": 2,
                    "all_declared_papers_have_direct_scientific_support": True,
                    "diagnostic_flags": [
                        "heterogeneous_structural_support_profiles",
                        "multi_paper_statement",
                        "multi_paper_synthesis_statement",
                    ],
                },
                {
                    "statement_id": "S3",
                    "text": "single-paper",
                    "epistemic_role": "reported",
                    "claim_kind": "mechanism",
                    "eligible_as_premise": True,
                    "paper_ids": ["P6"],
                    "paper_count": 1,
                    "multi_paper": False,
                    "scientific_support_node_ids": ["n6"],
                    "scientific_support_edge_ids": ["e6"],
                    "support_path_ids": ["path:1"],
                    "scientific_support_node_count": 1,
                    "scientific_support_edge_count": 1,
                    "support_path_count": 1,
                    "direct_scientific_support_paper_ids": ["P6"],
                    "paper_contributions": [
                        {
                            "paper_id": "P6",
                            "declared_in_statement": True,
                            "direct_support_node_ids": ["n6"],
                            "direct_support_edge_ids": ["e6"],
                            "support_path_ids": ["path:1"],
                            "node_types": ["MechanismClaim"],
                            "edge_relations": [
                                "SUPPORTED_MECHANISM_INTERPRETATION"
                            ],
                            "path_structure_types": [
                                "CROSS_PAPER_MECHANISTIC"
                            ],
                            "direct_support_unit_count": 2,
                            "has_direct_scientific_support": True,
                        }
                    ],
                    "distinct_structural_support_profile_count": 1,
                    "all_declared_papers_have_direct_scientific_support": True,
                },
            ],
        }
    )


def test_ec2a_only_marks_heterogeneous_complete_multi_paper_statement():
    report = EvidenceFamilyCandidateAssessor().assess(
        _compression()
    )

    assert report.policy.diagnostic_only is True
    assert report.policy.automatic_statement_decomposition_allowed is False
    assert report.decomposition_candidate_count == 1
    assert report.decomposition_candidate_statement_ids == ["S2"]
    assert report.homogeneous_multi_paper_statement_ids == ["S1"]
    assert report.candidate_paper_ids == ["P3", "P4", "P5"]


def test_ec2a_groups_papers_by_exact_structural_support_family():
    report = EvidenceFamilyCandidateAssessor().assess(
        _compression()
    )
    by_id = {
        row.statement_id: row
        for row in report.statement_diagnostics
    }

    candidate = by_id["S2"]
    assert candidate.evidence_family_count == 2

    families = {
        (
            tuple(family.node_types),
            tuple(family.edge_relations),
        ): family.paper_ids
        for family in candidate.evidence_families
    }
    assert families[
        (
            ("CoordinationMotif",),
            ("HAS_MOTIF",),
        )
    ] == ["P3", "P4"]
    assert families[
        (
            ("MechanismClaim",),
            ("SUPPORTED_MECHANISM_INTERPRETATION",),
        )
    ] == ["P5"]


def test_ec2a_tracks_path_lineage_separately_from_decomposition():
    report = EvidenceFamilyCandidateAssessor().assess(
        _compression()
    )

    assert (
        report.eligible_statements_without_explicit_path_lineage_count
        == 2
    )
    assert (
        report.eligible_statements_with_explicit_path_lineage_count
        == 1
    )
    assert set(
        report.eligible_statements_without_explicit_path_lineage_ids
    ) == {"S1", "S2"}

    by_id = {
        row.statement_id: row
        for row in report.statement_diagnostics
    }
    assert by_id["S2"].decomposition_candidate is True
    assert by_id["S2"].has_explicit_path_lineage is False
    assert (
        "eligible_statement_has_no_explicit_support_path_lineage"
        in by_id["S2"].path_lineage_diagnostic_flags
    )


def test_ec2a_incomplete_provenance_blocks_candidate():
    compression = _compression()
    payload = compression.model_dump(mode="json")

    for card in payload["statement_cards"]:
        if card["statement_id"] == "S2":
            card[
                "all_declared_papers_have_direct_scientific_support"
            ] = False
            card[
                "declared_without_direct_scientific_support_paper_ids"
            ] = ["P5"]
            card["diagnostic_flags"].append(
                "declared_paper_without_direct_scientific_support"
            )

    modified = EvidenceCompressionReport.model_validate(
        payload
    )
    report = EvidenceFamilyCandidateAssessor().assess(
        modified
    )
    by_id = {
        row.statement_id: row
        for row in report.statement_diagnostics
    }

    assert by_id["S2"].decomposition_candidate is False
    assert (
        "declared_paper_provenance_incomplete"
        in by_id["S2"].decomposition_blockers
    )
