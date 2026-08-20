from __future__ import annotations

from pipeline_core.discovery.explorer_contracts import (
    CorpusScope,
    EvidenceCatalog,
    ExplorationReport,
    ExplorerStatement,
    ExplorerTask,
    GraphExplorerPacket,
    NodeEvidence,
    PaperScope,
    ProvenanceSummary,
    RetrievalSummary,
    UnresolvedConnection,
)
from dac_her.hypothesis_context import HypothesisContextBuilder


def make_packet_and_report():
    packet = GraphExplorerPacket(
        packet_id="packet:test-v260",
        packet_sha256="packet-sha-v260",
        task=ExplorerTask(
            task_id="task:test-v260",
            question="How could coordination and charge redistribution affect HER?",
            traversal_mode="mechanism",
            objective="explain_connection",
        ),
        corpus=CorpusScope(
            corpus_id="dac_her_test",
            projection_mode="mechanism",
            substrate_version="v2.5.0-h1",
            papers=[
                PaperScope(paper_id="Kiwook_1", quality_status="complete", absence_claims_allowed=True),
                PaperScope(paper_id="Kiwook_2", quality_status="complete", absence_claims_allowed=True),
                PaperScope(
                    paper_id="Kiwook_10",
                    quality_status="partial_acceptable",
                    source_token_coverage=0.97,
                    quarantine_token_fraction=0.03,
                    absence_claims_allowed=False,
                ),
            ],
        ),
        retrieval_summary=RetrievalSummary(algorithm="fixture"),
        direct_concept_hits=[],
        paths=[],
        evidence_catalog=EvidenceCatalog(
            nodes={
                "n:reported": NodeEvidence(
                    node_id="n:reported",
                    node_type="MechanismClaim",
                    label="Coordination modulates hydrogen adsorption energetics.",
                    node_text="Coordination modulates hydrogen adsorption energetics in the reported system.",
                    graph_layer="canonical",
                    evidence_status="source_asserted",
                    source_paper_id="Kiwook_1",
                    source_paper_ids=["Kiwook_1"],
                    absence_claims_allowed=True,
                ),
                "n:candidate": NodeEvidence(
                    node_id="n:candidate",
                    node_type="BridgeConcept",
                    label="Charge redistribution may mediate adsorption changes.",
                    node_text="Unverified semantic candidate linking charge redistribution and adsorption change.",
                    graph_layer="bridge_candidate",
                    evidence_status="semantic_candidate",
                    requires_verification=True,
                    source_paper_id="Kiwook_2",
                    source_paper_ids=["Kiwook_2"],
                    absence_claims_allowed=True,
                ),
                "n:k10": NodeEvidence(
                    node_id="n:k10",
                    node_type="MechanismClaim",
                    label="Axial Co-O coordination is associated with improved activity and charge transfer.",
                    node_text="The authors attribute improved activity and charge transfer to axial Co-O coordination.",
                    graph_layer="canonical",
                    evidence_status="source_asserted",
                    source_paper_id="Kiwook_10",
                    source_paper_ids=["Kiwook_10"],
                    absence_claims_allowed=False,
                ),
            },
            edges={},
        ),
        alignment_contexts=[],
        provenance_summary=ProvenanceSummary(
            strict_provenance=True,
            edge_count=0,
            pointer_grounded_edge_count=0,
            pointer_recovered_from_traversal_count=0,
            derived_alignment_edge_count=0,
            missing_pointer_edge_count=0,
            materialized_node_count=3,
            suppressed_alignment_member_node_count=0,
        ),
    )

    report = ExplorationReport(
        report_id="report:test-v260",
        task_id=packet.task.task_id,
        source_packet_sha256=packet.packet_sha256,
        statements=[
            ExplorerStatement(
                statement_id="s:reported",
                text="The reported mechanism links coordination to hydrogen adsorption energetics.",
                epistemic_role="reported",
                claim_kind="mechanism",
                support_node_ids=["n:reported"],
                paper_ids=["Kiwook_1"],
            ),
            ExplorerStatement(
                statement_id="s:candidate",
                text="The exploratory evidence suggests a tentative charge-redistribution connection that requires verification.",
                epistemic_role="evidence_synthesis",
                claim_kind="association",
                support_node_ids=["n:candidate"],
                paper_ids=["Kiwook_2"],
                requires_verification=True,
            ),
            ExplorerStatement(
                statement_id="s:k10",
                text="Axial Co-O coordination is reported with improved activity and charge transfer in the supplied Kiwook_10 evidence.",
                epistemic_role="reported",
                claim_kind="mechanism",
                support_node_ids=["n:k10"],
                paper_ids=["Kiwook_10"],
            ),
            ExplorerStatement(
                statement_id="s:gap",
                text="The supplied packet does not establish hydrogen-spillover mediation for the axial Co-O system.",
                epistemic_role="unresolved",
                claim_kind="scope_limit",
                support_node_ids=["n:k10"],
                paper_ids=["Kiwook_10"],
            ),
        ],
        direct_findings=["s:reported", "s:k10"],
        mechanism_routes=[],
        recurring_mechanistic_motifs=[],
        cross_paper_connections=[],
        evidence_tensions=[],
        unresolved_connections=[
            UnresolvedConnection(
                gap_id="gap:test",
                statement_id="s:gap",
                reason="partial_source_scope",
            )
        ],
        reported_design_levers=[],
    )
    return packet, report


def make_context():
    packet, report = make_packet_and_report()
    return HypothesisContextBuilder().build(packet, report)
