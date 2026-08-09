from __future__ import annotations

import hashlib
import json
from typing import Any

from dac_her.explorer_contracts import ExplorationReport, GraphExplorerPacket
from dac_her.explorer_validation import ExplorationReportValidator
from dac_her.hypothesis_contracts import (
    HypothesisContext,
    HypothesisDesignLeverContext,
    HypothesisEvidenceStatement,
    HypothesisGapContext,
    HypothesisMotifContext,
    HypothesisPolicy,
    HypothesisRouteContext,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(_model_dump(value)).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:length]}"


def _sorted_unique(values: list[str] | set[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value).strip()})


def _is_alignment_node(node: Any) -> bool:
    return (
        str(getattr(node, "graph_layer", "")) == "corpus_alignment"
        or str(getattr(node, "node_type", "")) in {"CorpusAlignment", "CorpusPattern"}
    )


def _is_alignment_edge(edge: Any) -> bool:
    return (
        str(getattr(edge, "graph_layer", "")) == "corpus_alignment"
        or str(getattr(edge, "evidence_status", "")) == "derived_corpus_alignment"
    )


def _path_uses_alignment(path: Any) -> bool:
    return any(
        str(getattr(step, "edge_class", "")) in {"registry_alignment", "pattern_alignment"}
        for step in getattr(path, "steps", [])
    )


def _path_requires_verification(path: Any) -> bool:
    quality = getattr(path, "quality", None)
    candidate_fraction = float(getattr(quality, "candidate_fraction", 0.0) or 0.0)
    return candidate_fraction > 0.0 or any(
        bool(getattr(step, "requires_verification", False))
        for step in getattr(path, "steps", [])
    )


class HypothesisContextBuilder:
    """Build the bounded input surface for Hypothesis Maker v2.6.x.

    The builder consumes the frozen GraphExplorerPacket plus an accepted
    ExplorationReport. It does not generate scientific content. It only recovers
    premise provenance and marks which Explorer statements may be used as positive
    scientific premises versus research gaps.
    """

    def __init__(
        self,
        *,
        validator: ExplorationReportValidator | None = None,
        policy: HypothesisPolicy | None = None,
    ) -> None:
        self.validator = validator or ExplorationReportValidator()
        self.policy = policy or HypothesisPolicy()

    def build(
        self,
        packet: GraphExplorerPacket,
        report: ExplorationReport,
        *,
        require_valid_report: bool = True,
    ) -> HypothesisContext:
        if report.source_packet_sha256 != packet.packet_sha256:
            raise ValueError(
                "ExplorationReport source_packet_sha256 does not match GraphExplorerPacket."
            )

        if require_valid_report:
            validation = self.validator.validate(packet, report)
            if not validation.passes:
                details = "; ".join(
                    f"{issue.code}@{issue.location}: {issue.message}"
                    for issue in validation.issues
                    if issue.severity == "error"
                )
                raise ValueError(f"source ExplorationReport failed validation: {details}")

        nodes = packet.evidence_catalog.nodes
        edges = packet.evidence_catalog.edges
        paths = {path.path_id: path for path in packet.paths}
        hits = {hit.hit_id: hit for hit in packet.direct_concept_hits}
        blocked_papers = {
            str(paper.paper_id)
            for paper in packet.corpus.papers
            if paper.absence_claims_allowed is False
        }

        evidence_statements: list[HypothesisEvidenceStatement] = []
        for statement in report.statements:
            scientific_nodes: set[str] = set()
            scientific_edges: set[str] = set()
            alignment_paths: set[str] = set()
            restrictions: list[str] = []
            candidate_dependent = bool(statement.requires_verification)

            for node_id in statement.support_node_ids:
                node = nodes.get(node_id)
                if node is not None and not _is_alignment_node(node):
                    scientific_nodes.add(str(node_id))
                    candidate_dependent = candidate_dependent or bool(
                        getattr(node, "requires_verification", False)
                    )

            for edge_id in statement.support_edge_ids:
                edge = edges.get(edge_id)
                if edge is not None and not _is_alignment_edge(edge):
                    scientific_edges.add(str(edge_id))
                    candidate_dependent = candidate_dependent or bool(
                        getattr(edge, "requires_verification", False)
                    )

            for hit_id in statement.support_direct_hit_ids:
                hit = hits.get(hit_id)
                if hit is None:
                    continue
                node = nodes.get(hit.node_evidence_ref)
                if node is not None and not _is_alignment_node(node):
                    scientific_nodes.add(str(hit.node_evidence_ref))
                    candidate_dependent = candidate_dependent or bool(
                        getattr(hit, "requires_verification", False)
                    ) or bool(getattr(node, "requires_verification", False))

            for path_id in statement.support_path_ids:
                path = paths.get(path_id)
                if path is None:
                    continue
                if _path_uses_alignment(path):
                    alignment_paths.add(str(path_id))
                candidate_dependent = candidate_dependent or _path_requires_verification(path)
                for step in path.steps:
                    edge = edges.get(step.edge_evidence_ref)
                    if edge is not None and not _is_alignment_edge(edge):
                        scientific_edges.add(str(step.edge_evidence_ref))
                    for node_id in (step.scientific_source, step.scientific_target):
                        node = nodes.get(node_id)
                        if node is not None and not _is_alignment_node(node):
                            scientific_nodes.add(str(node_id))

            if statement.epistemic_role == "navigation_note":
                restrictions.append("navigation_note_not_positive_premise")
            if statement.epistemic_role == "unresolved":
                restrictions.append("unresolved_not_positive_premise")
            if statement.claim_kind == "scope_limit":
                restrictions.append("scope_limit_not_positive_premise")
            if not scientific_nodes and not scientific_edges:
                restrictions.append("missing_scientific_support")
            if alignment_paths:
                restrictions.append("alignment_path_not_scientific_premise")
            if candidate_dependent:
                restrictions.append("candidate_requires_verification")
            if set(map(str, statement.paper_ids)) & blocked_papers:
                restrictions.append("partial_paper_absence_not_allowed")

            blockers = {
                "navigation_note_not_positive_premise",
                "unresolved_not_positive_premise",
                "scope_limit_not_positive_premise",
                "missing_scientific_support",
                "alignment_path_not_scientific_premise",
            }
            eligible_as_premise = (
                statement.epistemic_role in {"reported", "evidence_synthesis"}
                and not (set(restrictions) & blockers)
            )
            eligible_as_gap = statement.epistemic_role == "unresolved"

            evidence_statements.append(
                HypothesisEvidenceStatement(
                    statement_id=statement.statement_id,
                    text=statement.text,
                    epistemic_role=statement.epistemic_role,
                    claim_kind=statement.claim_kind,
                    paper_ids=_sorted_unique(list(map(str, statement.paper_ids))),
                    scientific_support_node_ids=_sorted_unique(scientific_nodes),
                    scientific_support_edge_ids=_sorted_unique(scientific_edges),
                    support_path_ids=_sorted_unique(list(map(str, statement.support_path_ids))),
                    alignment_path_ids=_sorted_unique(alignment_paths),
                    requires_verification=candidate_dependent,
                    eligible_as_premise=eligible_as_premise,
                    eligible_as_gap=eligible_as_gap,
                    premise_restrictions=sorted(set(restrictions)),
                )
            )

        route_contexts = [
            HypothesisRouteContext(
                route_id=route.route_id,
                statement_ids=list(route.statement_ids),
                paper_ids=list(route.paper_ids),
                structural_type=route.structural_type,
                uses_alignment=route.uses_alignment,
                uses_reverse_navigation=route.uses_reverse_navigation,
                navigation_heavy=route.navigation_heavy,
                requires_verification=route.requires_verification,
            )
            for route in report.mechanism_routes
        ]
        motif_contexts = [
            HypothesisMotifContext(
                motif_id=motif.motif_id,
                label=motif.label,
                statement_ids=list(motif.statement_ids),
                paper_ids=list(motif.paper_ids),
                cross_paper=motif.cross_paper,
            )
            for motif in report.recurring_mechanistic_motifs
        ]
        lever_contexts = [
            HypothesisDesignLeverContext(
                lever_id=lever.lever_id,
                label=lever.label,
                statement_ids=list(lever.statement_ids),
                paper_ids=list(lever.paper_ids),
            )
            for lever in report.reported_design_levers
        ]
        gap_contexts = [
            HypothesisGapContext(
                gap_id=gap.gap_id,
                statement_id=gap.statement_id,
                reason=gap.reason,
                related_path_ids=list(gap.related_path_ids),
            )
            for gap in report.unresolved_connections
        ]

        report_sha = _sha256_json(report)
        context_id = _stable_id(
            "hypothesis_context",
            packet.packet_sha256,
            report.report_id,
            report_sha,
        )

        payload = {
            "schema_version": "hypothesis-context-v1",
            "context_id": context_id,
            "source_packet_id": packet.packet_id,
            "source_packet_sha256": packet.packet_sha256,
            "source_report_id": report.report_id,
            "source_report_sha256": report_sha,
            "task_id": report.task_id,
            "question": packet.task.question,
            "corpus_id": packet.corpus.corpus_id,
            "evidence_statements": [x.model_dump(mode="json") for x in evidence_statements],
            "mechanism_routes": [x.model_dump(mode="json") for x in route_contexts],
            "mechanistic_motifs": [x.model_dump(mode="json") for x in motif_contexts],
            "reported_design_levers": [x.model_dump(mode="json") for x in lever_contexts],
            "research_gaps": [x.model_dump(mode="json") for x in gap_contexts],
            "partial_absence_blocked_paper_ids": sorted(blocked_papers),
            "policy": self.policy.model_dump(mode="json"),
        }
        context_sha = _sha256_json(payload)
        return HypothesisContext(
            **payload,
            context_sha256=context_sha,
        )
