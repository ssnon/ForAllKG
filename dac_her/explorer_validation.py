from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict

from pipeline_core.discovery_semantics import (
    contains_strong_causal_language,
    edge_has_strong_causal_semantics,
    is_alignment_edge,
    is_alignment_node,
    is_mechanism_edge,
    is_mechanism_node,
)
from pipeline_core.domain_profile import DiscoverySemantics
from dac_her.domains import get_domain_profile
from pipeline_core.explorer_text_safety import contains_absence_language
from pipeline_core.discovery.explorer_contracts import (
    ExplorationReport,
    ExplorerStatement,
    GraphExplorerPacket,
)



_HYPOTHESIS_PATTERNS = (
    re.compile(r"\bwe propose\b", re.I),
    re.compile(r"\bwe hypothesi[sz]e\b", re.I),
    re.compile(r"\bour hypothesis\b", re.I),
    re.compile(r"\bnovel catalyst\b", re.I),
)

_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:\s*(?:eV|meV|V|mV|A|mA|K|°C|C|%|nm|Å|pm|cm|mm|s|ms|h|mol|M|pH))?", re.I)





class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Literal["error", "warning"]
    code: str
    location: str
    message: str


class ExplorationValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passes: bool
    errors: int
    warnings: int
    issues: list[ValidationIssue]


def _numbers(text: str) -> set[str]:
    return {" ".join(match.group(0).split()).lower() for match in _NUMBER_RE.finditer(text)}


def _contains_hypothesis_language(text: str) -> bool:
    return any(pattern.search(text) for pattern in _HYPOTHESIS_PATTERNS)


def _label_has_grounded_causal_support(
    *,
    semantics: DiscoverySemantics,
    statement_ids: Iterable[str],
    node_ids: Iterable[str],
    edge_ids: Iterable[str],
    statements: dict[str, ExplorerStatement],
    nodes: dict[str, Any],
    edges: dict[str, Any],
    hits: dict[str, Any],
) -> bool:
    candidate_node_ids = {str(x) for x in node_ids}
    candidate_edge_ids = {str(x) for x in edge_ids}
    candidate_statement_texts: list[str] = []
    for statement_id in statement_ids:
        statement = statements.get(statement_id)
        if statement is None:
            continue
        candidate_statement_texts.append(statement.text)
        candidate_node_ids.update(statement.support_node_ids)
        candidate_edge_ids.update(statement.support_edge_ids)
        for hit_id in statement.support_direct_hit_ids:
            hit = hits.get(hit_id)
            if hit is not None:
                candidate_node_ids.add(hit.node_evidence_ref)
    if any(contains_strong_causal_language(text, semantics) for text in candidate_statement_texts):
        return True
    for node_id in candidate_node_ids:
        node = nodes.get(node_id)
        if node is None or is_alignment_node(node):
            continue
        text = f"{getattr(node, 'label', '')}\n{getattr(node, 'node_text', '')}"
        if contains_strong_causal_language(text, semantics):
            return True
    for edge_id in candidate_edge_ids:
        edge = edges.get(edge_id)
        if edge is None or is_alignment_edge(edge):
            continue
        if edge_has_strong_causal_semantics(edge, semantics):
            return True
    return False


class ExplorationReportValidator:
    def validate(
        self,
        packet: GraphExplorerPacket,
        report: ExplorationReport,
    ) -> ExplorationValidationResult:
        issues: list[ValidationIssue] = []
        semantics = get_domain_profile(
            packet.domain_profile_id
        ).discovery

        def error(code: str, location: str, message: str) -> None:
            issues.append(ValidationIssue(severity="error", code=code, location=location, message=message))

        def warning(code: str, location: str, message: str) -> None:
            issues.append(ValidationIssue(severity="warning", code=code, location=location, message=message))

        if report.task_id != packet.task.task_id:
            error("TASK_ID_MISMATCH", "report.task_id", "Report task_id does not match the packet task_id.")
        if report.source_packet_sha256 != packet.packet_sha256:
            error("PACKET_SHA_MISMATCH", "report.source_packet_sha256", "Report source_packet_sha256 does not match the packet.")

        nodes = packet.evidence_catalog.nodes
        edges = packet.evidence_catalog.edges
        paths = {path.path_id: path for path in packet.paths}
        hits = {hit.hit_id: hit for hit in packet.direct_concept_hits}
        papers = {paper.paper_id: paper for paper in packet.corpus.papers}
        statements = {statement.statement_id: statement for statement in report.statements}

        if len(statements) != len(report.statements):
            error("DUPLICATE_STATEMENT_ID", "report.statements", "Statement IDs must be unique.")

        referenced_section_ids: list[tuple[str, str]] = []
        referenced_section_ids.extend((f"direct_findings[{i}]", value) for i, value in enumerate(report.direct_findings))
        for i, route in enumerate(report.mechanism_routes):
            referenced_section_ids.extend((f"mechanism_routes[{i}].statement_ids", value) for value in route.statement_ids)
        for i, motif in enumerate(report.recurring_mechanistic_motifs):
            referenced_section_ids.extend((f"recurring_mechanistic_motifs[{i}].statement_ids", value) for value in motif.statement_ids)
        for i, connection in enumerate(report.cross_paper_connections):
            referenced_section_ids.extend((f"cross_paper_connections[{i}].statement_ids", value) for value in connection.statement_ids)
        for i, tension in enumerate(report.evidence_tensions):
            referenced_section_ids.append((f"evidence_tensions[{i}].statement_id", tension.statement_id))
            referenced_section_ids.extend((f"evidence_tensions[{i}].side_a_statement_ids", value) for value in tension.side_a_statement_ids)
            referenced_section_ids.extend((f"evidence_tensions[{i}].side_b_statement_ids", value) for value in tension.side_b_statement_ids)
        for i, unresolved in enumerate(report.unresolved_connections):
            referenced_section_ids.append((f"unresolved_connections[{i}].statement_id", unresolved.statement_id))
        for i, lever in enumerate(report.reported_design_levers):
            referenced_section_ids.extend((f"reported_design_levers[{i}].statement_ids", value) for value in lever.statement_ids)
        for location, statement_id in referenced_section_ids:
            if statement_id not in statements:
                error("UNKNOWN_STATEMENT_REF", location, f"Unknown statement ID: {statement_id}")

        for index, statement in enumerate(report.statements):
            location = f"statements[{index}]"
            self._validate_statement(
                statement,
                location=location,
                semantics=semantics,
                nodes=nodes,
                edges=edges,
                paths=paths,
                hits=hits,
                papers=papers,
                error=error,
                warning=warning,
            )

        for index, route in enumerate(report.mechanism_routes):
            location = f"mechanism_routes[{index}]"
            route_paths = []
            for path_id in route.path_ids:
                path = paths.get(path_id)
                if path is None:
                    error("UNKNOWN_PATH_REF", location + ".path_ids", f"Unknown path ID: {path_id}")
                else:
                    route_paths.append(path)
            for node_id in route.mechanism_node_ids:
                node = nodes.get(node_id)
                if node is None:
                    error("UNKNOWN_NODE_REF", location + ".mechanism_node_ids", f"Unknown node ID: {node_id}")
                elif not is_mechanism_node(node_id, node, semantics):
                    error("NON_MECHANISM_NODE", location + ".mechanism_node_ids", f"Node is not mechanism-bearing: {node_id}")
            for edge_id in route.mechanism_edge_ids:
                edge = edges.get(edge_id)
                if edge is None:
                    error("UNKNOWN_EDGE_REF", location + ".mechanism_edge_ids", f"Unknown edge ID: {edge_id}")
                elif not is_mechanism_edge(edge, semantics):
                    error("NON_MECHANISM_EDGE", location + ".mechanism_edge_ids", f"Edge is not mechanism-bearing: {edge_id}")
            if route.structural_type.startswith("CROSS_PAPER") and len(set(route.paper_ids)) < 2:
                error("CROSS_PAPER_REQUIRES_TWO_PAPERS", location + ".paper_ids", "Cross-paper route must name at least two papers.")
            if route_paths:
                path_types = {path.quality.path_type for path in route_paths}
                if route.structural_type not in path_types:
                    warning(
                        "ROUTE_TYPE_NOT_PRESENT_IN_PATHS",
                        location + ".structural_type",
                        f"Declared structural_type {route.structural_type!r} is not present among referenced path types {sorted(path_types)!r}.",
                    )
                expected_alignment = any(any(step.edge_class in {"registry_alignment", "pattern_alignment"} for step in path.steps) for path in route_paths)
                expected_reverse = any(any(step.traversal_direction == "reverse" for step in path.steps) for path in route_paths)
                expected_navigation_heavy = any(
                    path.quality.navigation_burden == "high" or "navigation_heavy" in path.quality.path_tags
                    for path in route_paths
                )
                expected_verification = any(
                    path.quality.candidate_fraction > 0 or any(step.requires_verification for step in path.steps)
                    for path in route_paths
                )
                if route.uses_alignment != expected_alignment:
                    error("ROUTE_ALIGNMENT_FLAG_MISMATCH", location + ".uses_alignment", "uses_alignment does not match referenced path evidence.")
                if route.uses_reverse_navigation != expected_reverse:
                    error("ROUTE_REVERSE_FLAG_MISMATCH", location + ".uses_reverse_navigation", "uses_reverse_navigation does not match referenced path evidence.")
                if route.navigation_heavy != expected_navigation_heavy:
                    warning("ROUTE_NAVIGATION_FLAG_MISMATCH", location + ".navigation_heavy", "navigation_heavy does not match deterministic path-quality metadata.")
                if expected_verification and not route.requires_verification:
                    error("CANDIDATE_REQUIRES_VERIFICATION", location + ".requires_verification", "Route uses candidate evidence but is not marked requires_verification.")

        for index, motif in enumerate(report.recurring_mechanistic_motifs):
            location = f"recurring_mechanistic_motifs[{index}]"
            for path_id in motif.path_ids:
                if path_id not in paths:
                    error("UNKNOWN_PATH_REF", location + ".path_ids", f"Unknown path ID: {path_id}")

            motif_scientific_papers: set[str] = set()
            valid_mechanism_evidence_count = 0
            for node_id in motif.mechanism_node_ids:
                node = nodes.get(node_id)
                if node is None:
                    error("UNKNOWN_NODE_REF", location + ".mechanism_node_ids", f"Unknown node ID: {node_id}")
                elif not is_mechanism_node(node_id, node, semantics):
                    error("NON_MECHANISM_NODE", location + ".mechanism_node_ids", f"Node is not mechanism-bearing: {node_id}")
                else:
                    valid_mechanism_evidence_count += 1
                    if not is_alignment_node(node):
                        if node.source_paper_id:
                            motif_scientific_papers.add(str(node.source_paper_id))
                        motif_scientific_papers.update(
                            str(x) for x in node.source_paper_ids if str(x).strip()
                        )

            for edge_id in motif.mechanism_edge_ids:
                edge = edges.get(edge_id)
                if edge is None:
                    error("UNKNOWN_EDGE_REF", location + ".mechanism_edge_ids", f"Unknown edge ID: {edge_id}")
                elif not is_mechanism_edge(edge, semantics):
                    error("NON_MECHANISM_EDGE", location + ".mechanism_edge_ids", f"Edge is not mechanism-bearing: {edge_id}")
                else:
                    valid_mechanism_evidence_count += 1
                    if not is_alignment_edge(edge):
                        motif_scientific_papers.update(
                            str(x) for x in edge.source_paper_ids if str(x).strip()
                        )

            if valid_mechanism_evidence_count == 0:
                error(
                    "MOTIF_REQUIRES_MECHANISM_EVIDENCE",
                    location,
                    "A mechanistic motif must contain at least one mechanism-bearing scientific node or edge.",
                )

            declared_papers = set(motif.paper_ids)
            if motif.cross_paper and len(declared_papers) < 2:
                error(
                    "CROSS_PAPER_REQUIRES_TWO_PAPERS",
                    location + ".paper_ids",
                    "cross_paper=True requires at least two distinct papers.",
                )
            if declared_papers != motif_scientific_papers:
                error(
                    "MOTIF_PAPER_SCOPE_MISMATCH",
                    location + ".paper_ids",
                    "Motif paper_ids must equal the papers represented by its mechanism-bearing scientific evidence; "
                    f"declared={sorted(declared_papers)!r}, evidence={sorted(motif_scientific_papers)!r}.",
                )

            expected_cross_paper = len(motif_scientific_papers) >= 2
            if motif.cross_paper != expected_cross_paper:
                error(
                    "MOTIF_CROSS_PAPER_FLAG_MISMATCH",
                    location + ".cross_paper",
                    "cross_paper must be derived from mechanism-bearing motif evidence, not path or statement scope.",
                )

            self._validate_free_text_label(
                semantics=semantics,
                label=motif.label,
                location=location + ".label",
                statement_ids=motif.statement_ids,
                node_ids=motif.mechanism_node_ids,
                edge_ids=motif.mechanism_edge_ids,
                statements=statements,
                nodes=nodes,
                edges=edges,
                hits=hits,
                error=error,
            )

        for index, connection in enumerate(report.cross_paper_connections):
            location = f"cross_paper_connections[{index}]"
            if len(set(connection.paper_ids)) < 2:
                error("CROSS_PAPER_REQUIRES_TWO_PAPERS", location + ".paper_ids", "Cross-paper connection must name at least two distinct papers.")
            referenced_paths = [paths[path_id] for path_id in connection.path_ids if path_id in paths]
            for path_id in connection.path_ids:
                if path_id not in paths:
                    error("UNKNOWN_PATH_REF", location + ".path_ids", f"Unknown path ID: {path_id}")
            actual_alignment_edges = {
                step.edge_evidence_ref
                for path in referenced_paths
                for step in path.steps
                if step.edge_class in {"registry_alignment", "pattern_alignment"}
            }
            if connection.uses_alignment != bool(actual_alignment_edges):
                error("CONNECTION_ALIGNMENT_FLAG_MISMATCH", location + ".uses_alignment", "uses_alignment does not match referenced paths.")
            unknown_alignment = set(connection.alignment_edge_ids) - set(edges)
            for edge_id in sorted(unknown_alignment):
                error("UNKNOWN_EDGE_REF", location + ".alignment_edge_ids", f"Unknown edge ID: {edge_id}")
            if connection.uses_alignment and not set(connection.alignment_edge_ids).issubset(actual_alignment_edges):
                error("ALIGNMENT_EDGE_NOT_IN_PATH", location + ".alignment_edge_ids", "alignment_edge_ids must be drawn from referenced paths.")

        for index, unresolved in enumerate(report.unresolved_connections):
            location = f"unresolved_connections[{index}]"
            for path_id in unresolved.related_path_ids:
                if path_id not in paths:
                    error("UNKNOWN_PATH_REF", location + ".related_path_ids", f"Unknown path ID: {path_id}")

        for index, lever in enumerate(report.reported_design_levers):
            location = f"reported_design_levers[{index}]"
            for node_id in [*lever.mechanism_node_ids, *lever.outcome_node_ids]:
                if node_id not in nodes:
                    error("UNKNOWN_NODE_REF", location, f"Unknown node ID: {node_id}")
            self._validate_free_text_label(
                semantics=semantics,
                label=lever.label,
                location=location + ".label",
                statement_ids=lever.statement_ids,
                node_ids=[*lever.mechanism_node_ids, *lever.outcome_node_ids],
                edge_ids=[],
                statements=statements,
                nodes=nodes,
                edges=edges,
                hits=hits,
                error=error,
            )

        errors = sum(issue.severity == "error" for issue in issues)
        warnings = sum(issue.severity == "warning" for issue in issues)
        return ExplorationValidationResult(
            passes=errors == 0,
            errors=errors,
            warnings=warnings,
            issues=issues,
        )

    def _validate_free_text_label(
        self,
        *,
        semantics: DiscoverySemantics,
        label: str,
        location: str,
        statement_ids: Iterable[str],
        node_ids: Iterable[str],
        edge_ids: Iterable[str],
        statements: dict[str, ExplorerStatement],
        nodes: dict[str, Any],
        edges: dict[str, Any],
        hits: dict[str, Any],
        error: Any,
    ) -> None:
        if _contains_hypothesis_language(label):
            error(
                "HYPOTHESIS_LANGUAGE_FORBIDDEN",
                location,
                "Graph Explorer labels may summarize supplied evidence but may not introduce hypothesis language.",
            )
        if contains_strong_causal_language(label, semantics) and not _label_has_grounded_causal_support(
            semantics=semantics,
            statement_ids=statement_ids,
            node_ids=node_ids,
            edge_ids=edge_ids,
            statements=statements,
            nodes=nodes,
            edges=edges,
            hits=hits,
        ):
            error(
                "UNSUPPORTED_CAUSAL_LABEL",
                location,
                "Label introduces strong causal language that is not present in its grounded supporting statements, nodes, or scientific edges.",
            )

    def _validate_statement(
        self,
        statement: ExplorerStatement,
        *,
        location: str,
        semantics: DiscoverySemantics,
        nodes: dict[str, Any],
        edges: dict[str, Any],
        paths: dict[str, Any],
        hits: dict[str, Any],
        papers: dict[str, Any],
        error: Any,
        warning: Any,
    ) -> None:
        support_refs = (
            len(statement.support_node_ids)
            + len(statement.support_edge_ids)
            + len(statement.support_path_ids)
            + len(statement.support_direct_hit_ids)
        )
        if support_refs == 0:
            error("UNGROUNDED_STATEMENT", location, "Every Explorer statement requires at least one packet support reference.")

        support_nodes = []
        support_edges = []
        support_paths = []
        support_hits = []
        for node_id in statement.support_node_ids:
            node = nodes.get(node_id)
            if node is None:
                error("UNKNOWN_NODE_REF", location + ".support_node_ids", f"Unknown node ID: {node_id}")
            else:
                support_nodes.append((node_id, node))
        for edge_id in statement.support_edge_ids:
            edge = edges.get(edge_id)
            if edge is None:
                error("UNKNOWN_EDGE_REF", location + ".support_edge_ids", f"Unknown edge ID: {edge_id}")
            else:
                support_edges.append((edge_id, edge))
        for path_id in statement.support_path_ids:
            path = paths.get(path_id)
            if path is None:
                error("UNKNOWN_PATH_REF", location + ".support_path_ids", f"Unknown path ID: {path_id}")
            else:
                support_paths.append(path)
        for hit_id in statement.support_direct_hit_ids:
            hit = hits.get(hit_id)
            if hit is None:
                error("UNKNOWN_DIRECT_HIT_REF", location + ".support_direct_hit_ids", f"Unknown direct-hit ID: {hit_id}")
            else:
                support_hits.append(hit)

        for hit in support_hits:
            node = nodes.get(hit.node_evidence_ref)
            if node is not None:
                support_nodes.append((hit.node_evidence_ref, node))

        if statement.epistemic_role == "reported":
            grounded_non_alignment = any(not is_alignment_node(node) for _, node in support_nodes) or any(not is_alignment_edge(edge) for _, edge in support_edges)
            if not grounded_non_alignment:
                error(
                    "REPORTED_REQUIRES_SOURCE_GROUNDED_EVIDENCE",
                    location,
                    "A reported statement cannot be supported only by corpus-alignment/navigation structures.",
                )

        if statement.claim_kind == "mechanism":
            mechanism_support = any(is_mechanism_node(node_id, node, semantics) for node_id, node in support_nodes)
            mechanism_support = mechanism_support or any(is_mechanism_edge(edge, semantics) for _, edge in support_edges)
            mechanism_support = mechanism_support or any(path.quality.mechanism_bearing for path in support_paths)
            if not mechanism_support:
                error("MECHANISM_REQUIRES_MECHANISM_EVIDENCE", location, "Mechanism claim lacks mechanism-bearing node, edge, or path support.")

        candidate_support = any(node.requires_verification for _, node in support_nodes)
        candidate_support = candidate_support or any(edge.requires_verification for _, edge in support_edges)
        candidate_support = candidate_support or any(hit.requires_verification for hit in support_hits)
        candidate_support = candidate_support or any(
            path.quality.candidate_fraction > 0 or any(step.requires_verification for step in path.steps)
            for path in support_paths
        )
        if candidate_support and not statement.requires_verification:
            error("CANDIDATE_REQUIRES_VERIFICATION", location + ".requires_verification", "Statement uses candidate evidence but is not marked requires_verification.")

        path_has_reverse = any(any(step.traversal_direction == "reverse" for step in path.steps) for path in support_paths)
        scientific_claim = statement.claim_kind in {"observation", "mechanism", "association", "comparison"}
        if scientific_claim and path_has_reverse and not (statement.support_edge_ids or statement.support_node_ids or statement.support_direct_hit_ids):
            error(
                "REVERSE_PATH_NEEDS_SCIENTIFIC_SUPPORT",
                location,
                "Scientific claim cites a reverse-navigation path but no original scientific edge/node support.",
            )

        evidence_texts: list[str] = []
        evidence_papers: set[str] = set()
        for _, node in support_nodes:
            evidence_texts.extend([node.label, node.node_text])
            evidence_papers.update(node.source_paper_ids)
            if node.source_paper_id:
                evidence_papers.add(node.source_paper_id)
        for _, edge in support_edges:
            evidence_texts.extend([
                edge.scientific_source,
                edge.relation,
                edge.scientific_target,
                json.dumps(edge.evidence_pointers, ensure_ascii=False),
            ])
            evidence_papers.update(edge.source_paper_ids)
        for path in support_paths:
            evidence_papers.update(path.visited_paper_ids)
            evidence_papers.update(path.supporting_paper_ids)
        for hit in support_hits:
            node = nodes.get(hit.node_evidence_ref)
            if node is not None:
                evidence_papers.update(node.source_paper_ids)
                if node.source_paper_id:
                    evidence_papers.add(node.source_paper_id)

        unsupported_papers = set(statement.paper_ids) - evidence_papers
        if unsupported_papers:
            error(
                "STATEMENT_PAPER_MISMATCH",
                location + ".paper_ids",
                "Statement names papers not present in its cited support: " + ", ".join(sorted(unsupported_papers)),
            )

        statement_numbers = _numbers(statement.text)
        if statement_numbers:
            support_numbers = _numbers("\n".join(evidence_texts))
            missing_numbers = statement_numbers - support_numbers
            if missing_numbers:
                error(
                    "UNSUPPORTED_NUMERIC_ASSERTION",
                    location + ".text",
                    "Numeric values are absent from cited evidence text: " + ", ".join(sorted(missing_numbers)),
                )

        if contains_absence_language(statement.text):
            for paper_id in statement.paper_ids:
                paper = papers.get(paper_id)
                if paper is None:
                    error("UNKNOWN_PAPER_REF", location + ".paper_ids", f"Unknown paper ID: {paper_id}")
                elif not paper.absence_claims_allowed:
                    error(
                        "PAPER_ABSENCE_CLAIM_NOT_ALLOWED",
                        location + ".text",
                        f"Paper-specific absence claim is not allowed for {paper_id} under current extraction completeness metadata.",
                    )
            if statement.claim_kind != "scope_limit":
                warning(
                    "ABSENCE_LANGUAGE_SHOULD_BE_SCOPE_LIMIT",
                    location + ".claim_kind",
                    "Absence language should normally be expressed as a packet-scoped scope_limit statement.",
                )

        if _contains_hypothesis_language(statement.text):
            error(
                "HYPOTHESIS_LANGUAGE_FORBIDDEN",
                location + ".text",
                "Graph Explorer may organize supplied evidence but may not introduce novel hypotheses.",
            )
