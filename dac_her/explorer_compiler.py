from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict

from pipeline_core.discovery.explorer_contracts import (
    CrossPaperConnection,
    EvidenceTension,
    ExplorationReport,
    ExplorerStatement,
    GraphExplorerPacket,
    MechanismRoute,
    MechanisticMotif,
    ReportedDesignLever,
    UnresolvedConnection,
)
from pipeline_core.discovery_semantics import (
    is_alignment_edge,
    is_alignment_node,
    is_mechanism_edge,
    is_mechanism_node,
)
from dac_her.domains import get_domain_profile
from pipeline_core.discovery.explorer_draft import ExplorationDraft



_ALLOWED_ROUTE_TYPES = {
    "DIRECT_MECHANISTIC",
    "CROSS_PAPER_MECHANISTIC",
    "CROSS_PAPER_BRIDGE",
    "SHARED_ENTITY_BRIDGE",
    "SCAFFOLD_NAVIGATION",
    "CANDIDATE_EXPLORATION",
}

_ROUTE_TYPE_PRIORITY = (
    "CANDIDATE_EXPLORATION",
    "CROSS_PAPER_MECHANISTIC",
    "DIRECT_MECHANISTIC",
    "CROSS_PAPER_BRIDGE",
    "SHARED_ENTITY_BRIDGE",
    "SCAFFOLD_NAVIGATION",
)


class CompileIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    location: str
    message: str


class ExplorationCompileError(ValueError):
    def __init__(self, issues: list[CompileIssue]) -> None:
        self.issues = issues
        super().__init__("; ".join(f"{issue.code}: {issue.message}" for issue in issues))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:length]}"


def _sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value).strip()})


def _path_requires_verification(path: Any) -> bool:
    return bool(
        float(path.quality.candidate_fraction or 0.0) > 0.0
        or any(bool(step.requires_verification) for step in path.steps)
    )


def _path_uses_alignment(path: Any) -> bool:
    return any(step.edge_class in {"registry_alignment", "pattern_alignment"} for step in path.steps)


def _path_uses_reverse(path: Any) -> bool:
    return any(step.traversal_direction == "reverse" for step in path.steps)


def _path_navigation_heavy(path: Any) -> bool:
    return path.quality.navigation_burden == "high" or "navigation_heavy" in set(path.quality.path_tags)


def _select_route_type(paths: list[Any]) -> str:
    present = {str(path.quality.path_type) for path in paths if str(path.quality.path_type) in _ALLOWED_ROUTE_TYPES}
    if not present:
        return "SCAFFOLD_NAVIGATION"
    for route_type in _ROUTE_TYPE_PRIORITY:
        if route_type in present:
            return route_type
    return sorted(present)[0]


class ExplorationReportCompiler:
    """Deterministically enrich an LLM-owned ExplorationDraft.

    The LLM chooses textual organization and evidence references.  This compiler
    owns graph bookkeeping: stable IDs, source-paper scope, verification flags,
    route type/flags, mechanism-node/edge membership, and report identity.
    """

    def compile(self, packet: GraphExplorerPacket, draft: ExplorationDraft) -> ExplorationReport:
        issues: list[CompileIssue] = []
        semantics = get_domain_profile(
            packet.domain_profile_id
        ).discovery
        paths = {path.path_id: path for path in packet.paths}
        nodes = packet.evidence_catalog.nodes
        edges = packet.evidence_catalog.edges
        hits = {hit.hit_id: hit for hit in packet.direct_concept_hits}

        statement_drafts = {row.local_id: row for row in draft.statements}
        if len(statement_drafts) != len(draft.statements):
            issues.append(
                CompileIssue(
                    code="DUPLICATE_LOCAL_STATEMENT_ID",
                    location="draft.statements",
                    message="Explorer statement local IDs must be unique.",
                )
            )

        def require_statement(local_id: str, location: str) -> None:
            if local_id not in statement_drafts:
                issues.append(
                    CompileIssue(
                        code="UNKNOWN_LOCAL_STATEMENT_REF",
                        location=location,
                        message=f"Unknown statement local_id: {local_id}",
                    )
                )

        for index, local_id in enumerate(draft.direct_finding_local_ids):
            require_statement(local_id, f"draft.direct_finding_local_ids[{index}]")
        for i, route in enumerate(draft.mechanism_routes):
            for local_id in route.statement_local_ids:
                require_statement(local_id, f"draft.mechanism_routes[{i}].statement_local_ids")
            for path_id in route.path_ids:
                if path_id not in paths:
                    issues.append(CompileIssue(code="UNKNOWN_PATH_REF", location=f"draft.mechanism_routes[{i}].path_ids", message=f"Unknown path ID: {path_id}"))
        for i, motif in enumerate(draft.recurring_mechanistic_motifs):
            for local_id in motif.statement_local_ids:
                require_statement(local_id, f"draft.recurring_mechanistic_motifs[{i}].statement_local_ids")
            for path_id in motif.path_ids:
                if path_id not in paths:
                    issues.append(CompileIssue(code="UNKNOWN_PATH_REF", location=f"draft.recurring_mechanistic_motifs[{i}].path_ids", message=f"Unknown path ID: {path_id}"))
        for i, connection in enumerate(draft.cross_paper_connections):
            for local_id in connection.statement_local_ids:
                require_statement(local_id, f"draft.cross_paper_connections[{i}].statement_local_ids")
            for path_id in connection.path_ids:
                if path_id not in paths:
                    issues.append(CompileIssue(code="UNKNOWN_PATH_REF", location=f"draft.cross_paper_connections[{i}].path_ids", message=f"Unknown path ID: {path_id}"))
        for i, tension in enumerate(draft.evidence_tensions):
            require_statement(tension.statement_local_id, f"draft.evidence_tensions[{i}].statement_local_id")
            for local_id in tension.side_a_statement_local_ids:
                require_statement(local_id, f"draft.evidence_tensions[{i}].side_a_statement_local_ids")
            for local_id in tension.side_b_statement_local_ids:
                require_statement(local_id, f"draft.evidence_tensions[{i}].side_b_statement_local_ids")
        for i, unresolved in enumerate(draft.unresolved_connections):
            require_statement(unresolved.statement_local_id, f"draft.unresolved_connections[{i}].statement_local_id")
            for path_id in unresolved.related_path_ids:
                if path_id not in paths:
                    issues.append(CompileIssue(code="UNKNOWN_PATH_REF", location=f"draft.unresolved_connections[{i}].related_path_ids", message=f"Unknown path ID: {path_id}"))
        for i, lever in enumerate(draft.reported_design_levers):
            for local_id in lever.statement_local_ids:
                require_statement(local_id, f"draft.reported_design_levers[{i}].statement_local_ids")

        if issues:
            raise ExplorationCompileError(issues)

        def support_papers(statement: Any, *, include_paths: bool = True) -> list[str]:
            papers: set[str] = set()
            for node_id in statement.support_node_ids:
                node = nodes.get(node_id)
                if node is None or is_alignment_node(node):
                    continue
                if node.source_paper_id:
                    papers.add(str(node.source_paper_id))
                papers.update(str(x) for x in node.source_paper_ids if str(x).strip())
            for edge_id in statement.support_edge_ids:
                edge = edges.get(edge_id)
                if edge is None or is_alignment_edge(edge):
                    continue
                papers.update(str(x) for x in edge.source_paper_ids if str(x).strip())
            if include_paths:
                for path_id in statement.support_path_ids:
                    path = paths.get(path_id)
                    if path is not None:
                        papers.update(path.visited_paper_ids)
                        papers.update(path.supporting_paper_ids)
            for hit_id in statement.support_direct_hit_ids:
                hit = hits.get(hit_id)
                if hit is None:
                    continue
                node = nodes.get(hit.node_evidence_ref)
                if node is not None and not is_alignment_node(node):
                    if node.source_paper_id:
                        papers.add(str(node.source_paper_id))
                    papers.update(str(x) for x in node.source_paper_ids if str(x).strip())
            return sorted(papers)

        def support_requires_verification(statement: Any) -> bool:
            for node_id in statement.support_node_ids:
                node = nodes.get(node_id)
                if node is not None and node.requires_verification:
                    return True
            for edge_id in statement.support_edge_ids:
                edge = edges.get(edge_id)
                if edge is not None and edge.requires_verification:
                    return True
            for path_id in statement.support_path_ids:
                path = paths.get(path_id)
                if path is not None and _path_requires_verification(path):
                    return True
            for hit_id in statement.support_direct_hit_ids:
                hit = hits.get(hit_id)
                if hit is not None and hit.requires_verification:
                    return True
            return False

        final_statements: list[ExplorerStatement] = []
        statement_id_map: dict[str, str] = {}
        for row in draft.statements:
            canonical_support = {
                "node": sorted(row.support_node_ids),
                "edge": sorted(row.support_edge_ids),
                "path": sorted(row.support_path_ids),
                "hit": sorted(row.support_direct_hit_ids),
            }
            statement_id = _stable_id(
                "stmt",
                packet.packet_sha256,
                row.text,
                row.epistemic_role,
                row.claim_kind,
                _canonical_json(canonical_support),
            )
            statement_id_map[row.local_id] = statement_id
            final_statements.append(
                ExplorerStatement(
                    statement_id=statement_id,
                    text=row.text,
                    epistemic_role=row.epistemic_role,
                    claim_kind=row.claim_kind,
                    support_node_ids=_sorted_unique(row.support_node_ids),
                    support_edge_ids=_sorted_unique(row.support_edge_ids),
                    support_path_ids=_sorted_unique(row.support_path_ids),
                    support_direct_hit_ids=_sorted_unique(row.support_direct_hit_ids),
                    paper_ids=support_papers(row),
                    requires_verification=support_requires_verification(row),
                )
            )

        statements_by_id = {row.statement_id: row for row in final_statements}

        def final_statement_ids(local_ids: Iterable[str]) -> list[str]:
            return [statement_id_map[local_id] for local_id in local_ids]

        def referenced_statement_papers(local_ids: Iterable[str]) -> set[str]:
            papers: set[str] = set()
            for final_id in final_statement_ids(local_ids):
                papers.update(statements_by_id[final_id].paper_ids)
            return papers

        mechanism_routes: list[MechanismRoute] = []
        for row in draft.mechanism_routes:
            route_paths = [paths[path_id] for path_id in row.path_ids]
            mechanism_node_ids = _sorted_unique(
                node_id
                for path in route_paths
                for node_id in path.quality.mechanism_node_ids
                if node_id in nodes and is_mechanism_node(node_id, nodes[node_id], semantics)
            )
            mechanism_edge_ids = _sorted_unique(
                step.edge_evidence_ref
                for path in route_paths
                for step in path.steps
                if step.edge_evidence_ref in edges and is_mechanism_edge(edges[step.edge_evidence_ref], semantics)
            )
            paper_ids = _sorted_unique(
                paper_id
                for path in route_paths
                for paper_id in path.visited_paper_ids
            )
            statement_ids = final_statement_ids(row.statement_local_ids)
            route_id = _stable_id("route", packet.packet_sha256, _canonical_json(sorted(row.path_ids)), _canonical_json(statement_ids))
            mechanism_routes.append(
                MechanismRoute(
                    route_id=route_id,
                    path_ids=_sorted_unique(row.path_ids),
                    statement_ids=statement_ids,
                    mechanism_node_ids=mechanism_node_ids,
                    mechanism_edge_ids=mechanism_edge_ids,
                    paper_ids=paper_ids,
                    structural_type=_select_route_type(route_paths),
                    navigation_heavy=any(_path_navigation_heavy(path) for path in route_paths),
                    uses_alignment=any(_path_uses_alignment(path) for path in route_paths),
                    uses_reverse_navigation=any(_path_uses_reverse(path) for path in route_paths),
                    requires_verification=any(_path_requires_verification(path) for path in route_paths),
                )
            )

        motifs: list[MechanisticMotif] = []
        for row in draft.recurring_mechanistic_motifs:
            # v2.5.1-h2 provenance precedence:
            #
            # * statement_local_ids are narrative/reference links.
            # * explicit motif support_node_ids/support_edge_ids, when present,
            #   are the authoritative scientific provenance of the motif.
            # * only when explicit motif support is absent do we fall back to
            #   the scientific supports of referenced statements.
            # * path_ids are navigation context only and never expand motif
            #   scientific scope.
            #
            # This prevents a cross-paper synthesis statement from promoting a
            # single-paper motif into a cross-paper recurring motif merely
            # because that statement also cites another paper for a scope limit.
            motif_statement_drafts = [
                statement_drafts[local_id]
                for local_id in row.statement_local_ids
            ]

            has_explicit_scientific_support = bool(row.support_node_ids or row.support_edge_ids)
            supported_node_ids: set[str] = set()
            supported_edge_ids: set[str] = set()
            supported_direct_hit_ids: set[str] = set()

            if has_explicit_scientific_support:
                supported_node_ids.update(row.support_node_ids)
                supported_edge_ids.update(row.support_edge_ids)
            else:
                for statement in motif_statement_drafts:
                    supported_node_ids.update(statement.support_node_ids)
                    supported_edge_ids.update(statement.support_edge_ids)
                    supported_direct_hit_ids.update(statement.support_direct_hit_ids)

            for hit_id in supported_direct_hit_ids:
                hit = hits.get(hit_id)
                if hit is not None:
                    supported_node_ids.add(hit.node_evidence_ref)

            mechanism_node_ids = {
                node_id
                for node_id in supported_node_ids
                if node_id in nodes and is_mechanism_node(node_id, nodes[node_id], semantics)
            }
            mechanism_edge_ids = {
                edge_id
                for edge_id in supported_edge_ids
                if edge_id in edges and is_mechanism_edge(edges[edge_id], semantics)
            }

            # A MechanisticMotif's paper scope is defined by its actual
            # mechanism-bearing scientific evidence, not by all papers named in
            # explanatory statements.  This gives the validator an independent,
            # deterministic invariant for motif scope.
            paper_ids: set[str] = set()
            for node_id in mechanism_node_ids:
                node = nodes.get(node_id)
                if node is None or is_alignment_node(node):
                    continue
                if node.source_paper_id:
                    paper_ids.add(str(node.source_paper_id))
                paper_ids.update(str(x) for x in node.source_paper_ids if str(x).strip())
            for edge_id in mechanism_edge_ids:
                edge = edges.get(edge_id)
                if edge is None or is_alignment_edge(edge):
                    continue
                paper_ids.update(str(x) for x in edge.source_paper_ids if str(x).strip())

            statement_ids = final_statement_ids(row.statement_local_ids)
            motif_id = _stable_id("motif", packet.packet_sha256, row.label, _canonical_json(statement_ids), _canonical_json(sorted(row.path_ids)))
            motifs.append(
                MechanisticMotif(
                    motif_id=motif_id,
                    label=row.label,
                    statement_ids=statement_ids,
                    mechanism_node_ids=sorted(mechanism_node_ids),
                    mechanism_edge_ids=sorted(mechanism_edge_ids),
                    path_ids=_sorted_unique(row.path_ids),
                    paper_ids=sorted(paper_ids),
                    cross_paper=len(paper_ids) >= 2,
                )
            )

        connections: list[CrossPaperConnection] = []
        for row in draft.cross_paper_connections:
            connection_paths = [paths[path_id] for path_id in row.path_ids]
            paper_ids = referenced_statement_papers(row.statement_local_ids)
            for path in connection_paths:
                paper_ids.update(path.visited_paper_ids)
            alignment_edge_ids = _sorted_unique(
                step.edge_evidence_ref
                for path in connection_paths
                for step in path.steps
                if step.edge_class in {"registry_alignment", "pattern_alignment"}
            )
            statement_ids = final_statement_ids(row.statement_local_ids)
            connection_id = _stable_id("connection", packet.packet_sha256, _canonical_json(sorted(row.path_ids)), _canonical_json(statement_ids))
            connections.append(
                CrossPaperConnection(
                    connection_id=connection_id,
                    statement_ids=statement_ids,
                    path_ids=_sorted_unique(row.path_ids),
                    paper_ids=sorted(paper_ids),
                    uses_alignment=bool(alignment_edge_ids),
                    alignment_edge_ids=alignment_edge_ids,
                    requires_verification=any(_path_requires_verification(path) for path in connection_paths),
                )
            )

        tensions: list[EvidenceTension] = []
        for row in draft.evidence_tensions:
            statement_id = statement_id_map[row.statement_local_id]
            side_a = final_statement_ids(row.side_a_statement_local_ids)
            side_b = final_statement_ids(row.side_b_statement_local_ids)
            paper_ids = set(statements_by_id[statement_id].paper_ids)
            for final_id in [*side_a, *side_b]:
                paper_ids.update(statements_by_id[final_id].paper_ids)
            tension_id = _stable_id("tension", packet.packet_sha256, statement_id, _canonical_json(side_a), _canonical_json(side_b), row.tension_type)
            tensions.append(
                EvidenceTension(
                    tension_id=tension_id,
                    statement_id=statement_id,
                    side_a_statement_ids=side_a,
                    side_b_statement_ids=side_b,
                    tension_type=row.tension_type,
                    paper_ids=sorted(paper_ids),
                )
            )

        unresolved: list[UnresolvedConnection] = []
        for row in draft.unresolved_connections:
            statement_id = statement_id_map[row.statement_local_id]
            gap_id = _stable_id("gap", packet.packet_sha256, statement_id, _canonical_json(sorted(row.related_path_ids)), row.reason)
            unresolved.append(
                UnresolvedConnection(
                    gap_id=gap_id,
                    statement_id=statement_id,
                    related_path_ids=_sorted_unique(row.related_path_ids),
                    reason=row.reason,
                )
            )

        levers: list[ReportedDesignLever] = []
        for row in draft.reported_design_levers:
            statement_ids = final_statement_ids(row.statement_local_ids)
            mechanism_node_ids = _sorted_unique(row.mechanism_node_ids)
            outcome_node_ids = _sorted_unique(row.outcome_node_ids)
            paper_ids = referenced_statement_papers(row.statement_local_ids)
            for node_id in [*mechanism_node_ids, *outcome_node_ids]:
                node = nodes.get(node_id)
                if node is None or is_alignment_node(node):
                    continue
                if node.source_paper_id:
                    paper_ids.add(str(node.source_paper_id))
                paper_ids.update(str(x) for x in node.source_paper_ids if str(x).strip())
            lever_id = _stable_id("lever", packet.packet_sha256, row.label, _canonical_json(statement_ids), _canonical_json(mechanism_node_ids), _canonical_json(outcome_node_ids))
            levers.append(
                ReportedDesignLever(
                    lever_id=lever_id,
                    label=row.label,
                    statement_ids=statement_ids,
                    mechanism_node_ids=mechanism_node_ids,
                    outcome_node_ids=outcome_node_ids,
                    paper_ids=sorted(paper_ids),
                )
            )

        provisional = ExplorationReport(
            report_id="pending",
            task_id=packet.task.task_id,
            source_packet_sha256=packet.packet_sha256,
            statements=final_statements,
            direct_findings=final_statement_ids(draft.direct_finding_local_ids),
            mechanism_routes=mechanism_routes,
            recurring_mechanistic_motifs=motifs,
            cross_paper_connections=connections,
            evidence_tensions=tensions,
            unresolved_connections=unresolved,
            reported_design_levers=levers,
        )
        content = provisional.model_dump(mode="json", exclude={"report_id"})
        report_id = _stable_id("report", packet.packet_sha256, _canonical_json(content))
        return provisional.model_copy(update={"report_id": report_id})
