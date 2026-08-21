from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.discovery_semantics import (
    contains_strong_causal_language,
    is_alignment_edge,
    is_alignment_node,
    is_mechanism_edge,
    is_mechanism_node,
)
from pipeline_core.domain.domain_profile import (
    ScientificDomainProfile,
)
from pipeline_core.discovery.explorer_contracts import GraphExplorerPacket
from pipeline_core.discovery.explorer_draft import ExplorationDraft
from pipeline_core.discovery.explorer_text_safety import contains_absence_language


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExplorerNormalizationAction(_StrictModel):
    action: Literal[
        "downgrade_mechanism_to_association",
        "drop_unsupported_strong_causal_statement",
        "drop_unverifiable_paper_absence_statement",
        "prune_dependent_statement_refs",
        "drop_dependent_section_object",
        "drop_unsupported_mechanistic_motif",
        "blocked_strong_causal_mechanism",
    ]
    location: str
    reason: str
    local_id: str | None = None


class ExplorerNormalizationAudit(_StrictModel):
    schema_version: Literal[
        "graph-explorer-normalization-v1"
    ] = "graph-explorer-normalization-v1"
    domain_profile_id: str
    applied: bool = False
    action_count: int = 0
    blocked_count: int = 0
    actions: list[ExplorerNormalizationAction] = Field(
        default_factory=list
    )


@dataclass(frozen=True)
class ExplorerNormalizationResult:
    draft: ExplorationDraft
    audit: ExplorerNormalizationAudit


def _path_map(packet: GraphExplorerPacket) -> dict[str, Any]:
    return {row.path_id: row for row in packet.paths}


def _hit_map(packet: GraphExplorerPacket) -> dict[str, Any]:
    return {row.hit_id: row for row in packet.direct_concept_hits}


class ExplorerDraftNormalizer:
    """Conservative one-way normalization after LLM repair is exhausted.

    Allowed transformations only reduce epistemic strength or remove an
    unsupported higher-order construct. Scientific text and evidence references
    are never rewritten, invented, fuzzily matched, or redirected.
    """

    def __init__(
        self,
        *,
        domain_profile: ScientificDomainProfile,
    ) -> None:
        self.domain_profile = domain_profile

    def _statement_has_mechanism_support(
        self,
        *,
        packet: GraphExplorerPacket,
        statement: Any,
        semantics: Any,
    ) -> bool:
        nodes = packet.evidence_catalog.nodes
        edges = packet.evidence_catalog.edges
        paths = _path_map(packet)
        hits = _hit_map(packet)

        for node_id in statement.support_node_ids:
            node = nodes.get(node_id)
            if (
                node is not None
                and is_mechanism_node(
                    node_id,
                    node,
                    semantics,
                )
            ):
                return True

        for edge_id in statement.support_edge_ids:
            edge = edges.get(edge_id)
            if (
                edge is not None
                and is_mechanism_edge(
                    edge,
                    semantics,
                )
            ):
                return True

        for path_id in statement.support_path_ids:
            path = paths.get(path_id)
            if (
                path is not None
                and bool(path.quality.mechanism_bearing)
            ):
                return True

        for hit_id in statement.support_direct_hit_ids:
            hit = hits.get(hit_id)
            if hit is None:
                continue
            if bool(hit.mechanism_bearing):
                return True
            node = nodes.get(hit.node_evidence_ref)
            if (
                node is not None
                and is_mechanism_node(
                    hit.node_evidence_ref,
                    node,
                    semantics,
                )
            ):
                return True

        return False

    def _motif_has_mechanism_support(
        self,
        *,
        packet: GraphExplorerPacket,
        motif: Any,
        statements: dict[str, Any],
        semantics: Any,
    ) -> bool:
        nodes = packet.evidence_catalog.nodes
        edges = packet.evidence_catalog.edges
        hits = _hit_map(packet)

        supported_nodes = set(motif.support_node_ids)
        supported_edges = set(motif.support_edge_ids)
        supported_hits: set[str] = set()

        if not supported_nodes and not supported_edges:
            for local_id in motif.statement_local_ids:
                statement = statements.get(local_id)
                if statement is None:
                    continue
                supported_nodes.update(
                    statement.support_node_ids
                )
                supported_edges.update(
                    statement.support_edge_ids
                )
                supported_hits.update(
                    statement.support_direct_hit_ids
                )

        for hit_id in supported_hits:
            hit = hits.get(hit_id)
            if hit is not None:
                supported_nodes.add(hit.node_evidence_ref)

        return any(
            node_id in nodes
            and is_mechanism_node(
                node_id,
                nodes[node_id],
                semantics,
            )
            for node_id in supported_nodes
        ) or any(
            edge_id in edges
            and is_mechanism_edge(
                edges[edge_id],
                semantics,
            )
            for edge_id in supported_edges
        )

    def _statement_support_papers(
        self,
        *,
        packet: GraphExplorerPacket,
        statement: Any,
    ) -> set[str]:
        """Mirror compiler paper-scope recovery for pre-compile safety checks."""
        nodes = packet.evidence_catalog.nodes
        edges = packet.evidence_catalog.edges
        paths = _path_map(packet)
        hits = _hit_map(packet)
        papers: set[str] = set()

        for node_id in statement.support_node_ids:
            node = nodes.get(node_id)
            if node is None or is_alignment_node(node):
                continue
            if node.source_paper_id:
                papers.add(str(node.source_paper_id))
            papers.update(
                str(value)
                for value in node.source_paper_ids
                if str(value).strip()
            )

        for edge_id in statement.support_edge_ids:
            edge = edges.get(edge_id)
            if edge is None or is_alignment_edge(edge):
                continue
            papers.update(
                str(value)
                for value in edge.source_paper_ids
                if str(value).strip()
            )

        for path_id in statement.support_path_ids:
            path = paths.get(path_id)
            if path is None:
                continue
            papers.update(str(value) for value in path.visited_paper_ids)
            papers.update(str(value) for value in path.supporting_paper_ids)

        for hit_id in statement.support_direct_hit_ids:
            hit = hits.get(hit_id)
            if hit is None:
                continue
            node = nodes.get(hit.node_evidence_ref)
            if node is None or is_alignment_node(node):
                continue
            if node.source_paper_id:
                papers.add(str(node.source_paper_id))
            papers.update(
                str(value)
                for value in node.source_paper_ids
                if str(value).strip()
            )

        return {value for value in papers if value.strip()}

    def _blocked_absence_papers(
        self,
        *,
        packet: GraphExplorerPacket,
        statement: Any,
    ) -> list[str]:
        if not contains_absence_language(statement.text):
            return []
        paper_scope = {row.paper_id: row for row in packet.corpus.papers}
        return sorted(
            paper_id
            for paper_id in self._statement_support_papers(
                packet=packet,
                statement=statement,
            )
            if paper_id in paper_scope
            and paper_scope[paper_id].absence_claims_allowed is False
        )

    def normalize(
        self,
        packet: GraphExplorerPacket,
        draft: ExplorationDraft,
    ) -> ExplorerNormalizationResult:
        if (
            packet.domain_profile_id
            != self.domain_profile.profile_id
        ):
            raise ValueError(
                "Graph Explorer normalizer domain profile mismatch: "
                f"packet={packet.domain_profile_id!r}, "
                f"normalizer={self.domain_profile.profile_id!r}"
            )

        semantics = self.domain_profile.discovery
        actions: list[ExplorerNormalizationAction] = []

        normalized_statements = []
        for index, statement in enumerate(draft.statements):
            blocked_absence_papers = self._blocked_absence_papers(
                packet=packet,
                statement=statement,
            )
            if blocked_absence_papers:
                # Extraction incompleteness cannot establish a paper-specific
                # negative fact. Rewriting the sentence or silently removing a
                # paper from provenance would change scientific meaning, so the
                # entire statement is conservatively removed. Phase 1+2 cascade
                # pruning then removes or repairs every dependent draft object.
                actions.append(
                    ExplorerNormalizationAction(
                        action="drop_unverifiable_paper_absence_statement",
                        location=f"statements[{index}]",
                        local_id=statement.local_id,
                        reason=(
                            "Paper-specific absence cannot be inferred because "
                            "extraction completeness does not permit absence "
                            "claims for supporting paper(s): "
                            + ", ".join(blocked_absence_papers)
                        ),
                    )
                )
                continue

            if (
                statement.claim_kind == "mechanism"
                and not self._statement_has_mechanism_support(
                    packet=packet,
                    statement=statement,
                    semantics=semantics,
                )
            ):
                if contains_strong_causal_language(
                    statement.text,
                    semantics,
                ):
                    # Safe one-way weakening: a strongly causal sentence cannot
                    # be relabelled as a mere association without changing its
                    # scientific meaning. Drop the unsupported statement instead
                    # and deterministically prune all dependent local references
                    # below. No text or evidence reference is rewritten.
                    actions.append(
                        ExplorerNormalizationAction(
                            action="drop_unsupported_strong_causal_statement",
                            location=f"statements[{index}]",
                            local_id=statement.local_id,
                            reason=(
                                "Mechanism claim lacks mechanism-bearing support "
                                "and uses profile-defined strong causal language. "
                                "The statement was removed rather than silently "
                                "downgraded to an association."
                            ),
                        )
                    )
                else:
                    normalized_statements.append(
                        statement.model_copy(
                            update={
                                "claim_kind": "association"
                            }
                        )
                    )
                    actions.append(
                        ExplorerNormalizationAction(
                            action="downgrade_mechanism_to_association",
                            location=f"statements[{index}]",
                            local_id=statement.local_id,
                            reason=(
                                "Mechanism claim lacks mechanism-bearing "
                                "evidence and contains no profile-defined "
                                "strong causal language. Only claim strength "
                                "was reduced; text/support were unchanged."
                            ),
                        )
                    )
            else:
                normalized_statements.append(statement)

        statement_map = {
            row.local_id: row
            for row in normalized_statements
        }
        surviving_statement_ids = set(statement_map)
        dropped_statement_ids = {
            row.local_id
            for row in draft.statements
            if row.local_id not in surviving_statement_ids
        }

        def _pruned_refs(
            values: list[str],
            *,
            location: str,
            local_id: str | None,
        ) -> list[str]:
            kept = [
                value
                for value in values
                if value in surviving_statement_ids
            ]
            removed = [
                value
                for value in values
                if value in dropped_statement_ids
            ]
            if removed:
                actions.append(
                    ExplorerNormalizationAction(
                        action="prune_dependent_statement_refs",
                        location=location,
                        local_id=local_id,
                        reason=(
                            "Removed references to statements that were safely "
                            "dropped during deterministic epistemic weakening: "
                            + ", ".join(sorted(set(removed)))
                        ),
                    )
                )
            return kept

        def _drop_section_object(
            *,
            location: str,
            local_id: str | None,
            reason: str,
        ) -> None:
            actions.append(
                ExplorerNormalizationAction(
                    action="drop_dependent_section_object",
                    location=location,
                    local_id=local_id,
                    reason=reason,
                )
            )

        normalized_direct_findings = _pruned_refs(
            list(draft.direct_finding_local_ids),
            location="direct_finding_local_ids",
            local_id=None,
        )

        normalized_routes = []
        for index, route in enumerate(draft.mechanism_routes):
            refs = _pruned_refs(
                list(route.statement_local_ids),
                location=f"mechanism_routes[{index}].statement_local_ids",
                local_id=route.local_id,
            )
            if not refs:
                _drop_section_object(
                    location=f"mechanism_routes[{index}]",
                    local_id=route.local_id,
                    reason=(
                        "Mechanism route lost every statement reference after an "
                        "unsupported strong-causal statement was removed."
                    ),
                )
                continue
            normalized_routes.append(
                route.model_copy(update={"statement_local_ids": refs})
            )

        normalized_motifs = []
        for index, motif in enumerate(
            draft.recurring_mechanistic_motifs
        ):
            motif_refs = _pruned_refs(
                list(motif.statement_local_ids),
                location=(
                    "recurring_mechanistic_motifs"
                    f"[{index}].statement_local_ids"
                ),
                local_id=motif.local_id,
            )
            if not motif_refs:
                _drop_section_object(
                    location=f"recurring_mechanistic_motifs[{index}]",
                    local_id=motif.local_id,
                    reason=(
                        "Mechanistic motif lost every statement reference after "
                        "unsupported statement removal."
                    ),
                )
                continue
            motif = motif.model_copy(
                update={"statement_local_ids": motif_refs}
            )
            if self._motif_has_mechanism_support(
                packet=packet,
                motif=motif,
                statements=statement_map,
                semantics=semantics,
            ):
                normalized_motifs.append(motif)
            else:
                actions.append(
                    ExplorerNormalizationAction(
                        action="drop_unsupported_mechanistic_motif",
                        location=(
                            "recurring_mechanistic_motifs"
                            f"[{index}]"
                        ),
                        local_id=motif.local_id,
                        reason=(
                            "Mechanistic motif has no mechanism-bearing "
                            "scientific node or edge under the selected "
                            "domain semantics. The unsupported higher-order "
                            "motif was removed without rewriting evidence."
                        ),
                    )
                )

        normalized_connections = []
        for index, connection in enumerate(
            draft.cross_paper_connections
        ):
            refs = _pruned_refs(
                list(connection.statement_local_ids),
                location=(
                    f"cross_paper_connections[{index}]"
                    ".statement_local_ids"
                ),
                local_id=connection.local_id,
            )
            if not refs:
                _drop_section_object(
                    location=f"cross_paper_connections[{index}]",
                    local_id=connection.local_id,
                    reason=(
                        "Cross-paper connection lost every statement reference "
                        "after unsupported statement removal."
                    ),
                )
                continue
            normalized_connections.append(
                connection.model_copy(update={"statement_local_ids": refs})
            )

        normalized_tensions = []
        for index, tension in enumerate(draft.evidence_tensions):
            primary_survives = (
                tension.statement_local_id in surviving_statement_ids
            )
            side_a = _pruned_refs(
                list(tension.side_a_statement_local_ids),
                location=(
                    f"evidence_tensions[{index}]"
                    ".side_a_statement_local_ids"
                ),
                local_id=tension.local_id,
            )
            side_b = _pruned_refs(
                list(tension.side_b_statement_local_ids),
                location=(
                    f"evidence_tensions[{index}]"
                    ".side_b_statement_local_ids"
                ),
                local_id=tension.local_id,
            )
            if not primary_survives or not side_a or not side_b:
                _drop_section_object(
                    location=f"evidence_tensions[{index}]",
                    local_id=tension.local_id,
                    reason=(
                        "Evidence tension became structurally incomplete after "
                        "unsupported statement removal."
                    ),
                )
                continue
            normalized_tensions.append(
                tension.model_copy(
                    update={
                        "side_a_statement_local_ids": side_a,
                        "side_b_statement_local_ids": side_b,
                    }
                )
            )

        normalized_unresolved = []
        for index, unresolved in enumerate(
            draft.unresolved_connections
        ):
            if unresolved.statement_local_id not in surviving_statement_ids:
                _drop_section_object(
                    location=f"unresolved_connections[{index}]",
                    local_id=unresolved.local_id,
                    reason=(
                        "Unresolved connection referenced a statement removed "
                        "during deterministic epistemic weakening."
                    ),
                )
                continue
            normalized_unresolved.append(unresolved)

        normalized_levers = []
        for index, lever in enumerate(draft.reported_design_levers):
            refs = _pruned_refs(
                list(lever.statement_local_ids),
                location=(
                    f"reported_design_levers[{index}]"
                    ".statement_local_ids"
                ),
                local_id=lever.local_id,
            )
            if not refs:
                _drop_section_object(
                    location=f"reported_design_levers[{index}]",
                    local_id=lever.local_id,
                    reason=(
                        "Reported design lever lost every statement reference "
                        "after unsupported statement removal."
                    ),
                )
                continue
            normalized_levers.append(
                lever.model_copy(update={"statement_local_ids": refs})
            )

        applied_actions = [
            row
            for row in actions
            if row.action
            != "blocked_strong_causal_mechanism"
        ]
        blocked_actions = [
            row
            for row in actions
            if row.action
            == "blocked_strong_causal_mechanism"
        ]

        normalized = draft.model_copy(
            update={
                "statements": normalized_statements,
                "direct_finding_local_ids": normalized_direct_findings,
                "mechanism_routes": normalized_routes,
                "recurring_mechanistic_motifs": normalized_motifs,
                "cross_paper_connections": normalized_connections,
                "evidence_tensions": normalized_tensions,
                "unresolved_connections": normalized_unresolved,
                "reported_design_levers": normalized_levers,
            }
        )
        audit = ExplorerNormalizationAudit(
            domain_profile_id=self.domain_profile.profile_id,
            applied=bool(applied_actions),
            action_count=len(applied_actions),
            blocked_count=len(blocked_actions),
            actions=actions,
        )
        return ExplorerNormalizationResult(
            draft=normalized,
            audit=audit,
        )
