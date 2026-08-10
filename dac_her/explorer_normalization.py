from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dac_her.discovery_semantics import (
    contains_strong_causal_language,
    is_mechanism_edge,
    is_mechanism_node,
)
from dac_her.domains import get_domain_profile
from dac_her.explorer_contracts import GraphExplorerPacket
from dac_her.explorer_draft import ExplorationDraft


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExplorerNormalizationAction(_StrictModel):
    action: Literal[
        "downgrade_mechanism_to_association",
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

    def normalize(
        self,
        packet: GraphExplorerPacket,
        draft: ExplorationDraft,
    ) -> ExplorerNormalizationResult:
        profile = get_domain_profile(
            packet.domain_profile_id
        )
        semantics = profile.discovery
        actions: list[ExplorerNormalizationAction] = []

        normalized_statements = []
        for index, statement in enumerate(draft.statements):
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
                    actions.append(
                        ExplorerNormalizationAction(
                            action="blocked_strong_causal_mechanism",
                            location=f"statements[{index}]",
                            local_id=statement.local_id,
                            reason=(
                                "Mechanism claim lacks mechanism-bearing "
                                "support, but the text uses strong causal "
                                "language. Automatic downgrading is blocked "
                                "so strict validation can reject it."
                            ),
                        )
                    )
                    normalized_statements.append(statement)
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
        normalized_motifs = []
        for index, motif in enumerate(
            draft.recurring_mechanistic_motifs
        ):
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
                "recurring_mechanistic_motifs": (
                    normalized_motifs
                ),
            }
        )
        audit = ExplorerNormalizationAudit(
            domain_profile_id=profile.profile_id,
            applied=bool(applied_actions),
            action_count=len(applied_actions),
            blocked_count=len(blocked_actions),
            actions=actions,
        )
        return ExplorerNormalizationResult(
            draft=normalized,
            audit=audit,
        )
