from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dac_her.explorer_contracts import GraphExplorerPacket
from dac_her.hypothesis_contracts import HypothesisContext
from dac_her.path_lineage_diagnostics import (
    StatementPathLineageAssessor,
    StatementPathLineageDiagnostic,
    StatementPathLineageReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathLineagePropagationPolicy(StrictModel):
    propagation_mode: Literal["minimal_deterministic_cover"] = "minimal_deterministic_cover"
    eligible_statements_only: Literal[True] = True
    preserve_existing_explicit_paths: Literal[True] = True
    require_full_scientific_support_cover: Literal[True] = True
    edge_support_preferred_over_node_support: Literal[True] = True
    paper_overlap_alone_never_propagated: Literal[True] = True
    scientific_support_content_changed: Literal[False] = False
    premise_eligibility_changed: Literal[False] = False


class PathCoverCandidate(StrictModel):
    path_ids: list[str] = Field(default_factory=list)
    cardinality: int = 0
    covered_support_ids: list[str] = Field(default_factory=list)
    support_coverage: float = 0.0
    summed_support_overlap_count: int = 0
    mechanism_bearing_count: int = 0
    mechanistic_content_score_sum: int = 0
    navigation_edge_fraction_sum: float = 0.0
    bundle_ranks: list[int] = Field(default_factory=list)


class StatementPathPropagationCard(StrictModel):
    statement_id: str
    eligible_as_premise: bool = False
    action: Literal[
        "propagated_minimal_cover",
        "preserved_existing_explicit",
        "unchanged_ineligible",
        "skipped_no_deterministic_candidate",
        "skipped_incomplete_candidate_union_cover",
        "skipped_empty_scientific_support",
    ]
    cover_basis: Literal["scientific_edges", "scientific_nodes", "none"] = "none"
    cover_universe_ids: list[str] = Field(default_factory=list)
    before_support_path_ids: list[str] = Field(default_factory=list)
    after_support_path_ids: list[str] = Field(default_factory=list)
    propagated_path_ids: list[str] = Field(default_factory=list)
    deterministic_candidate_path_ids: list[str] = Field(default_factory=list)
    deterministic_candidate_count: int = 0
    minimum_cover_size: int | None = None
    minimum_cover_tie_count: int = 0
    selected_cover: PathCoverCandidate | None = None
    scientific_support_node_ids_before: list[str] = Field(default_factory=list)
    scientific_support_node_ids_after: list[str] = Field(default_factory=list)
    scientific_support_edge_ids_before: list[str] = Field(default_factory=list)
    scientific_support_edge_ids_after: list[str] = Field(default_factory=list)
    eligible_as_premise_before: bool = False
    eligible_as_premise_after: bool = False
    scientific_support_unchanged: bool = True
    premise_eligibility_unchanged: bool = True


class PathLineagePropagationReport(StrictModel):
    schema_version: Literal["path-lineage-propagation-report-v1"] = "path-lineage-propagation-report-v1"
    report_id: str
    report_sha256: str
    source_packet_id: str
    source_packet_sha256: str
    source_context_id: str
    source_context_sha256_before: str
    output_context_sha256_after: str
    domain_profile_id: str
    eligible_statement_count: int = 0
    propagated_statement_count: int = 0
    preserved_existing_explicit_statement_count: int = 0
    skipped_statement_count: int = 0
    total_propagated_path_id_count: int = 0
    scientific_support_changed_statement_count: int = 0
    premise_eligibility_changed_statement_count: int = 0
    pre_explicit_path_lineage_statement_count: int = 0
    post_explicit_path_lineage_statement_count: int = 0
    statement_cards: list[StatementPathPropagationCard] = Field(default_factory=list)
    policy: PathLineagePropagationPolicy = Field(default_factory=PathLineagePropagationPolicy)


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _context_rehash(context: HypothesisContext) -> HypothesisContext:
    payload = context.model_dump(mode="json")
    payload.pop("context_sha256", None)
    return context.model_copy(update={"context_sha256": _sha256_json(payload)})


def _mechanistic_content_score(value: str | None) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(str(value or "").strip().lower(), 0)


def _cover_basis(diagnostic: StatementPathLineageDiagnostic) -> tuple[str, set[str]]:
    edges = set(diagnostic.scientific_support_edge_ids)
    if edges:
        return "scientific_edges", edges
    nodes = set(diagnostic.scientific_support_node_ids)
    if nodes:
        return "scientific_nodes", nodes
    return "none", set()


def _candidate_support(
    diagnostic: StatementPathLineageDiagnostic,
    *,
    basis: str,
) -> dict[str, set[str]]:
    support: dict[str, set[str]] = {}
    for overlap in diagnostic.path_overlaps:
        if not overlap.attribution_candidate:
            continue
        values = (
            set(overlap.overlapping_scientific_edge_ids)
            if basis == "scientific_edges"
            else set(overlap.overlapping_scientific_node_ids)
            if basis == "scientific_nodes"
            else set()
        )
        if values:
            support[overlap.path_id] = values
    return support


def _cover_tie_key(rows: tuple[Any, ...], support_map: dict[str, set[str]]) -> tuple[Any, ...]:
    return (
        -sum(len(support_map[row.path_id]) for row in rows),
        -sum(bool(row.mechanism_bearing) for row in rows),
        -sum(_mechanistic_content_score(row.mechanistic_content) for row in rows),
        round(sum(float(row.navigation_edge_fraction) for row in rows), 12),
        tuple(sorted(int(row.bundle_rank) for row in rows)),
        tuple(sorted(row.path_id for row in rows)),
    )


def _cover_card(
    rows: tuple[Any, ...],
    *,
    universe: set[str],
    support_map: dict[str, set[str]],
) -> PathCoverCandidate:
    covered = set().union(*(support_map[row.path_id] for row in rows))
    return PathCoverCandidate(
        path_ids=[
            row.path_id
            for row in sorted(rows, key=lambda value: (value.bundle_rank, value.path_id))
        ],
        cardinality=len(rows),
        covered_support_ids=sorted(covered),
        support_coverage=len(covered) / len(universe) if universe else 0.0,
        summed_support_overlap_count=sum(len(support_map[row.path_id]) for row in rows),
        mechanism_bearing_count=sum(bool(row.mechanism_bearing) for row in rows),
        mechanistic_content_score_sum=sum(
            _mechanistic_content_score(row.mechanistic_content) for row in rows
        ),
        navigation_edge_fraction_sum=float(
            sum(float(row.navigation_edge_fraction) for row in rows)
        ),
        bundle_ranks=sorted(int(row.bundle_rank) for row in rows),
    )


def _minimum_full_cover(
    diagnostic: StatementPathLineageDiagnostic,
) -> tuple[str, set[str], PathCoverCandidate | None, int]:
    basis, universe = _cover_basis(diagnostic)
    if not universe:
        return basis, universe, None, 0

    support_map = _candidate_support(diagnostic, basis=basis)
    row_by_id = {
        row.path_id: row
        for row in diagnostic.path_overlaps
        if row.path_id in support_map
    }
    rows = [
        row_by_id[path_id]
        for path_id in diagnostic.deterministic_attribution_candidate_path_ids
        if path_id in row_by_id
    ]

    union = set().union(*support_map.values()) if support_map else set()
    if not rows or union != universe:
        return basis, universe, None, 0

    for size in range(1, len(rows) + 1):
        full_covers = []
        for combo in itertools.combinations(rows, size):
            covered = set().union(*(support_map[row.path_id] for row in combo))
            if covered == universe:
                full_covers.append(combo)
        if full_covers:
            full_covers.sort(key=lambda combo: _cover_tie_key(combo, support_map))
            chosen = full_covers[0]
            return (
                basis,
                universe,
                _cover_card(chosen, universe=universe, support_map=support_map),
                len(full_covers),
            )
    return basis, universe, None, 0


class MinimalPathLineagePropagator:
    """Fill only missing eligible support_path_ids using an exact minimum cover."""

    def propagate(
        self,
        packet: GraphExplorerPacket,
        context: HypothesisContext,
        *,
        diagnostic: StatementPathLineageReport | None = None,
    ) -> tuple[HypothesisContext, PathLineagePropagationReport]:
        if context.source_packet_id != packet.packet_id:
            raise ValueError("context/packet ID mismatch")
        if context.source_packet_sha256 != packet.packet_sha256:
            raise ValueError("context/packet SHA mismatch")

        diagnostic = diagnostic or StatementPathLineageAssessor().assess(packet, context)
        if diagnostic.source_context_id != context.context_id:
            raise ValueError("diagnostic/context ID mismatch")
        if diagnostic.source_context_sha256 != context.context_sha256:
            raise ValueError("diagnostic/context SHA mismatch")

        diagnostic_by_id = {
            row.statement_id: row for row in diagnostic.statement_diagnostics
        }
        updated_statements = []
        cards: list[StatementPathPropagationCard] = []

        for statement in context.evidence_statements:
            before_nodes = list(statement.scientific_support_node_ids)
            before_edges = list(statement.scientific_support_edge_ids)
            before_paths = list(statement.support_path_ids)
            before_eligible = bool(statement.eligible_as_premise)
            row = diagnostic_by_id[statement.statement_id]

            basis = "none"
            universe: set[str] = set()
            selected_cover = None
            tie_count = 0
            propagated_paths: list[str] = []

            if not before_eligible:
                action = "unchanged_ineligible"
                updated = statement
            elif before_paths:
                action = "preserved_existing_explicit"
                updated = statement
            else:
                basis, universe = _cover_basis(row)
                if not universe:
                    action = "skipped_empty_scientific_support"
                    updated = statement
                elif row.deterministic_attribution_candidate_count == 0:
                    action = "skipped_no_deterministic_candidate"
                    updated = statement
                else:
                    basis, universe, selected_cover, tie_count = _minimum_full_cover(row)
                    if selected_cover is None:
                        action = "skipped_incomplete_candidate_union_cover"
                        updated = statement
                    else:
                        action = "propagated_minimal_cover"
                        propagated_paths = list(selected_cover.path_ids)
                        updated = statement.model_copy(
                            update={"support_path_ids": propagated_paths}
                        )

            after_nodes = list(updated.scientific_support_node_ids)
            after_edges = list(updated.scientific_support_edge_ids)
            after_eligible = bool(updated.eligible_as_premise)

            support_unchanged = before_nodes == after_nodes and before_edges == after_edges
            eligibility_unchanged = before_eligible == after_eligible
            if not support_unchanged:
                raise RuntimeError(
                    f"{statement.statement_id}: PL1-B changed scientific support"
                )
            if not eligibility_unchanged:
                raise RuntimeError(
                    f"{statement.statement_id}: PL1-B changed premise eligibility"
                )

            cards.append(
                StatementPathPropagationCard(
                    statement_id=statement.statement_id,
                    eligible_as_premise=before_eligible,
                    action=action,
                    cover_basis=basis,
                    cover_universe_ids=sorted(universe),
                    before_support_path_ids=before_paths,
                    after_support_path_ids=list(updated.support_path_ids),
                    propagated_path_ids=propagated_paths,
                    deterministic_candidate_path_ids=list(
                        row.deterministic_attribution_candidate_path_ids
                    ),
                    deterministic_candidate_count=row.deterministic_attribution_candidate_count,
                    minimum_cover_size=(
                        selected_cover.cardinality if selected_cover is not None else None
                    ),
                    minimum_cover_tie_count=tie_count,
                    selected_cover=selected_cover,
                    scientific_support_node_ids_before=before_nodes,
                    scientific_support_node_ids_after=after_nodes,
                    scientific_support_edge_ids_before=before_edges,
                    scientific_support_edge_ids_after=after_edges,
                    eligible_as_premise_before=before_eligible,
                    eligible_as_premise_after=after_eligible,
                    scientific_support_unchanged=support_unchanged,
                    premise_eligibility_unchanged=eligibility_unchanged,
                )
            )
            updated_statements.append(updated)

        updated_context = _context_rehash(
            context.model_copy(update={"evidence_statements": updated_statements})
        )
        eligible_cards = [card for card in cards if card.eligible_as_premise]
        propagated_cards = [
            card for card in eligible_cards if card.action == "propagated_minimal_cover"
        ]

        payload = {
            "schema_version": "path-lineage-propagation-report-v1",
            "report_id": _stable_id(
                "path_lineage_propagation_report",
                packet.packet_sha256,
                context.context_sha256,
                updated_context.context_sha256,
            ),
            "source_packet_id": packet.packet_id,
            "source_packet_sha256": packet.packet_sha256,
            "source_context_id": context.context_id,
            "source_context_sha256_before": context.context_sha256,
            "output_context_sha256_after": updated_context.context_sha256,
            "domain_profile_id": context.domain_profile_id,
            "eligible_statement_count": len(eligible_cards),
            "propagated_statement_count": len(propagated_cards),
            "preserved_existing_explicit_statement_count": sum(
                card.action == "preserved_existing_explicit" for card in eligible_cards
            ),
            "skipped_statement_count": sum(
                card.action.startswith("skipped_") for card in eligible_cards
            ),
            "total_propagated_path_id_count": sum(
                len(card.propagated_path_ids) for card in propagated_cards
            ),
            "scientific_support_changed_statement_count": sum(
                not card.scientific_support_unchanged for card in cards
            ),
            "premise_eligibility_changed_statement_count": sum(
                not card.premise_eligibility_unchanged for card in cards
            ),
            "pre_explicit_path_lineage_statement_count": sum(
                bool(card.before_support_path_ids) for card in eligible_cards
            ),
            "post_explicit_path_lineage_statement_count": sum(
                bool(card.after_support_path_ids) for card in eligible_cards
            ),
            "statement_cards": [card.model_dump(mode="json") for card in cards],
            "policy": PathLineagePropagationPolicy().model_dump(mode="json"),
        }
        report = PathLineagePropagationReport(
            **payload,
            report_sha256=_sha256_json(payload),
        )
        return updated_context, report
