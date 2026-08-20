from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.explorer_contracts import GraphExplorerPacket
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathLineageDiagnosticPolicy(StrictModel):
    diagnostic_only: Literal[True] = True
    scientific_selection_changed: Literal[False] = False
    automatic_path_propagation_allowed: Literal[False] = False
    paper_overlap_alone_is_attribution: Literal[False] = False
    node_overlap_with_existing_statement_edges_is_attribution: Literal[False] = False
    missing_explicit_path_lineage_invalidates_premise: Literal[False] = False


class StatementPathOverlap(StrictModel):
    path_id: str
    bundle_rank: int
    path_type: str
    mechanistic_content: str | None = None
    mechanism_bearing: bool = False
    navigation_edge_fraction: float = 0.0

    path_paper_ids: list[str] = Field(default_factory=list)
    statement_path_paper_overlap_ids: list[str] = Field(default_factory=list)

    overlapping_scientific_node_ids: list[str] = Field(default_factory=list)
    overlapping_scientific_edge_ids: list[str] = Field(default_factory=list)
    node_overlap_count: int = 0
    edge_overlap_count: int = 0
    statement_node_coverage: float = 0.0
    statement_edge_coverage: float = 0.0

    relationship: Literal[
        "exact_support_route",
        "edge_supported_partial_route",
        "exact_node_support_route",
        "node_supported_partial_route",
        "node_context_only",
        "paper_context_only",
        "no_scientific_overlap",
    ]
    attribution_candidate: bool = False
    mechanistic_attribution_candidate: bool = False


class StatementPathLineageDiagnostic(StrictModel):
    statement_id: str
    text: str
    epistemic_role: str
    claim_kind: str
    eligible_as_premise: bool = False
    paper_ids: list[str] = Field(default_factory=list)

    scientific_support_node_ids: list[str] = Field(default_factory=list)
    scientific_support_edge_ids: list[str] = Field(default_factory=list)
    scientific_support_node_count: int = 0
    scientific_support_edge_count: int = 0

    explicit_support_path_ids: list[str] = Field(default_factory=list)
    explicit_support_path_count: int = 0
    has_explicit_path_lineage: bool = False

    deterministic_attribution_candidate_path_ids: list[str] = Field(
        default_factory=list
    )
    deterministic_attribution_candidate_count: int = 0
    deterministic_mechanistic_candidate_path_ids: list[str] = Field(
        default_factory=list
    )
    deterministic_mechanistic_candidate_count: int = 0

    candidate_union_overlapping_node_ids: list[str] = Field(default_factory=list)
    candidate_union_overlapping_edge_ids: list[str] = Field(default_factory=list)
    candidate_union_statement_node_coverage: float = 0.0
    candidate_union_statement_edge_coverage: float = 0.0
    candidate_union_covers_all_statement_nodes: bool = False
    candidate_union_covers_all_statement_edges: bool = False

    unattributed_scientific_node_ids: list[str] = Field(default_factory=list)
    unattributed_scientific_edge_ids: list[str] = Field(default_factory=list)

    deterministic_candidate_ids_not_explicit: list[str] = Field(default_factory=list)
    explicit_path_ids_not_deterministically_attributable: list[str] = Field(
        default_factory=list
    )

    missing_explicit_lineage_but_recoverable: bool = False
    missing_explicit_lineage_and_unrecoverable: bool = False

    path_overlaps: list[StatementPathOverlap] = Field(default_factory=list)
    diagnostic_flags: list[str] = Field(default_factory=list)


class PathUsageDiagnostic(StrictModel):
    path_id: str
    bundle_rank: int
    path_type: str
    mechanism_bearing: bool = False
    eligible_statement_ids: list[str] = Field(default_factory=list)
    eligible_statement_count: int = 0


class StatementPathLineageReport(StrictModel):
    schema_version: Literal["statement-path-lineage-report-v1"] = (
        "statement-path-lineage-report-v1"
    )
    report_id: str
    report_sha256: str

    source_packet_id: str
    source_packet_sha256: str
    source_context_id: str
    source_context_sha256: str
    domain_profile_id: str

    selected_path_count: int = 0
    selected_mechanistic_path_count: int = 0

    eligible_statement_count: int = 0
    eligible_with_explicit_path_lineage_count: int = 0
    eligible_without_explicit_path_lineage_count: int = 0

    eligible_with_deterministic_attribution_count: int = 0
    eligible_with_deterministic_mechanistic_attribution_count: int = 0

    recoverable_missing_explicit_path_lineage_count: int = 0
    unrecoverable_missing_explicit_path_lineage_count: int = 0

    eligible_candidate_union_full_edge_coverage_count: int = 0
    eligible_candidate_union_full_node_coverage_count: int = 0

    eligible_statement_deterministic_attribution_fraction: float = 0.0
    eligible_statement_mechanistic_attribution_fraction: float = 0.0

    noneligible_statement_count: int = 0
    noneligible_with_explicit_path_lineage_count: int = 0

    selected_paths_attributable_to_eligible_statement_count: int = 0
    selected_paths_not_attributable_to_any_eligible_statement_ids: list[str] = Field(
        default_factory=list
    )
    selected_mechanistic_paths_not_attributable_to_any_eligible_statement_ids: list[
        str
    ] = Field(default_factory=list)

    relationship_counts: dict[str, int] = Field(default_factory=dict)
    statement_diagnostics: list[StatementPathLineageDiagnostic] = Field(
        default_factory=list
    )
    path_usage: list[PathUsageDiagnostic] = Field(default_factory=list)

    policy: PathLineageDiagnosticPolicy = Field(
        default_factory=PathLineageDiagnosticPolicy
    )


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _sorted_unique(values: Any) -> list[str]:
    return sorted(
        {
            str(value)
            for value in values
            if str(value).strip()
        }
    )


def _path_paper_ids(path: Any) -> list[str]:
    return _sorted_unique(
        list(getattr(path, "supporting_paper_ids", []))
        + list(getattr(path, "visited_paper_ids", []))
    )


def _path_edge_ids(path: Any) -> set[str]:
    return {
        str(step.selected_original_edge_id)
        for step in path.steps
        if str(step.selected_original_edge_id).strip()
    }


def _path_node_ids(
    packet: GraphExplorerPacket,
    path: Any,
) -> set[str]:
    catalog_nodes = packet.evidence_catalog.nodes
    nodes = {
        str(node_id)
        for node_id in path.node_ids
        if str(node_id) in catalog_nodes
    }

    for step in path.steps:
        for node_id in (
            step.scientific_source,
            step.scientific_target,
        ):
            node_id = str(node_id)
            if node_id in catalog_nodes:
                nodes.add(node_id)

        edge = packet.evidence_catalog.edges.get(
            str(step.selected_original_edge_id)
        )
        if edge is not None:
            nodes.update(
                node_id
                for node_id in edge.supporting_node_ids
                if node_id in catalog_nodes
            )
    return nodes


def _mechanism_bearing(path: Any) -> bool:
    path_type = str(path.quality.path_type or "").strip()
    structural = str(
        path.quality.path_structure_type or ""
    ).strip()
    return bool(
        path.quality.mechanism_bearing
        or path_type
        in {
            "DIRECT_MECHANISTIC",
            "CROSS_PAPER_MECHANISTIC",
        }
        or structural
        in {
            "DIRECT_MECHANISTIC",
            "CROSS_PAPER_MECHANISTIC",
        }
    )


def _coverage(
    overlap: set[str],
    support: set[str],
) -> float:
    if not support:
        return 1.0
    return len(overlap) / len(support)


def _relationship(
    *,
    statement_nodes: set[str],
    statement_edges: set[str],
    overlap_nodes: set[str],
    overlap_edges: set[str],
    paper_overlap: set[str],
) -> tuple[str, bool]:
    if statement_edges:
        if overlap_edges:
            edge_complete = overlap_edges == statement_edges
            node_complete = (
                not statement_nodes
                or overlap_nodes == statement_nodes
            )
            if edge_complete and node_complete:
                return "exact_support_route", True
            return "edge_supported_partial_route", True
        if overlap_nodes:
            return "node_context_only", False
        if paper_overlap:
            return "paper_context_only", False
        return "no_scientific_overlap", False

    if statement_nodes:
        if overlap_nodes:
            if overlap_nodes == statement_nodes:
                return "exact_node_support_route", True
            return "node_supported_partial_route", True
        if paper_overlap:
            return "paper_context_only", False
        return "no_scientific_overlap", False

    if paper_overlap:
        return "paper_context_only", False
    return "no_scientific_overlap", False


class StatementPathLineageAssessor:
    """PL1-A diagnostic-only statement-to-selected-path attribution analysis.

    Attribution is conservative:
    - if a statement has scientific edge support, at least one edge overlap is
      required;
    - node-only overlap does not attribute a statement that already has edges;
    - paper overlap alone never attributes a scientific statement.
    """

    def assess(
        self,
        packet: GraphExplorerPacket,
        context: HypothesisContext,
    ) -> StatementPathLineageReport:
        if context.source_packet_id != packet.packet_id:
            raise ValueError("context/packet ID mismatch")
        if context.source_packet_sha256 != packet.packet_sha256:
            raise ValueError("context/packet SHA mismatch")
        if context.domain_profile_id != packet.domain_profile_id:
            raise ValueError(
                "context/packet domain profile mismatch"
            )

        packet_path_ids = {
            path.path_id
            for path in packet.paths
        }
        catalog_node_ids = set(
            packet.evidence_catalog.nodes
        )
        catalog_edge_ids = set(
            packet.evidence_catalog.edges
        )

        path_nodes = {
            path.path_id: _path_node_ids(
                packet,
                path,
            )
            for path in packet.paths
        }
        path_edges = {
            path.path_id: _path_edge_ids(path)
            for path in packet.paths
        }

        path_statement_usage: dict[
            str,
            list[str],
        ] = {
            path.path_id: []
            for path in packet.paths
        }
        relationship_counter: Counter[str] = Counter()

        diagnostics: list[
            StatementPathLineageDiagnostic
        ] = []

        for statement in context.evidence_statements:
            statement_nodes = set(
                map(
                    str,
                    statement.scientific_support_node_ids,
                )
            )
            statement_edges = set(
                map(
                    str,
                    statement.scientific_support_edge_ids,
                )
            )
            explicit_paths = set(
                map(
                    str,
                    statement.support_path_ids,
                )
            )

            unknown_nodes = sorted(
                statement_nodes - catalog_node_ids
            )
            unknown_edges = sorted(
                statement_edges - catalog_edge_ids
            )
            unknown_paths = sorted(
                explicit_paths - packet_path_ids
            )
            if unknown_nodes:
                raise ValueError(
                    f"{statement.statement_id}: scientific support nodes "
                    f"missing from packet: {unknown_nodes}"
                )
            if unknown_edges:
                raise ValueError(
                    f"{statement.statement_id}: scientific support edges "
                    f"missing from packet: {unknown_edges}"
                )
            if unknown_paths:
                raise ValueError(
                    f"{statement.statement_id}: explicit support paths "
                    f"missing from packet: {unknown_paths}"
                )

            statement_papers = set(
                map(str, statement.paper_ids)
            )
            overlaps: list[
                StatementPathOverlap
            ] = []

            for path in packet.paths:
                path_id = path.path_id
                overlap_nodes = (
                    statement_nodes
                    & path_nodes[path_id]
                )
                overlap_edges = (
                    statement_edges
                    & path_edges[path_id]
                )
                paper_overlap = (
                    statement_papers
                    & set(_path_paper_ids(path))
                )

                relationship, candidate = (
                    _relationship(
                        statement_nodes=statement_nodes,
                        statement_edges=statement_edges,
                        overlap_nodes=overlap_nodes,
                        overlap_edges=overlap_edges,
                        paper_overlap=paper_overlap,
                    )
                )
                mechanism = _mechanism_bearing(path)
                mechanistic_candidate = (
                    candidate
                    and mechanism
                )

                relationship_counter[
                    relationship
                ] += 1

                overlaps.append(
                    StatementPathOverlap(
                        path_id=path_id,
                        bundle_rank=path.bundle_rank,
                        path_type=str(
                            path.quality.path_structure_type
                            or path.quality.path_type
                            or "UNKNOWN"
                        ),
                        mechanistic_content=(
                            path.quality.mechanistic_content
                        ),
                        mechanism_bearing=mechanism,
                        navigation_edge_fraction=float(
                            path.quality.navigation_edge_fraction
                        ),
                        path_paper_ids=_path_paper_ids(
                            path
                        ),
                        statement_path_paper_overlap_ids=sorted(
                            paper_overlap
                        ),
                        overlapping_scientific_node_ids=sorted(
                            overlap_nodes
                        ),
                        overlapping_scientific_edge_ids=sorted(
                            overlap_edges
                        ),
                        node_overlap_count=len(
                            overlap_nodes
                        ),
                        edge_overlap_count=len(
                            overlap_edges
                        ),
                        statement_node_coverage=_coverage(
                            overlap_nodes,
                            statement_nodes,
                        ),
                        statement_edge_coverage=_coverage(
                            overlap_edges,
                            statement_edges,
                        ),
                        relationship=relationship,
                        attribution_candidate=(
                            candidate
                        ),
                        mechanistic_attribution_candidate=(
                            mechanistic_candidate
                        ),
                    )
                )

            relationship_rank = {
                "exact_support_route": 0,
                "edge_supported_partial_route": 1,
                "exact_node_support_route": 2,
                "node_supported_partial_route": 3,
                "node_context_only": 4,
                "paper_context_only": 5,
                "no_scientific_overlap": 6,
            }
            overlaps.sort(
                key=lambda row: (
                    not row.attribution_candidate,
                    relationship_rank[
                        row.relationship
                    ],
                    -row.statement_edge_coverage,
                    -row.statement_node_coverage,
                    not row.mechanism_bearing,
                    row.bundle_rank,
                    row.path_id,
                )
            )

            candidates = [
                row
                for row in overlaps
                if row.attribution_candidate
            ]
            mechanistic_candidates = [
                row
                for row in candidates
                if row.mechanistic_attribution_candidate
            ]

            candidate_ids = [
                row.path_id
                for row in candidates
            ]
            mechanistic_candidate_ids = [
                row.path_id
                for row in mechanistic_candidates
            ]

            union_nodes = {
                node_id
                for row in candidates
                for node_id in row.overlapping_scientific_node_ids
            }
            union_edges = {
                edge_id
                for row in candidates
                for edge_id in row.overlapping_scientific_edge_ids
            }

            if statement.eligible_as_premise:
                for path_id in candidate_ids:
                    path_statement_usage[
                        path_id
                    ].append(
                        statement.statement_id
                    )

            flags: list[str] = []
            has_explicit = bool(
                explicit_paths
            )
            has_candidate = bool(
                candidates
            )

            if (
                statement.eligible_as_premise
                and not has_explicit
                and has_candidate
            ):
                flags.append(
                    "missing_explicit_path_lineage_but_deterministically_recoverable"
                )
            if (
                statement.eligible_as_premise
                and not has_explicit
                and not has_candidate
            ):
                flags.append(
                    "missing_explicit_path_lineage_and_no_deterministic_attribution"
                )
            if (
                explicit_paths
                - set(candidate_ids)
            ):
                flags.append(
                    "explicit_path_not_supported_by_conservative_overlap_rule"
                )
            if (
                set(candidate_ids)
                - explicit_paths
            ):
                flags.append(
                    "deterministic_candidate_not_explicitly_linked"
                )

            diagnostics.append(
                StatementPathLineageDiagnostic(
                    statement_id=statement.statement_id,
                    text=statement.text,
                    epistemic_role=statement.epistemic_role,
                    claim_kind=statement.claim_kind,
                    eligible_as_premise=(
                        statement.eligible_as_premise
                    ),
                    paper_ids=list(
                        statement.paper_ids
                    ),
                    scientific_support_node_ids=sorted(
                        statement_nodes
                    ),
                    scientific_support_edge_ids=sorted(
                        statement_edges
                    ),
                    scientific_support_node_count=len(
                        statement_nodes
                    ),
                    scientific_support_edge_count=len(
                        statement_edges
                    ),
                    explicit_support_path_ids=sorted(
                        explicit_paths
                    ),
                    explicit_support_path_count=len(
                        explicit_paths
                    ),
                    has_explicit_path_lineage=(
                        has_explicit
                    ),
                    deterministic_attribution_candidate_path_ids=(
                        candidate_ids
                    ),
                    deterministic_attribution_candidate_count=len(
                        candidates
                    ),
                    deterministic_mechanistic_candidate_path_ids=(
                        mechanistic_candidate_ids
                    ),
                    deterministic_mechanistic_candidate_count=len(
                        mechanistic_candidates
                    ),
                    candidate_union_overlapping_node_ids=sorted(
                        union_nodes
                    ),
                    candidate_union_overlapping_edge_ids=sorted(
                        union_edges
                    ),
                    candidate_union_statement_node_coverage=_coverage(
                        union_nodes,
                        statement_nodes,
                    ),
                    candidate_union_statement_edge_coverage=_coverage(
                        union_edges,
                        statement_edges,
                    ),
                    candidate_union_covers_all_statement_nodes=(
                        not statement_nodes
                        or union_nodes == statement_nodes
                    ),
                    candidate_union_covers_all_statement_edges=(
                        not statement_edges
                        or union_edges == statement_edges
                    ),
                    unattributed_scientific_node_ids=sorted(
                        statement_nodes
                        - union_nodes
                    ),
                    unattributed_scientific_edge_ids=sorted(
                        statement_edges
                        - union_edges
                    ),
                    deterministic_candidate_ids_not_explicit=sorted(
                        set(candidate_ids)
                        - explicit_paths
                    ),
                    explicit_path_ids_not_deterministically_attributable=sorted(
                        explicit_paths
                        - set(candidate_ids)
                    ),
                    missing_explicit_lineage_but_recoverable=(
                        statement.eligible_as_premise
                        and not has_explicit
                        and has_candidate
                    ),
                    missing_explicit_lineage_and_unrecoverable=(
                        statement.eligible_as_premise
                        and not has_explicit
                        and not has_candidate
                    ),
                    path_overlaps=overlaps,
                    diagnostic_flags=sorted(
                        set(flags)
                    ),
                )
            )

        eligible = [
            row
            for row in diagnostics
            if row.eligible_as_premise
        ]
        noneligible = [
            row
            for row in diagnostics
            if not row.eligible_as_premise
        ]

        selected_mechanistic = [
            path
            for path in packet.paths
            if _mechanism_bearing(path)
        ]

        path_usage: list[
            PathUsageDiagnostic
        ] = []
        for path in sorted(
            packet.paths,
            key=lambda value: (
                value.bundle_rank,
                value.path_id,
            ),
        ):
            statement_ids = sorted(
                set(
                    path_statement_usage[
                        path.path_id
                    ]
                )
            )
            path_usage.append(
                PathUsageDiagnostic(
                    path_id=path.path_id,
                    bundle_rank=path.bundle_rank,
                    path_type=str(
                        path.quality.path_structure_type
                        or path.quality.path_type
                        or "UNKNOWN"
                    ),
                    mechanism_bearing=(
                        _mechanism_bearing(path)
                    ),
                    eligible_statement_ids=(
                        statement_ids
                    ),
                    eligible_statement_count=len(
                        statement_ids
                    ),
                )
            )

        attributable_path_ids = {
            row.path_id
            for row in path_usage
            if row.eligible_statement_count > 0
        }
        selected_path_ids = {
            path.path_id
            for path in packet.paths
        }
        selected_mechanistic_ids = {
            path.path_id
            for path in selected_mechanistic
        }

        report_id = _stable_id(
            "statement_path_lineage_report",
            packet.packet_sha256,
            context.context_id,
            context.context_sha256,
        )

        payload = {
            "schema_version": "statement-path-lineage-report-v1",
            "report_id": report_id,
            "source_packet_id": packet.packet_id,
            "source_packet_sha256": packet.packet_sha256,
            "source_context_id": context.context_id,
            "source_context_sha256": context.context_sha256,
            "domain_profile_id": context.domain_profile_id,
            "selected_path_count": len(
                packet.paths
            ),
            "selected_mechanistic_path_count": len(
                selected_mechanistic
            ),
            "eligible_statement_count": len(
                eligible
            ),
            "eligible_with_explicit_path_lineage_count": sum(
                row.has_explicit_path_lineage
                for row in eligible
            ),
            "eligible_without_explicit_path_lineage_count": sum(
                not row.has_explicit_path_lineage
                for row in eligible
            ),
            "eligible_with_deterministic_attribution_count": sum(
                row.deterministic_attribution_candidate_count
                > 0
                for row in eligible
            ),
            "eligible_with_deterministic_mechanistic_attribution_count": sum(
                row.deterministic_mechanistic_candidate_count
                > 0
                for row in eligible
            ),
            "recoverable_missing_explicit_path_lineage_count": sum(
                row.missing_explicit_lineage_but_recoverable
                for row in eligible
            ),
            "unrecoverable_missing_explicit_path_lineage_count": sum(
                row.missing_explicit_lineage_and_unrecoverable
                for row in eligible
            ),
            "eligible_candidate_union_full_edge_coverage_count": sum(
                row.candidate_union_covers_all_statement_edges
                for row in eligible
            ),
            "eligible_candidate_union_full_node_coverage_count": sum(
                row.candidate_union_covers_all_statement_nodes
                for row in eligible
            ),
            "eligible_statement_deterministic_attribution_fraction": (
                sum(
                    row.deterministic_attribution_candidate_count
                    > 0
                    for row in eligible
                )
                / len(eligible)
                if eligible
                else 0.0
            ),
            "eligible_statement_mechanistic_attribution_fraction": (
                sum(
                    row.deterministic_mechanistic_candidate_count
                    > 0
                    for row in eligible
                )
                / len(eligible)
                if eligible
                else 0.0
            ),
            "noneligible_statement_count": len(
                noneligible
            ),
            "noneligible_with_explicit_path_lineage_count": sum(
                row.has_explicit_path_lineage
                for row in noneligible
            ),
            "selected_paths_attributable_to_eligible_statement_count": len(
                attributable_path_ids
            ),
            "selected_paths_not_attributable_to_any_eligible_statement_ids": sorted(
                selected_path_ids
                - attributable_path_ids
            ),
            "selected_mechanistic_paths_not_attributable_to_any_eligible_statement_ids": sorted(
                selected_mechanistic_ids
                - attributable_path_ids
            ),
            "relationship_counts": dict(
                sorted(
                    relationship_counter.items()
                )
            ),
            "statement_diagnostics": [
                row.model_dump(mode="json")
                for row in diagnostics
            ],
            "path_usage": [
                row.model_dump(mode="json")
                for row in path_usage
            ],
            "policy": PathLineageDiagnosticPolicy().model_dump(
                mode="json"
            ),
        }

        return StatementPathLineageReport(
            **payload,
            report_sha256=_sha256_json(payload),
        )
