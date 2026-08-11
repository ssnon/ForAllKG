from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dac_her.explorer_contracts import (
    ExplorationReport,
    GraphExplorerPacket,
)
from dac_her.hypothesis_contracts import HypothesisContext


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceCompressionPolicy(StrictModel):
    diagnostic_only: Literal[True] = True
    scientific_selection_changed: Literal[False] = False
    multi_paper_statement_implies_overcompression: Literal[False] = False
    structural_profile_diversity_implies_semantic_distinctness: Literal[False] = False
    requires_semantic_interpretation_before_decomposition: Literal[True] = True


class PaperEvidenceContribution(StrictModel):
    paper_id: str
    declared_in_statement: bool = False
    direct_support_node_ids: list[str] = Field(default_factory=list)
    direct_support_edge_ids: list[str] = Field(default_factory=list)
    support_path_ids: list[str] = Field(default_factory=list)
    node_types: list[str] = Field(default_factory=list)
    edge_relations: list[str] = Field(default_factory=list)
    path_structure_types: list[str] = Field(default_factory=list)
    direct_support_unit_count: int = 0
    has_direct_scientific_support: bool = False


class EvidenceCompressionStatementCard(StrictModel):
    statement_id: str
    text: str
    epistemic_role: str
    claim_kind: str
    eligible_as_premise: bool = False
    premise_restrictions: list[str] = Field(default_factory=list)

    paper_ids: list[str] = Field(default_factory=list)
    paper_count: int = 0
    multi_paper: bool = False

    scientific_support_node_ids: list[str] = Field(default_factory=list)
    scientific_support_edge_ids: list[str] = Field(default_factory=list)
    support_path_ids: list[str] = Field(default_factory=list)
    alignment_path_ids: list[str] = Field(default_factory=list)

    scientific_support_node_count: int = 0
    scientific_support_edge_count: int = 0
    support_path_count: int = 0
    alignment_path_count: int = 0

    direct_scientific_support_paper_ids: list[str] = Field(default_factory=list)
    path_context_paper_ids: list[str] = Field(default_factory=list)
    declared_without_direct_scientific_support_paper_ids: list[str] = Field(
        default_factory=list
    )
    direct_support_without_declared_paper_ids: list[str] = Field(
        default_factory=list
    )

    paper_contributions: list[PaperEvidenceContribution] = Field(
        default_factory=list
    )
    distinct_structural_support_profile_count: int = 0
    all_declared_papers_have_direct_scientific_support: bool = False
    diagnostic_flags: list[str] = Field(default_factory=list)


class PaperStatementIncidence(StrictModel):
    paper_id: str
    eligible_statement_ids: list[str] = Field(default_factory=list)
    eligible_statement_count: int = 0
    has_single_paper_statement: bool = False
    only_multi_paper_statements: bool = False


class PaperIncidenceGroup(StrictModel):
    eligible_statement_ids: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)


class EvidenceCompressionReport(StrictModel):
    schema_version: Literal["evidence-compression-report-v1"] = (
        "evidence-compression-report-v1"
    )
    report_id: str
    report_sha256: str

    source_packet_id: str
    source_packet_sha256: str
    source_report_id: str
    source_report_sha256: str
    source_context_id: str
    source_context_sha256: str
    domain_profile_id: str

    selected_path_paper_ids: list[str] = Field(default_factory=list)
    explorer_statement_paper_ids: list[str] = Field(default_factory=list)
    context_statement_paper_ids: list[str] = Field(default_factory=list)
    eligible_premise_paper_ids: list[str] = Field(default_factory=list)

    selected_path_paper_count: int = 0
    explorer_statement_paper_count: int = 0
    context_statement_paper_count: int = 0
    eligible_premise_paper_count: int = 0

    explorer_statement_count: int = 0
    context_statement_count: int = 0
    eligible_statement_count: int = 0

    eligible_single_paper_statement_count: int = 0
    eligible_multi_paper_statement_count: int = 0
    eligible_multi_paper_statement_fraction: float = 0.0
    eligible_statement_paper_incidence_count: int = 0
    mean_papers_per_eligible_statement: float = 0.0
    eligible_papers_per_statement_ratio: float = 0.0

    eligible_papers_with_single_paper_statement_ids: list[str] = Field(
        default_factory=list
    )
    eligible_papers_only_in_multi_paper_statements_ids: list[str] = Field(
        default_factory=list
    )
    eligible_papers_with_single_paper_statement_count: int = 0
    eligible_papers_only_in_multi_paper_statements_count: int = 0
    eligible_papers_only_in_multi_paper_statements_fraction: float = 0.0

    eligible_multi_paper_reported_statement_count: int = 0
    eligible_multi_paper_synthesis_statement_count: int = 0
    eligible_multi_paper_heterogeneous_profile_count: int = 0
    statements_with_declared_support_mismatch_count: int = 0

    paper_statement_incidence: list[PaperStatementIncidence] = Field(
        default_factory=list
    )
    repeated_paper_incidence_groups: list[PaperIncidenceGroup] = Field(
        default_factory=list
    )
    statement_cards: list[EvidenceCompressionStatementCard] = Field(
        default_factory=list
    )

    policy: EvidenceCompressionPolicy = Field(
        default_factory=EvidenceCompressionPolicy
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


def _node_paper_ids(node: Any) -> list[str]:
    values: list[str] = []
    direct = str(getattr(node, "source_paper_id", "") or "").strip()
    if direct:
        values.append(direct)
    values.extend(
        str(value)
        for value in getattr(node, "source_paper_ids", [])
        if str(value).strip()
    )
    return _sorted_unique(values)


def _edge_paper_ids(edge: Any) -> list[str]:
    return _sorted_unique(
        getattr(edge, "source_paper_ids", [])
    )


def _path_paper_ids(path: Any) -> list[str]:
    return _sorted_unique(
        list(getattr(path, "supporting_paper_ids", []))
        + list(getattr(path, "visited_paper_ids", []))
    )


def _path_structure_type(path: Any) -> str:
    quality = getattr(path, "quality", None)
    if quality is None:
        return ""
    value = str(
        getattr(quality, "path_structure_type", "")
        or getattr(quality, "path_type", "")
        or ""
    ).strip()
    return value


class EvidenceCompressionAssessor:
    """Diagnose Explorer -> HypothesisContext evidence compression.

    EC1 never changes Explorer statements, premise eligibility, hypothesis
    generation, ranking, novelty, semantic review, or feasibility.
    """

    def assess(
        self,
        packet: GraphExplorerPacket,
        report: ExplorationReport,
        context: HypothesisContext,
    ) -> EvidenceCompressionReport:
        if report.source_packet_sha256 != packet.packet_sha256:
            raise ValueError("report/packet SHA mismatch")
        if context.source_packet_id != packet.packet_id:
            raise ValueError("context/packet ID mismatch")
        if context.source_packet_sha256 != packet.packet_sha256:
            raise ValueError("context/packet SHA mismatch")
        if context.source_report_id != report.report_id:
            raise ValueError("context/report ID mismatch")

        report_sha = _sha256_json(report)
        if context.source_report_sha256 != report_sha:
            raise ValueError("context/report SHA mismatch")
        if context.domain_profile_id != packet.domain_profile_id:
            raise ValueError("context/packet domain profile mismatch")

        report_by_id = {
            row.statement_id: row
            for row in report.statements
        }
        context_by_id = {
            row.statement_id: row
            for row in context.evidence_statements
        }
        if set(report_by_id) != set(context_by_id):
            raise ValueError(
                "ExplorerReport/HypothesisContext statement ID mismatch"
            )

        nodes = packet.evidence_catalog.nodes
        edges = packet.evidence_catalog.edges
        paths = {
            path.path_id: path
            for path in packet.paths
        }

        selected_path_papers: set[str] = set()
        for path in packet.paths:
            selected_path_papers.update(
                _path_paper_ids(path)
            )

        statement_cards: list[
            EvidenceCompressionStatementCard
        ] = []

        for statement in context.evidence_statements:
            report_statement = report_by_id[statement.statement_id]
            declared_papers = set(
                _sorted_unique(statement.paper_ids)
            )

            direct_node_papers: set[str] = set()
            direct_edge_papers: set[str] = set()

            for node_id in statement.scientific_support_node_ids:
                node = nodes.get(node_id)
                if node is not None:
                    direct_node_papers.update(
                        _node_paper_ids(node)
                    )

            for edge_id in statement.scientific_support_edge_ids:
                edge = edges.get(edge_id)
                if edge is not None:
                    direct_edge_papers.update(
                        _edge_paper_ids(edge)
                    )

            direct_support_papers = (
                direct_node_papers
                | direct_edge_papers
            )

            path_context_papers: set[str] = set()
            for path_id in statement.support_path_ids:
                path = paths.get(path_id)
                if path is not None:
                    path_context_papers.update(
                        _path_paper_ids(path)
                    )

            contribution_papers = sorted(
                declared_papers
                | direct_support_papers
                | path_context_papers
            )

            contributions: list[
                PaperEvidenceContribution
            ] = []
            profile_signatures: set[
                tuple[
                    tuple[str, ...],
                    tuple[str, ...],
                    tuple[str, ...],
                ]
            ] = set()

            for paper_id in contribution_papers:
                support_node_ids = []
                node_types = set()
                for node_id in statement.scientific_support_node_ids:
                    node = nodes.get(node_id)
                    if (
                        node is not None
                        and paper_id in _node_paper_ids(node)
                    ):
                        support_node_ids.append(
                            str(node_id)
                        )
                        node_types.add(
                            str(node.node_type)
                        )

                support_edge_ids = []
                edge_relations = set()
                for edge_id in statement.scientific_support_edge_ids:
                    edge = edges.get(edge_id)
                    if (
                        edge is not None
                        and paper_id in _edge_paper_ids(edge)
                    ):
                        support_edge_ids.append(
                            str(edge_id)
                        )
                        edge_relations.add(
                            str(edge.relation)
                        )

                support_path_ids = []
                path_types = set()
                for path_id in statement.support_path_ids:
                    path = paths.get(path_id)
                    if (
                        path is not None
                        and paper_id in _path_paper_ids(path)
                    ):
                        support_path_ids.append(
                            str(path_id)
                        )
                        path_type = _path_structure_type(path)
                        if path_type:
                            path_types.add(path_type)

                node_type_list = sorted(node_types)
                relation_list = sorted(edge_relations)
                path_type_list = sorted(path_types)
                direct_unit_count = (
                    len(support_node_ids)
                    + len(support_edge_ids)
                )

                contributions.append(
                    PaperEvidenceContribution(
                        paper_id=paper_id,
                        declared_in_statement=(
                            paper_id in declared_papers
                        ),
                        direct_support_node_ids=sorted(
                            support_node_ids
                        ),
                        direct_support_edge_ids=sorted(
                            support_edge_ids
                        ),
                        support_path_ids=sorted(
                            support_path_ids
                        ),
                        node_types=node_type_list,
                        edge_relations=relation_list,
                        path_structure_types=path_type_list,
                        direct_support_unit_count=(
                            direct_unit_count
                        ),
                        has_direct_scientific_support=(
                            direct_unit_count > 0
                        ),
                    )
                )

                if (
                    paper_id in declared_papers
                    and (
                        node_type_list
                        or relation_list
                        or path_type_list
                    )
                ):
                    profile_signatures.add(
                        (
                            tuple(node_type_list),
                            tuple(relation_list),
                            tuple(path_type_list),
                        )
                    )

            declared_without_direct = sorted(
                declared_papers
                - direct_support_papers
            )
            direct_without_declared = sorted(
                direct_support_papers
                - declared_papers
            )

            flags: list[str] = []
            paper_count = len(declared_papers)
            if paper_count > 1:
                flags.append("multi_paper_statement")
                if statement.epistemic_role == "reported":
                    flags.append(
                        "multi_paper_reported_statement"
                    )
                if statement.epistemic_role == "evidence_synthesis":
                    flags.append(
                        "multi_paper_synthesis_statement"
                    )
            if len(profile_signatures) > 1:
                flags.append(
                    "heterogeneous_structural_support_profiles"
                )
            if declared_without_direct:
                flags.append(
                    "declared_paper_without_direct_scientific_support"
                )
            if direct_without_declared:
                flags.append(
                    "direct_scientific_support_paper_not_declared"
                )

            # Report and context are expected to preserve statement-level
            # declared paper provenance exactly.
            if set(map(str, report_statement.paper_ids)) != declared_papers:
                flags.append(
                    "report_context_paper_provenance_mismatch"
                )

            statement_cards.append(
                EvidenceCompressionStatementCard(
                    statement_id=statement.statement_id,
                    text=statement.text,
                    epistemic_role=statement.epistemic_role,
                    claim_kind=statement.claim_kind,
                    eligible_as_premise=(
                        statement.eligible_as_premise
                    ),
                    premise_restrictions=list(
                        statement.premise_restrictions
                    ),
                    paper_ids=sorted(declared_papers),
                    paper_count=paper_count,
                    multi_paper=paper_count > 1,
                    scientific_support_node_ids=list(
                        statement.scientific_support_node_ids
                    ),
                    scientific_support_edge_ids=list(
                        statement.scientific_support_edge_ids
                    ),
                    support_path_ids=list(
                        statement.support_path_ids
                    ),
                    alignment_path_ids=list(
                        statement.alignment_path_ids
                    ),
                    scientific_support_node_count=len(
                        statement.scientific_support_node_ids
                    ),
                    scientific_support_edge_count=len(
                        statement.scientific_support_edge_ids
                    ),
                    support_path_count=len(
                        statement.support_path_ids
                    ),
                    alignment_path_count=len(
                        statement.alignment_path_ids
                    ),
                    direct_scientific_support_paper_ids=sorted(
                        direct_support_papers
                    ),
                    path_context_paper_ids=sorted(
                        path_context_papers
                    ),
                    declared_without_direct_scientific_support_paper_ids=(
                        declared_without_direct
                    ),
                    direct_support_without_declared_paper_ids=(
                        direct_without_declared
                    ),
                    paper_contributions=contributions,
                    distinct_structural_support_profile_count=len(
                        profile_signatures
                    ),
                    all_declared_papers_have_direct_scientific_support=(
                        not declared_without_direct
                    ),
                    diagnostic_flags=sorted(set(flags)),
                )
            )

        explorer_papers = sorted(
            {
                str(paper_id)
                for statement in report.statements
                for paper_id in statement.paper_ids
                if str(paper_id).strip()
            }
        )
        context_papers = sorted(
            {
                paper_id
                for card in statement_cards
                for paper_id in card.paper_ids
            }
        )
        eligible_cards = [
            card
            for card in statement_cards
            if card.eligible_as_premise
        ]
        eligible_papers = sorted(
            {
                paper_id
                for card in eligible_cards
                for paper_id in card.paper_ids
            }
        )

        paper_to_eligible_statements: dict[
            str,
            list[str],
        ] = defaultdict(list)
        single_paper_statement_papers: set[str] = set()

        for card in eligible_cards:
            for paper_id in card.paper_ids:
                paper_to_eligible_statements[paper_id].append(
                    card.statement_id
                )
            if card.paper_count == 1:
                single_paper_statement_papers.update(
                    card.paper_ids
                )

        incidence_rows: list[
            PaperStatementIncidence
        ] = []
        for paper_id in eligible_papers:
            statement_ids = sorted(
                set(
                    paper_to_eligible_statements.get(
                        paper_id,
                        [],
                    )
                )
            )
            has_single = (
                paper_id
                in single_paper_statement_papers
            )
            incidence_rows.append(
                PaperStatementIncidence(
                    paper_id=paper_id,
                    eligible_statement_ids=statement_ids,
                    eligible_statement_count=len(
                        statement_ids
                    ),
                    has_single_paper_statement=has_single,
                    only_multi_paper_statements=(
                        bool(statement_ids)
                        and not has_single
                    ),
                )
            )

        incidence_groups: dict[
            tuple[str, ...],
            list[str],
        ] = defaultdict(list)
        for row in incidence_rows:
            incidence_groups[
                tuple(row.eligible_statement_ids)
            ].append(row.paper_id)

        repeated_incidence_groups = [
            PaperIncidenceGroup(
                eligible_statement_ids=list(signature),
                paper_ids=sorted(paper_ids),
            )
            for signature, paper_ids in sorted(
                incidence_groups.items(),
                key=lambda item: (
                    item[0],
                    tuple(sorted(item[1])),
                ),
            )
            if len(paper_ids) >= 2
        ]

        multi_cards = [
            card
            for card in eligible_cards
            if card.multi_paper
        ]
        single_cards = [
            card
            for card in eligible_cards
            if not card.multi_paper
        ]

        incidence_count = sum(
            card.paper_count
            for card in eligible_cards
        )
        mean_papers = (
            incidence_count / len(eligible_cards)
            if eligible_cards
            else 0.0
        )
        unique_papers_per_statement = (
            len(eligible_papers) / len(eligible_cards)
            if eligible_cards
            else 0.0
        )

        only_multi_ids = sorted(
            row.paper_id
            for row in incidence_rows
            if row.only_multi_paper_statements
        )
        single_statement_ids = sorted(
            row.paper_id
            for row in incidence_rows
            if row.has_single_paper_statement
        )

        mismatch_count = sum(
            1
            for card in statement_cards
            if (
                card.declared_without_direct_scientific_support_paper_ids
                or card.direct_support_without_declared_paper_ids
                or "report_context_paper_provenance_mismatch"
                in card.diagnostic_flags
            )
        )

        payload = {
            "schema_version": "evidence-compression-report-v1",
            "report_id": _stable_id(
                "evidence_compression_report",
                packet.packet_sha256,
                report.report_id,
                report_sha,
                context.context_id,
                context.context_sha256,
            ),
            "source_packet_id": packet.packet_id,
            "source_packet_sha256": packet.packet_sha256,
            "source_report_id": report.report_id,
            "source_report_sha256": report_sha,
            "source_context_id": context.context_id,
            "source_context_sha256": context.context_sha256,
            "domain_profile_id": context.domain_profile_id,
            "selected_path_paper_ids": sorted(
                selected_path_papers
            ),
            "explorer_statement_paper_ids": explorer_papers,
            "context_statement_paper_ids": context_papers,
            "eligible_premise_paper_ids": eligible_papers,
            "selected_path_paper_count": len(
                selected_path_papers
            ),
            "explorer_statement_paper_count": len(
                explorer_papers
            ),
            "context_statement_paper_count": len(
                context_papers
            ),
            "eligible_premise_paper_count": len(
                eligible_papers
            ),
            "explorer_statement_count": len(
                report.statements
            ),
            "context_statement_count": len(
                context.evidence_statements
            ),
            "eligible_statement_count": len(
                eligible_cards
            ),
            "eligible_single_paper_statement_count": len(
                single_cards
            ),
            "eligible_multi_paper_statement_count": len(
                multi_cards
            ),
            "eligible_multi_paper_statement_fraction": (
                len(multi_cards) / len(eligible_cards)
                if eligible_cards
                else 0.0
            ),
            "eligible_statement_paper_incidence_count": (
                incidence_count
            ),
            "mean_papers_per_eligible_statement": (
                float(mean_papers)
            ),
            "eligible_papers_per_statement_ratio": (
                float(unique_papers_per_statement)
            ),
            "eligible_papers_with_single_paper_statement_ids": (
                single_statement_ids
            ),
            "eligible_papers_only_in_multi_paper_statements_ids": (
                only_multi_ids
            ),
            "eligible_papers_with_single_paper_statement_count": len(
                single_statement_ids
            ),
            "eligible_papers_only_in_multi_paper_statements_count": len(
                only_multi_ids
            ),
            "eligible_papers_only_in_multi_paper_statements_fraction": (
                len(only_multi_ids) / len(eligible_papers)
                if eligible_papers
                else 0.0
            ),
            "eligible_multi_paper_reported_statement_count": sum(
                1
                for card in multi_cards
                if card.epistemic_role == "reported"
            ),
            "eligible_multi_paper_synthesis_statement_count": sum(
                1
                for card in multi_cards
                if card.epistemic_role == "evidence_synthesis"
            ),
            "eligible_multi_paper_heterogeneous_profile_count": sum(
                1
                for card in multi_cards
                if card.distinct_structural_support_profile_count > 1
            ),
            "statements_with_declared_support_mismatch_count": (
                mismatch_count
            ),
            "paper_statement_incidence": [
                row.model_dump(mode="json")
                for row in incidence_rows
            ],
            "repeated_paper_incidence_groups": [
                row.model_dump(mode="json")
                for row in repeated_incidence_groups
            ],
            "statement_cards": [
                row.model_dump(mode="json")
                for row in statement_cards
            ],
            "policy": EvidenceCompressionPolicy().model_dump(
                mode="json"
            ),
        }

        return EvidenceCompressionReport(
            **payload,
            report_sha256=_sha256_json(payload),
        )
