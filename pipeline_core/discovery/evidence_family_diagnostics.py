from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.evidence_compression import (
    EvidenceCompressionReport,
    EvidenceCompressionStatementCard,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceFamilyDiagnosticPolicy(StrictModel):
    diagnostic_only: Literal[True] = True
    scientific_selection_changed: Literal[False] = False
    automatic_statement_decomposition_allowed: Literal[False] = False
    heterogeneous_structural_profile_is_semantic_difference: Literal[False] = False
    missing_explicit_path_lineage_invalidates_premise: Literal[False] = False
    decomposition_requires_semantic_review: Literal[True] = True


class EvidenceFamilyProfile(StrictModel):
    family_id: str
    paper_ids: list[str] = Field(default_factory=list)
    paper_count: int = 0
    node_types: list[str] = Field(default_factory=list)
    edge_relations: list[str] = Field(default_factory=list)
    path_structure_types: list[str] = Field(default_factory=list)
    direct_support_node_ids: list[str] = Field(default_factory=list)
    direct_support_edge_ids: list[str] = Field(default_factory=list)
    support_path_ids: list[str] = Field(default_factory=list)
    direct_support_unit_count: int = 0
    paper_direct_support_unit_counts: dict[str, int] = Field(default_factory=dict)


class EvidenceFamilyStatementDiagnostic(StrictModel):
    statement_id: str
    text: str
    epistemic_role: str
    claim_kind: str
    eligible_as_premise: bool = False

    paper_ids: list[str] = Field(default_factory=list)
    paper_count: int = 0
    multi_paper: bool = False

    structural_support_profile_count: int = 0
    evidence_family_count: int = 0
    evidence_families: list[EvidenceFamilyProfile] = Field(default_factory=list)

    decomposition_candidate: bool = False
    decomposition_candidate_basis: list[str] = Field(default_factory=list)
    decomposition_blockers: list[str] = Field(default_factory=list)

    explicit_support_path_ids: list[str] = Field(default_factory=list)
    explicit_support_path_count: int = 0
    has_explicit_path_lineage: bool = False
    path_lineage_diagnostic_flags: list[str] = Field(default_factory=list)

    provenance_complete_for_declared_papers: bool = False
    source_compression_flags: list[str] = Field(default_factory=list)


class EvidenceFamilyDiagnosticReport(StrictModel):
    schema_version: Literal["evidence-family-diagnostic-report-v1"] = (
        "evidence-family-diagnostic-report-v1"
    )
    report_id: str
    report_sha256: str

    source_compression_report_id: str
    source_compression_report_sha256: str
    source_packet_id: str
    source_packet_sha256: str
    source_report_id: str
    source_report_sha256: str
    source_context_id: str
    source_context_sha256: str
    domain_profile_id: str

    eligible_statement_count: int = 0
    eligible_multi_paper_statement_count: int = 0
    eligible_homogeneous_multi_paper_statement_count: int = 0
    eligible_heterogeneous_multi_paper_statement_count: int = 0

    decomposition_candidate_count: int = 0
    decomposition_candidate_statement_ids: list[str] = Field(default_factory=list)
    homogeneous_multi_paper_statement_ids: list[str] = Field(default_factory=list)
    heterogeneous_not_candidate_statement_ids: list[str] = Field(default_factory=list)

    candidate_family_count: int = 0
    candidate_paper_ids: list[str] = Field(default_factory=list)
    candidate_paper_count: int = 0

    eligible_statements_with_explicit_path_lineage_count: int = 0
    eligible_statements_without_explicit_path_lineage_count: int = 0
    eligible_statements_without_explicit_path_lineage_fraction: float = 0.0
    eligible_statements_without_explicit_path_lineage_ids: list[str] = Field(
        default_factory=list
    )

    statement_diagnostics: list[EvidenceFamilyStatementDiagnostic] = Field(
        default_factory=list
    )
    policy: EvidenceFamilyDiagnosticPolicy = Field(
        default_factory=EvidenceFamilyDiagnosticPolicy
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


def _profile_signature(contribution: Any) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    return (
        tuple(sorted(set(contribution.node_types))),
        tuple(sorted(set(contribution.edge_relations))),
        tuple(sorted(set(contribution.path_structure_types))),
    )


def _family_id(
    statement_id: str,
    signature: tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
    ],
) -> str:
    return _stable_id(
        "evidence_family",
        statement_id,
        json.dumps(
            signature,
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def _has_provenance_mismatch(
    card: EvidenceCompressionStatementCard,
) -> bool:
    mismatch_flags = {
        "declared_paper_without_direct_scientific_support",
        "direct_scientific_support_paper_not_declared",
        "report_context_paper_provenance_mismatch",
    }
    return (
        bool(
            card.declared_without_direct_scientific_support_paper_ids
        )
        or bool(
            card.direct_support_without_declared_paper_ids
        )
        or bool(
            mismatch_flags
            & set(card.diagnostic_flags)
        )
    )


class EvidenceFamilyCandidateAssessor:
    """EC2-A deterministic candidate diagnostics.

    A decomposition candidate is NOT automatically decomposed. EC2-A only
    identifies multi-paper eligible statements whose declared papers have
    complete direct scientific provenance and occupy more than one exact
    structural support profile.
    """

    def assess(
        self,
        compression: EvidenceCompressionReport,
    ) -> EvidenceFamilyDiagnosticReport:
        diagnostics: list[
            EvidenceFamilyStatementDiagnostic
        ] = []

        for card in compression.statement_cards:
            family_members: dict[
                tuple[
                    tuple[str, ...],
                    tuple[str, ...],
                    tuple[str, ...],
                ],
                list[Any],
            ] = defaultdict(list)

            for contribution in card.paper_contributions:
                if not contribution.declared_in_statement:
                    continue
                family_members[
                    _profile_signature(contribution)
                ].append(contribution)

            families: list[EvidenceFamilyProfile] = []
            for signature, contributions in sorted(
                family_members.items(),
                key=lambda item: (
                    item[0],
                    tuple(
                        sorted(
                            contribution.paper_id
                            for contribution in item[1]
                        )
                    ),
                ),
            ):
                node_types, relations, path_types = signature
                paper_ids = sorted(
                    contribution.paper_id
                    for contribution in contributions
                )
                direct_nodes = sorted(
                    {
                        node_id
                        for contribution in contributions
                        for node_id in contribution.direct_support_node_ids
                    }
                )
                direct_edges = sorted(
                    {
                        edge_id
                        for contribution in contributions
                        for edge_id in contribution.direct_support_edge_ids
                    }
                )
                path_ids = sorted(
                    {
                        path_id
                        for contribution in contributions
                        for path_id in contribution.support_path_ids
                    }
                )
                paper_unit_counts = {
                    contribution.paper_id: (
                        contribution.direct_support_unit_count
                    )
                    for contribution in sorted(
                        contributions,
                        key=lambda value: value.paper_id,
                    )
                }
                families.append(
                    EvidenceFamilyProfile(
                        family_id=_family_id(
                            card.statement_id,
                            signature,
                        ),
                        paper_ids=paper_ids,
                        paper_count=len(paper_ids),
                        node_types=list(node_types),
                        edge_relations=list(relations),
                        path_structure_types=list(path_types),
                        direct_support_node_ids=direct_nodes,
                        direct_support_edge_ids=direct_edges,
                        support_path_ids=path_ids,
                        direct_support_unit_count=sum(
                            contribution.direct_support_unit_count
                            for contribution in contributions
                        ),
                        paper_direct_support_unit_counts=paper_unit_counts,
                    )
                )

            provenance_complete = (
                card.all_declared_papers_have_direct_scientific_support
                and not _has_provenance_mismatch(card)
            )

            blockers: list[str] = []
            basis: list[str] = []

            if not card.eligible_as_premise:
                blockers.append(
                    "not_eligible_positive_premise"
                )
            if not card.multi_paper:
                blockers.append(
                    "not_multi_paper_statement"
                )
            if not provenance_complete:
                blockers.append(
                    "declared_paper_provenance_incomplete"
                )
            if len(families) <= 1:
                blockers.append(
                    "single_structural_support_family"
                )

            decomposition_candidate = (
                card.eligible_as_premise
                and card.multi_paper
                and provenance_complete
                and len(families) > 1
            )

            if decomposition_candidate:
                basis.extend(
                    [
                        "eligible_positive_premise",
                        "multi_paper_statement",
                        "complete_declared_paper_direct_provenance",
                        "multiple_structural_support_families",
                        "semantic_review_required_before_any_split",
                    ]
                )

            path_flags: list[str] = []
            if (
                card.eligible_as_premise
                and card.support_path_count == 0
            ):
                path_flags.append(
                    "eligible_statement_has_no_explicit_support_path_lineage"
                )
            if card.support_path_count > 0:
                path_flags.append(
                    "explicit_support_path_lineage_present"
                )

            diagnostics.append(
                EvidenceFamilyStatementDiagnostic(
                    statement_id=card.statement_id,
                    text=card.text,
                    epistemic_role=card.epistemic_role,
                    claim_kind=card.claim_kind,
                    eligible_as_premise=card.eligible_as_premise,
                    paper_ids=list(card.paper_ids),
                    paper_count=card.paper_count,
                    multi_paper=card.multi_paper,
                    structural_support_profile_count=(
                        card.distinct_structural_support_profile_count
                    ),
                    evidence_family_count=len(families),
                    evidence_families=families,
                    decomposition_candidate=decomposition_candidate,
                    decomposition_candidate_basis=basis,
                    decomposition_blockers=sorted(
                        set(blockers)
                    ),
                    explicit_support_path_ids=list(
                        card.support_path_ids
                    ),
                    explicit_support_path_count=(
                        card.support_path_count
                    ),
                    has_explicit_path_lineage=(
                        card.support_path_count > 0
                    ),
                    path_lineage_diagnostic_flags=(
                        path_flags
                    ),
                    provenance_complete_for_declared_papers=(
                        provenance_complete
                    ),
                    source_compression_flags=list(
                        card.diagnostic_flags
                    ),
                )
            )

        eligible = [
            row
            for row in diagnostics
            if row.eligible_as_premise
        ]
        eligible_multi = [
            row
            for row in eligible
            if row.multi_paper
        ]
        eligible_homogeneous = [
            row
            for row in eligible_multi
            if row.evidence_family_count <= 1
        ]
        eligible_heterogeneous = [
            row
            for row in eligible_multi
            if row.evidence_family_count > 1
        ]
        candidates = [
            row
            for row in eligible_heterogeneous
            if row.decomposition_candidate
        ]
        heterogeneous_not_candidate = [
            row
            for row in eligible_heterogeneous
            if not row.decomposition_candidate
        ]

        without_path = [
            row
            for row in eligible
            if not row.has_explicit_path_lineage
        ]
        with_path = [
            row
            for row in eligible
            if row.has_explicit_path_lineage
        ]

        candidate_papers = sorted(
            {
                paper_id
                for row in candidates
                for paper_id in row.paper_ids
            }
        )

        source_sha = compression.report_sha256
        report_id = _stable_id(
            "evidence_family_diagnostic_report",
            compression.report_id,
            source_sha,
        )

        payload = {
            "schema_version": "evidence-family-diagnostic-report-v1",
            "report_id": report_id,
            "source_compression_report_id": compression.report_id,
            "source_compression_report_sha256": source_sha,
            "source_packet_id": compression.source_packet_id,
            "source_packet_sha256": compression.source_packet_sha256,
            "source_report_id": compression.source_report_id,
            "source_report_sha256": compression.source_report_sha256,
            "source_context_id": compression.source_context_id,
            "source_context_sha256": compression.source_context_sha256,
            "domain_profile_id": compression.domain_profile_id,
            "eligible_statement_count": len(eligible),
            "eligible_multi_paper_statement_count": len(
                eligible_multi
            ),
            "eligible_homogeneous_multi_paper_statement_count": len(
                eligible_homogeneous
            ),
            "eligible_heterogeneous_multi_paper_statement_count": len(
                eligible_heterogeneous
            ),
            "decomposition_candidate_count": len(
                candidates
            ),
            "decomposition_candidate_statement_ids": [
                row.statement_id
                for row in candidates
            ],
            "homogeneous_multi_paper_statement_ids": [
                row.statement_id
                for row in eligible_homogeneous
            ],
            "heterogeneous_not_candidate_statement_ids": [
                row.statement_id
                for row in heterogeneous_not_candidate
            ],
            "candidate_family_count": sum(
                row.evidence_family_count
                for row in candidates
            ),
            "candidate_paper_ids": candidate_papers,
            "candidate_paper_count": len(
                candidate_papers
            ),
            "eligible_statements_with_explicit_path_lineage_count": len(
                with_path
            ),
            "eligible_statements_without_explicit_path_lineage_count": len(
                without_path
            ),
            "eligible_statements_without_explicit_path_lineage_fraction": (
                len(without_path) / len(eligible)
                if eligible
                else 0.0
            ),
            "eligible_statements_without_explicit_path_lineage_ids": [
                row.statement_id
                for row in without_path
            ],
            "statement_diagnostics": [
                row.model_dump(mode="json")
                for row in diagnostics
            ],
            "policy": EvidenceFamilyDiagnosticPolicy().model_dump(
                mode="json"
            ),
        }

        return EvidenceFamilyDiagnosticReport(
            **payload,
            report_sha256=_sha256_json(payload),
        )
