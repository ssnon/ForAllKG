from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dac_her.evidence_family_diagnostics import (
    EvidenceFamilyDiagnosticReport,
    EvidenceFamilyProfile,
)
from dac_her.explorer_contracts import GraphExplorerPacket
from dac_her.hypothesis_contracts import (
    HypothesisContext,
    HypothesisEvidenceStatement,
)
from dac_her.path_lineage_propagation import (
    MinimalPathLineagePropagator,
    PathLineagePropagationReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExistingConstituentResolutionPolicy(StrictModel):
    mode: Literal["existing_first_conditional_materialization"] = (
        "existing_first_conditional_materialization"
    )
    only_ec2a_candidates: Literal[True] = True
    require_evidence_synthesis_parent: Literal[True] = True
    require_eligible_existing_constituent: Literal[True] = True
    require_claim_kind_compatibility: Literal[True] = True
    require_family_paper_containment: Literal[True] = True
    require_full_scientific_support_containment: Literal[True] = True
    generated_family_children_can_resolve_existing: Literal[False] = False
    unresolved_family_is_materialized: Literal[True] = True
    parent_statement_retained: Literal[True] = True
    parent_statement_modified: Literal[False] = False
    scientific_support_invented: Literal[False] = False
    external_evidence_added: Literal[False] = False


class ExistingConstituentCandidate(StrictModel):
    statement_id: str
    claim_kind: str
    paper_ids: list[str] = Field(default_factory=list)
    scientific_support_node_ids: list[str] = Field(default_factory=list)
    scientific_support_edge_ids: list[str] = Field(default_factory=list)

    exact_support_match: bool = False
    full_support_containment: bool = False
    family_paper_containment: bool = False
    claim_kind_compatible: bool = False

    extra_edge_count: int = 0
    extra_node_count: int = 0
    extra_paper_count: int = 0

    deterministic_rank_key: list[str | int] = Field(default_factory=list)


class FamilyConstituentResolution(StrictModel):
    parent_statement_id: str
    family_id: str
    family_claim_kind: str
    family_paper_ids: list[str] = Field(default_factory=list)
    family_scientific_support_node_ids: list[str] = Field(default_factory=list)
    family_scientific_support_edge_ids: list[str] = Field(default_factory=list)

    resolution_status: Literal[
        "resolved_to_existing_statement",
        "materialized_new_family_child",
    ]
    resolution_basis: Literal[
        "exact_existing_scientific_support",
        "contained_existing_scientific_support",
        "contained_existing_node_support",
        "no_existing_constituent_materialized",
    ]

    existing_statement_id: str | None = None
    materialized_statement_id: str | None = None

    candidate_count: int = 0
    candidates: list[ExistingConstituentCandidate] = Field(default_factory=list)

    resulting_constituent_statement_id: str
    support_path_ids_after: list[str] = Field(default_factory=list)
    pl1b_minimum_cover_size: int | None = None


class ParentConstituentLineage(StrictModel):
    parent_statement_id: str
    family_ids: list[str] = Field(default_factory=list)
    constituent_statement_ids: list[str] = Field(default_factory=list)
    reused_existing_statement_ids: list[str] = Field(default_factory=list)
    materialized_statement_ids: list[str] = Field(default_factory=list)
    family_count: int = 0
    resolved_existing_family_count: int = 0
    materialized_family_count: int = 0
    parent_retained: Literal[True] = True


class ExistingConstituentResolutionReport(StrictModel):
    schema_version: Literal["existing-constituent-resolution-report-v1"] = (
        "existing-constituent-resolution-report-v1"
    )
    report_id: str
    report_sha256: str

    source_packet_id: str
    source_packet_sha256: str
    source_context_id: str
    source_context_sha256_before: str
    output_context_sha256_after: str
    source_family_diagnostic_report_id: str
    source_family_diagnostic_report_sha256: str
    domain_profile_id: str

    candidate_parent_count: int = 0
    selected_candidate_parent_count: int = 0
    candidate_family_count: int = 0
    resolved_existing_family_count: int = 0
    materialized_family_count: int = 0

    original_statement_count: int = 0
    output_statement_count: int = 0
    eligible_statement_count_before: int = 0
    eligible_statement_count_after: int = 0

    original_statement_changed_count: int = 0
    original_statement_missing_count: int = 0
    context_sha_unchanged: bool = False

    family_resolutions: list[FamilyConstituentResolution] = Field(
        default_factory=list
    )
    parent_lineages: list[ParentConstituentLineage] = Field(
        default_factory=list
    )

    path_lineage_propagation_report_id: str | None = None
    path_lineage_propagation_report_sha256: str | None = None

    policy: ExistingConstituentResolutionPolicy = Field(
        default_factory=ExistingConstituentResolutionPolicy
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


def _context_rehash(context: HypothesisContext) -> HypothesisContext:
    payload = context.model_dump(mode="json")
    payload.pop("context_sha256", None)
    return context.model_copy(
        update={"context_sha256": _sha256_json(payload)}
    )


def _family_claim_kind(
    family: EvidenceFamilyProfile,
    *,
    parent_claim_kind: str,
) -> str:
    node_types = set(family.node_types)
    relations = set(family.edge_relations)

    if (
        "MechanismClaim" in node_types
        or "SUPPORTED_MECHANISM_INTERPRETATION" in relations
    ):
        return "mechanism"

    if (
        {"ObservationClaim", "CoordinationMotif"} & node_types
        or {"SUPPORTED_OBSERVATION", "HAS_MOTIF"} & relations
    ):
        return "observation"

    return parent_claim_kind


def _candidate_rank_key(
    *,
    exact: bool,
    extra_edges: int,
    extra_nodes: int,
    extra_papers: int,
    statement_id: str,
) -> tuple[int, int, int, int, str]:
    return (
        0 if exact else 1,
        int(extra_edges),
        int(extra_nodes),
        int(extra_papers),
        statement_id,
    )


def _existing_constituent_candidates(
    *,
    family: EvidenceFamilyProfile,
    family_claim_kind: str,
    context: HypothesisContext,
    parent_statement_id: str,
) -> list[ExistingConstituentCandidate]:
    family_edges = set(family.direct_support_edge_ids)
    family_nodes = set(family.direct_support_node_ids)
    family_papers = set(family.paper_ids)

    candidates: list[ExistingConstituentCandidate] = []

    for statement in context.evidence_statements:
        if statement.statement_id == parent_statement_id:
            continue
        if statement.statement_id.startswith("stmtfam:"):
            continue
        if not statement.eligible_as_premise:
            continue

        kind_ok = statement.claim_kind == family_claim_kind
        if not kind_ok:
            continue

        statement_papers = set(statement.paper_ids)
        papers_ok = family_papers <= statement_papers
        if not papers_ok:
            continue

        statement_edges = set(statement.scientific_support_edge_ids)
        statement_nodes = set(statement.scientific_support_node_ids)

        if family_edges:
            support_ok = (
                family_edges <= statement_edges
                and family_nodes <= statement_nodes
            )
            if not support_ok:
                continue
            exact = (
                family_edges == statement_edges
                and family_nodes == statement_nodes
                and family_papers == statement_papers
            )
        elif family_nodes:
            support_ok = family_nodes <= statement_nodes
            if not support_ok:
                continue
            exact = (
                family_nodes == statement_nodes
                and family_papers == statement_papers
            )
        else:
            continue

        extra_edges = len(statement_edges - family_edges)
        extra_nodes = len(statement_nodes - family_nodes)
        extra_papers = len(statement_papers - family_papers)
        rank_key = _candidate_rank_key(
            exact=exact,
            extra_edges=extra_edges,
            extra_nodes=extra_nodes,
            extra_papers=extra_papers,
            statement_id=statement.statement_id,
        )

        candidates.append(
            ExistingConstituentCandidate(
                statement_id=statement.statement_id,
                claim_kind=statement.claim_kind,
                paper_ids=sorted(statement_papers),
                scientific_support_node_ids=sorted(statement_nodes),
                scientific_support_edge_ids=sorted(statement_edges),
                exact_support_match=exact,
                full_support_containment=True,
                family_paper_containment=True,
                claim_kind_compatible=True,
                extra_edge_count=extra_edges,
                extra_node_count=extra_nodes,
                extra_paper_count=extra_papers,
                deterministic_rank_key=[
                    rank_key[0],
                    rank_key[1],
                    rank_key[2],
                    rank_key[3],
                    rank_key[4],
                ],
            )
        )

    return sorted(
        candidates,
        key=lambda row: _candidate_rank_key(
            exact=row.exact_support_match,
            extra_edges=row.extra_edge_count,
            extra_nodes=row.extra_node_count,
            extra_papers=row.extra_paper_count,
            statement_id=row.statement_id,
        ),
    )


def find_existing_constituent(
    *,
    family: EvidenceFamilyProfile,
    family_claim_kind: str,
    context: HypothesisContext,
    parent_statement_id: str,
) -> tuple[
    ExistingConstituentCandidate | None,
    list[ExistingConstituentCandidate],
]:
    candidates = _existing_constituent_candidates(
        family=family,
        family_claim_kind=family_claim_kind,
        context=context,
        parent_statement_id=parent_statement_id,
    )
    return (
        candidates[0] if candidates else None,
        candidates,
    )


def _family_text(
    packet: GraphExplorerPacket,
    *,
    parent_statement_id: str,
    family: EvidenceFamilyProfile,
) -> str:
    fragments: list[str] = []
    seen: set[str] = set()

    for node_id in family.direct_support_node_ids:
        node = packet.evidence_catalog.nodes.get(node_id)
        if node is None:
            raise ValueError(
                f"{parent_statement_id}: family node missing from packet: "
                f"{node_id}"
            )
        text = str(node.node_text or node.label or node.node_id).strip()
        if text and text not in seen:
            seen.add(text)
            fragments.append(text)

    if not fragments:
        raise ValueError(
            f"{parent_statement_id}: unresolved family has no grounded node text"
        )

    return (
        f"Decomposed evidence family from {parent_statement_id} "
        f"(papers: {', '.join(family.paper_ids)}): "
        + " | ".join(fragments)
    )


def _materialized_child_id(
    *,
    parent_statement_id: str,
    family_id: str,
) -> str:
    return _stable_id(
        "stmtfam",
        parent_statement_id,
        family_id,
    )


def _materialized_child(
    packet: GraphExplorerPacket,
    *,
    parent: HypothesisEvidenceStatement,
    family: EvidenceFamilyProfile,
    family_claim_kind: str,
) -> HypothesisEvidenceStatement:
    return HypothesisEvidenceStatement(
        statement_id=_materialized_child_id(
            parent_statement_id=parent.statement_id,
            family_id=family.family_id,
        ),
        text=_family_text(
            packet,
            parent_statement_id=parent.statement_id,
            family=family,
        ),
        epistemic_role=parent.epistemic_role,
        claim_kind=family_claim_kind,
        paper_ids=list(family.paper_ids),
        scientific_support_node_ids=list(
            family.direct_support_node_ids
        ),
        scientific_support_edge_ids=list(
            family.direct_support_edge_ids
        ),
        support_path_ids=[],
        alignment_path_ids=[],
        requires_verification=parent.requires_verification,
        eligible_as_premise=True,
        eligible_as_gap=False,
        premise_restrictions=list(parent.premise_restrictions),
    )


class ExistingFirstConstituentResolver:
    """EC2-D existing-first constituent resolution.

    For every EC2-A heterogeneous synthesis family:
    1. look for an already-existing eligible statement whose scientific support
       fully contains the family support and whose claim kind is compatible;
    2. reuse that statement as the family constituent when found;
    3. materialize a new family child only when no such existing constituent
       exists.

    The parent synthesis is always retained unchanged.
    """

    def resolve(
        self,
        packet: GraphExplorerPacket,
        context: HypothesisContext,
        family_diagnostics: EvidenceFamilyDiagnosticReport,
        *,
        statement_ids: set[str] | None = None,
    ) -> tuple[
        HypothesisContext,
        ExistingConstituentResolutionReport,
        PathLineagePropagationReport | None,
    ]:
        if context.source_packet_id != packet.packet_id:
            raise ValueError("EC2-D context/packet ID mismatch")
        if context.source_packet_sha256 != packet.packet_sha256:
            raise ValueError("EC2-D context/packet SHA mismatch")
        if family_diagnostics.source_context_id != context.context_id:
            raise ValueError("EC2-D family diagnostics/context ID mismatch")
        if family_diagnostics.source_context_sha256 != context.context_sha256:
            raise ValueError("EC2-D family diagnostics/context SHA mismatch")
        if family_diagnostics.source_packet_id != packet.packet_id:
            raise ValueError("EC2-D family diagnostics/packet ID mismatch")

        source_statements = list(context.evidence_statements)
        source_by_id = {
            row.statement_id: row
            for row in source_statements
        }

        candidate_rows = {
            row.statement_id: row
            for row in family_diagnostics.statement_diagnostics
            if row.decomposition_candidate
        }
        candidate_ids = set(candidate_rows)

        if statement_ids is None:
            selected_ids = set(candidate_ids)
        else:
            unknown = set(statement_ids) - candidate_ids
            if unknown:
                raise ValueError(
                    "requested statement IDs are not EC2-A decomposition "
                    f"candidates: {sorted(unknown)}"
                )
            selected_ids = set(statement_ids)

        materialized_children: list[HypothesisEvidenceStatement] = []
        provisional_resolutions: list[
            tuple[
                str,
                EvidenceFamilyProfile,
                str,
                ExistingConstituentCandidate | None,
                list[ExistingConstituentCandidate],
                str,
            ]
        ] = []

        existing_ids = set(source_by_id)

        for parent_id in sorted(selected_ids):
            parent = source_by_id.get(parent_id)
            if parent is None:
                raise ValueError(
                    f"EC2-D candidate parent missing from context: {parent_id}"
                )
            if parent.epistemic_role != "evidence_synthesis":
                continue

            diagnostic = candidate_rows[parent_id]
            for family in diagnostic.evidence_families:
                family_kind = _family_claim_kind(
                    family,
                    parent_claim_kind=parent.claim_kind,
                )
                best, candidates = find_existing_constituent(
                    family=family,
                    family_claim_kind=family_kind,
                    context=context,
                    parent_statement_id=parent_id,
                )

                if best is not None:
                    provisional_resolutions.append(
                        (
                            parent_id,
                            family,
                            family_kind,
                            best,
                            candidates,
                            best.statement_id,
                        )
                    )
                    continue

                child = _materialized_child(
                    packet,
                    parent=parent,
                    family=family,
                    family_claim_kind=family_kind,
                )
                if child.statement_id in existing_ids:
                    raise ValueError(
                        "EC2-D materialized child ID collision: "
                        f"{child.statement_id}"
                    )
                existing_ids.add(child.statement_id)
                materialized_children.append(child)
                provisional_resolutions.append(
                    (
                        parent_id,
                        family,
                        family_kind,
                        None,
                        candidates,
                        child.statement_id,
                    )
                )

        propagation: PathLineagePropagationReport | None = None
        if materialized_children:
            pre_path_context = context.model_copy(
                update={
                    "evidence_statements": (
                        source_statements + materialized_children
                    )
                }
            )
            pre_path_context = _context_rehash(pre_path_context)
            output_context, propagation = (
                MinimalPathLineagePropagator().propagate(
                    packet,
                    pre_path_context,
                )
            )
        else:
            # No scientific representation changed; preserve the exact source
            # context and its SHA rather than producing a synthetic rehash.
            output_context = context

        output_by_id = {
            row.statement_id: row
            for row in output_context.evidence_statements
        }

        original_changed = 0
        original_missing = 0
        for original in source_statements:
            after = output_by_id.get(original.statement_id)
            if after is None:
                original_missing += 1
                continue
            if (
                after.model_dump(mode="json")
                != original.model_dump(mode="json")
            ):
                original_changed += 1

        if original_changed or original_missing:
            raise RuntimeError(
                "EC2-D changed or removed original statements; "
                f"changed={original_changed}, missing={original_missing}"
            )

        propagation_cards = {}
        if propagation is not None:
            propagation_cards = {
                row.statement_id: row
                for row in propagation.statement_cards
            }

        family_resolutions: list[FamilyConstituentResolution] = []
        for (
            parent_id,
            family,
            family_kind,
            best,
            candidates,
            resulting_id,
        ) in provisional_resolutions:
            resulting_statement = output_by_id[resulting_id]

            if best is not None:
                if best.exact_support_match:
                    basis = "exact_existing_scientific_support"
                elif family.direct_support_edge_ids:
                    basis = "contained_existing_scientific_support"
                else:
                    basis = "contained_existing_node_support"

                family_resolutions.append(
                    FamilyConstituentResolution(
                        parent_statement_id=parent_id,
                        family_id=family.family_id,
                        family_claim_kind=family_kind,
                        family_paper_ids=list(family.paper_ids),
                        family_scientific_support_node_ids=list(
                            family.direct_support_node_ids
                        ),
                        family_scientific_support_edge_ids=list(
                            family.direct_support_edge_ids
                        ),
                        resolution_status="resolved_to_existing_statement",
                        resolution_basis=basis,
                        existing_statement_id=best.statement_id,
                        candidate_count=len(candidates),
                        candidates=candidates,
                        resulting_constituent_statement_id=best.statement_id,
                        support_path_ids_after=list(
                            resulting_statement.support_path_ids
                        ),
                    )
                )
            else:
                card = propagation_cards.get(resulting_id)
                family_resolutions.append(
                    FamilyConstituentResolution(
                        parent_statement_id=parent_id,
                        family_id=family.family_id,
                        family_claim_kind=family_kind,
                        family_paper_ids=list(family.paper_ids),
                        family_scientific_support_node_ids=list(
                            family.direct_support_node_ids
                        ),
                        family_scientific_support_edge_ids=list(
                            family.direct_support_edge_ids
                        ),
                        resolution_status="materialized_new_family_child",
                        resolution_basis="no_existing_constituent_materialized",
                        materialized_statement_id=resulting_id,
                        candidate_count=0,
                        candidates=[],
                        resulting_constituent_statement_id=resulting_id,
                        support_path_ids_after=list(
                            resulting_statement.support_path_ids
                        ),
                        pl1b_minimum_cover_size=(
                            card.minimum_cover_size
                            if card is not None
                            else None
                        ),
                    )
                )

        lineages: list[ParentConstituentLineage] = []
        for parent_id in sorted(selected_ids):
            rows = [
                row
                for row in family_resolutions
                if row.parent_statement_id == parent_id
            ]
            if not rows:
                continue
            lineages.append(
                ParentConstituentLineage(
                    parent_statement_id=parent_id,
                    family_ids=[
                        row.family_id
                        for row in rows
                    ],
                    constituent_statement_ids=[
                        row.resulting_constituent_statement_id
                        for row in rows
                    ],
                    reused_existing_statement_ids=[
                        row.existing_statement_id
                        for row in rows
                        if row.existing_statement_id is not None
                    ],
                    materialized_statement_ids=[
                        row.materialized_statement_id
                        for row in rows
                        if row.materialized_statement_id is not None
                    ],
                    family_count=len(rows),
                    resolved_existing_family_count=sum(
                        row.resolution_status
                        == "resolved_to_existing_statement"
                        for row in rows
                    ),
                    materialized_family_count=sum(
                        row.resolution_status
                        == "materialized_new_family_child"
                        for row in rows
                    ),
                )
            )

        before_eligible = sum(
            row.eligible_as_premise
            for row in source_statements
        )
        after_eligible = sum(
            row.eligible_as_premise
            for row in output_context.evidence_statements
        )

        payload = {
            "schema_version": "existing-constituent-resolution-report-v1",
            "report_id": _stable_id(
                "existing_constituent_resolution_report",
                packet.packet_sha256,
                context.context_sha256,
                output_context.context_sha256,
                family_diagnostics.report_sha256,
            ),
            "source_packet_id": packet.packet_id,
            "source_packet_sha256": packet.packet_sha256,
            "source_context_id": context.context_id,
            "source_context_sha256_before": context.context_sha256,
            "output_context_sha256_after": output_context.context_sha256,
            "source_family_diagnostic_report_id": family_diagnostics.report_id,
            "source_family_diagnostic_report_sha256": (
                family_diagnostics.report_sha256
            ),
            "domain_profile_id": context.domain_profile_id,
            "candidate_parent_count": len(candidate_ids),
            "selected_candidate_parent_count": len(selected_ids),
            "candidate_family_count": len(family_resolutions),
            "resolved_existing_family_count": sum(
                row.resolution_status == "resolved_to_existing_statement"
                for row in family_resolutions
            ),
            "materialized_family_count": sum(
                row.resolution_status == "materialized_new_family_child"
                for row in family_resolutions
            ),
            "original_statement_count": len(source_statements),
            "output_statement_count": len(output_context.evidence_statements),
            "eligible_statement_count_before": before_eligible,
            "eligible_statement_count_after": after_eligible,
            "original_statement_changed_count": original_changed,
            "original_statement_missing_count": original_missing,
            "context_sha_unchanged": (
                context.context_sha256 == output_context.context_sha256
            ),
            "family_resolutions": [
                row.model_dump(mode="json")
                for row in family_resolutions
            ],
            "parent_lineages": [
                row.model_dump(mode="json")
                for row in lineages
            ],
            "path_lineage_propagation_report_id": (
                propagation.report_id
                if propagation is not None
                else None
            ),
            "path_lineage_propagation_report_sha256": (
                propagation.report_sha256
                if propagation is not None
                else None
            ),
            "policy": ExistingConstituentResolutionPolicy().model_dump(
                mode="json"
            ),
        }

        report = ExistingConstituentResolutionReport(
            **payload,
            report_sha256=_sha256_json(payload),
        )
        return output_context, report, propagation
