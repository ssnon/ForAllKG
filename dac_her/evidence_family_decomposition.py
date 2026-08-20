from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.evidence_family_diagnostics import (
    EvidenceFamilyDiagnosticReport,
    EvidenceFamilyProfile,
)
from pipeline_core.discovery.explorer_contracts import GraphExplorerPacket
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisEvidenceStatement,
)
from pipeline_core.discovery.path_lineage_propagation import (
    MinimalPathLineagePropagator,
    PathLineagePropagationReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceFamilyDecompositionPolicy(StrictModel):
    mode: Literal["conservative_additive_family_children"] = (
        "conservative_additive_family_children"
    )
    opt_in_only: Literal[True] = True
    parent_statement_retained: Literal[True] = True
    parent_statement_modified: Literal[False] = False
    only_ec2a_candidates: Literal[True] = True
    require_evidence_synthesis_parent: Literal[True] = True
    child_positive_premise_enabled: Literal[True] = True
    child_text_generation: Literal["deterministic_grounded_node_text_aggregation"] = (
        "deterministic_grounded_node_text_aggregation"
    )
    child_path_lineage: Literal["pl1b_minimal_deterministic_cover"] = (
        "pl1b_minimal_deterministic_cover"
    )
    external_evidence_added: Literal[False] = False
    scientific_support_invented: Literal[False] = False


class FamilyChildLineage(StrictModel):
    parent_statement_id: str
    family_id: str
    child_statement_id: str
    paper_ids: list[str] = Field(default_factory=list)

    node_types: list[str] = Field(default_factory=list)
    edge_relations: list[str] = Field(default_factory=list)

    scientific_support_node_ids: list[str] = Field(default_factory=list)
    scientific_support_edge_ids: list[str] = Field(default_factory=list)
    support_path_ids: list[str] = Field(default_factory=list)

    epistemic_role: str
    claim_kind: str
    text_origin: Literal["deterministic_grounded_node_text_aggregation"] = (
        "deterministic_grounded_node_text_aggregation"
    )

    eligible_as_premise: bool = True
    parent_retained: Literal[True] = True
    pl1b_minimum_cover_size: int | None = None


class ParentDecompositionCard(StrictModel):
    parent_statement_id: str
    parent_epistemic_role: str
    parent_claim_kind: str
    parent_paper_ids: list[str] = Field(default_factory=list)
    action: Literal[
        "expanded_with_family_children",
        "skipped_parent_not_evidence_synthesis",
        "skipped_not_selected",
        "skipped_not_candidate",
    ]
    family_count: int = 0
    child_statement_ids: list[str] = Field(default_factory=list)
    child_count: int = 0
    parent_retained: Literal[True] = True


class EvidenceFamilyDecompositionReport(StrictModel):
    schema_version: Literal["evidence-family-decomposition-report-v1"] = (
        "evidence-family-decomposition-report-v1"
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

    candidate_statement_count: int = 0
    selected_candidate_statement_count: int = 0
    expanded_parent_count: int = 0
    skipped_parent_count: int = 0

    original_statement_count: int = 0
    output_statement_count: int = 0
    child_statement_count: int = 0

    eligible_statement_count_before: int = 0
    eligible_statement_count_after: int = 0

    original_statement_changed_count: int = 0
    original_statement_missing_count: int = 0

    child_lineages: list[FamilyChildLineage] = Field(default_factory=list)
    parent_cards: list[ParentDecompositionCard] = Field(default_factory=list)

    child_path_lineage_propagation_report_id: str | None = None
    child_path_lineage_propagation_report_sha256: str | None = None

    policy: EvidenceFamilyDecompositionPolicy = Field(
        default_factory=EvidenceFamilyDecompositionPolicy
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
        update={
            "context_sha256": _sha256_json(payload),
        }
    )


def _family_claim_kind(
    family: EvidenceFamilyProfile,
    *,
    parent_claim_kind: str,
) -> str:
    node_types = set(family.node_types)
    relations = set(family.edge_relations)

    mechanism_markers = {
        "MechanismClaim",
    }
    mechanism_relations = {
        "SUPPORTED_MECHANISM_INTERPRETATION",
    }
    observation_markers = {
        "ObservationClaim",
        "CoordinationMotif",
    }
    observation_relations = {
        "SUPPORTED_OBSERVATION",
        "HAS_MOTIF",
    }

    if (
        node_types & mechanism_markers
        or relations & mechanism_relations
    ):
        return "mechanism"
    if (
        node_types & observation_markers
        or relations & observation_relations
    ):
        return "observation"
    return parent_claim_kind


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
            f"{parent_statement_id}: family has no grounded node text"
        )

    paper_text = ", ".join(family.paper_ids)
    joined = " | ".join(fragments)
    return (
        f"Decomposed evidence family from {parent_statement_id} "
        f"(papers: {paper_text}): {joined}"
    )


def _child_statement_id(
    *,
    parent_statement_id: str,
    family_id: str,
) -> str:
    return _stable_id(
        "stmtfam",
        parent_statement_id,
        family_id,
    )


def _child_statement(
    packet: GraphExplorerPacket,
    *,
    parent: HypothesisEvidenceStatement,
    family: EvidenceFamilyProfile,
) -> HypothesisEvidenceStatement:
    child_id = _child_statement_id(
        parent_statement_id=parent.statement_id,
        family_id=family.family_id,
    )
    return HypothesisEvidenceStatement(
        statement_id=child_id,
        text=_family_text(
            packet,
            parent_statement_id=parent.statement_id,
            family=family,
        ),
        epistemic_role=parent.epistemic_role,
        claim_kind=_family_claim_kind(
            family,
            parent_claim_kind=parent.claim_kind,
        ),
        paper_ids=list(family.paper_ids),
        support_path_ids=[],
        alignment_path_ids=[],
        scientific_support_node_ids=list(
            family.direct_support_node_ids
        ),
        scientific_support_edge_ids=list(
            family.direct_support_edge_ids
        ),
        requires_verification=parent.requires_verification,
        premise_restrictions=list(
            parent.premise_restrictions
        ),
        eligible_as_premise=True,
        eligible_as_gap=False,
    )


class ConservativeEvidenceFamilyDecomposer:
    """EC2-B additive family decomposition.

    The original candidate synthesis statement is retained byte-for-byte.
    New family-addressable child statements are added using only existing
    packet scientific support. PL1-B then provides minimal path lineage for
    the newly-added children.
    """

    def decompose(
        self,
        packet: GraphExplorerPacket,
        context: HypothesisContext,
        family_diagnostics: EvidenceFamilyDiagnosticReport,
        *,
        statement_ids: set[str] | None = None,
    ) -> tuple[
        HypothesisContext,
        EvidenceFamilyDecompositionReport,
        PathLineagePropagationReport,
    ]:
        if context.source_packet_id != packet.packet_id:
            raise ValueError("context/packet ID mismatch")
        if context.source_packet_sha256 != packet.packet_sha256:
            raise ValueError("context/packet SHA mismatch")
        if (
            family_diagnostics.source_context_id
            != context.context_id
        ):
            raise ValueError(
                "family diagnostics/context ID mismatch"
            )
        if (
            family_diagnostics.source_context_sha256
            != context.context_sha256
        ):
            raise ValueError(
                "family diagnostics/context SHA mismatch"
            )
        if (
            family_diagnostics.source_packet_id
            != packet.packet_id
        ):
            raise ValueError(
                "family diagnostics/packet ID mismatch"
            )

        source_statements = list(
            context.evidence_statements
        )
        source_by_id = {
            row.statement_id: row
            for row in source_statements
        }
        existing_ids = set(source_by_id)

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

        child_statements: list[
            HypothesisEvidenceStatement
        ] = []
        parent_cards: list[
            ParentDecompositionCard
        ] = []

        for parent_id in sorted(candidate_ids):
            parent = source_by_id.get(
                parent_id
            )
            if parent is None:
                raise ValueError(
                    f"candidate parent missing from context: {parent_id}"
                )
            diagnostic = candidate_rows[parent_id]

            if parent_id not in selected_ids:
                parent_cards.append(
                    ParentDecompositionCard(
                        parent_statement_id=parent_id,
                        parent_epistemic_role=parent.epistemic_role,
                        parent_claim_kind=parent.claim_kind,
                        parent_paper_ids=list(parent.paper_ids),
                        action="skipped_not_selected",
                        family_count=diagnostic.evidence_family_count,
                    )
                )
                continue

            if parent.epistemic_role != "evidence_synthesis":
                parent_cards.append(
                    ParentDecompositionCard(
                        parent_statement_id=parent_id,
                        parent_epistemic_role=parent.epistemic_role,
                        parent_claim_kind=parent.claim_kind,
                        parent_paper_ids=list(parent.paper_ids),
                        action="skipped_parent_not_evidence_synthesis",
                        family_count=diagnostic.evidence_family_count,
                    )
                )
                continue

            children = [
                _child_statement(
                    packet,
                    parent=parent,
                    family=family,
                )
                for family in diagnostic.evidence_families
            ]
            child_ids = [
                child.statement_id
                for child in children
            ]

            collisions = (
                set(child_ids) & existing_ids
            )
            if collisions:
                raise ValueError(
                    "decomposition child ID collision: "
                    f"{sorted(collisions)}"
                )
            if len(set(child_ids)) != len(child_ids):
                raise ValueError(
                    f"{parent_id}: duplicate generated child IDs"
                )

            existing_ids.update(child_ids)
            child_statements.extend(children)

            parent_cards.append(
                ParentDecompositionCard(
                    parent_statement_id=parent_id,
                    parent_epistemic_role=parent.epistemic_role,
                    parent_claim_kind=parent.claim_kind,
                    parent_paper_ids=list(parent.paper_ids),
                    action="expanded_with_family_children",
                    family_count=diagnostic.evidence_family_count,
                    child_statement_ids=child_ids,
                    child_count=len(child_ids),
                )
            )

        pre_path_context = context.model_copy(
            update={
                "evidence_statements": (
                    source_statements
                    + child_statements
                )
            }
        )
        pre_path_context = _context_rehash(
            pre_path_context
        )

        output_context, propagation = (
            MinimalPathLineagePropagator().propagate(
                packet,
                pre_path_context,
            )
        )

        output_by_id = {
            row.statement_id: row
            for row in output_context.evidence_statements
        }

        original_changed = 0
        original_missing = 0
        for original in source_statements:
            after = output_by_id.get(
                original.statement_id
            )
            if after is None:
                original_missing += 1
                continue
            if (
                after.model_dump(mode="json")
                != original.model_dump(mode="json")
            ):
                original_changed += 1

        # PL1-B preserves existing explicit paths, so all original statements
        # in a PL1-B context must remain byte-for-byte unchanged.
        if original_changed or original_missing:
            raise RuntimeError(
                "EC2-B changed or removed original context statements; "
                f"changed={original_changed}, missing={original_missing}"
            )

        propagation_cards = {
            row.statement_id: row
            for row in propagation.statement_cards
        }

        child_lineages: list[
            FamilyChildLineage
        ] = []
        for parent_id in sorted(selected_ids):
            diagnostic = candidate_rows[parent_id]
            parent = source_by_id[parent_id]
            if parent.epistemic_role != "evidence_synthesis":
                continue

            for family in diagnostic.evidence_families:
                child_id = _child_statement_id(
                    parent_statement_id=parent_id,
                    family_id=family.family_id,
                )
                child = output_by_id[child_id]
                propagation_card = propagation_cards[
                    child_id
                ]

                child_lineages.append(
                    FamilyChildLineage(
                        parent_statement_id=parent_id,
                        family_id=family.family_id,
                        child_statement_id=child_id,
                        paper_ids=list(family.paper_ids),
                        node_types=list(family.node_types),
                        edge_relations=list(family.edge_relations),
                        scientific_support_node_ids=list(
                            child.scientific_support_node_ids
                        ),
                        scientific_support_edge_ids=list(
                            child.scientific_support_edge_ids
                        ),
                        support_path_ids=list(
                            child.support_path_ids
                        ),
                        epistemic_role=child.epistemic_role,
                        claim_kind=child.claim_kind,
                        eligible_as_premise=(
                            child.eligible_as_premise
                        ),
                        pl1b_minimum_cover_size=(
                            propagation_card.minimum_cover_size
                        ),
                    )
                )

        before_eligible = sum(
            bool(row.eligible_as_premise)
            for row in source_statements
        )
        after_eligible = sum(
            bool(row.eligible_as_premise)
            for row in output_context.evidence_statements
        )

        expanded_parents = sum(
            card.action
            == "expanded_with_family_children"
            for card in parent_cards
        )
        skipped_parents = len(parent_cards) - expanded_parents

        payload = {
            "schema_version": "evidence-family-decomposition-report-v1",
            "report_id": _stable_id(
                "evidence_family_decomposition_report",
                packet.packet_sha256,
                context.context_sha256,
                output_context.context_sha256,
                family_diagnostics.report_id,
            ),
            "source_packet_id": packet.packet_id,
            "source_packet_sha256": packet.packet_sha256,
            "source_context_id": context.context_id,
            "source_context_sha256_before": context.context_sha256,
            "output_context_sha256_after": output_context.context_sha256,
            "source_family_diagnostic_report_id": (
                family_diagnostics.report_id
            ),
            "source_family_diagnostic_report_sha256": (
                family_diagnostics.report_sha256
            ),
            "domain_profile_id": context.domain_profile_id,
            "candidate_statement_count": len(candidate_ids),
            "selected_candidate_statement_count": len(selected_ids),
            "expanded_parent_count": expanded_parents,
            "skipped_parent_count": skipped_parents,
            "original_statement_count": len(source_statements),
            "output_statement_count": len(
                output_context.evidence_statements
            ),
            "child_statement_count": len(child_lineages),
            "eligible_statement_count_before": before_eligible,
            "eligible_statement_count_after": after_eligible,
            "original_statement_changed_count": original_changed,
            "original_statement_missing_count": original_missing,
            "child_lineages": [
                row.model_dump(mode="json")
                for row in child_lineages
            ],
            "parent_cards": [
                row.model_dump(mode="json")
                for row in parent_cards
            ],
            "child_path_lineage_propagation_report_id": (
                propagation.report_id
            ),
            "child_path_lineage_propagation_report_sha256": (
                propagation.report_sha256
            ),
            "policy": EvidenceFamilyDecompositionPolicy().model_dump(
                mode="json"
            ),
        }

        report = EvidenceFamilyDecompositionReport(
            **payload,
            report_sha256=_sha256_json(payload),
        )
        return (
            output_context,
            report,
            propagation,
        )
