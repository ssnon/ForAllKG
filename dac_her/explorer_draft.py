from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from dac_her.explorer_contracts import ClaimKind, EpistemicRole


class StrictDraftModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExplorerStatementDraft(StrictDraftModel):
    local_id: str
    text: str
    epistemic_role: EpistemicRole
    claim_kind: ClaimKind
    support_node_ids: list[str] = Field(default_factory=list)
    support_edge_ids: list[str] = Field(default_factory=list)
    support_path_ids: list[str] = Field(default_factory=list)
    support_direct_hit_ids: list[str] = Field(default_factory=list)


class MechanismRouteDraft(StrictDraftModel):
    local_id: str
    path_ids: list[str] = Field(min_length=1)
    statement_local_ids: list[str] = Field(min_length=1)


class MechanisticMotifDraft(StrictDraftModel):
    local_id: str
    label: str
    statement_local_ids: list[str] = Field(min_length=1)
    path_ids: list[str] = Field(default_factory=list)
    support_node_ids: list[str] = Field(default_factory=list)
    support_edge_ids: list[str] = Field(default_factory=list)


class CrossPaperConnectionDraft(StrictDraftModel):
    local_id: str
    statement_local_ids: list[str] = Field(min_length=1)
    path_ids: list[str] = Field(min_length=1)


class EvidenceTensionDraft(StrictDraftModel):
    local_id: str
    statement_local_id: str
    side_a_statement_local_ids: list[str] = Field(min_length=1)
    side_b_statement_local_ids: list[str] = Field(min_length=1)
    tension_type: Literal[
        "context_dependency",
        "qualitative_difference",
        "quantitative_difference",
        "potential_conflict",
        "insufficient_context",
    ]


class UnresolvedConnectionDraft(StrictDraftModel):
    local_id: str
    statement_local_id: str
    related_path_ids: list[str] = Field(default_factory=list)
    reason: Literal[
        "alignment_only",
        "navigation_heavy",
        "candidate_only",
        "missing_direct_relation_in_packet",
        "insufficient_provenance",
        "partial_source_scope",
    ]


class ReportedDesignLeverDraft(StrictDraftModel):
    local_id: str
    label: str
    statement_local_ids: list[str] = Field(min_length=1)
    mechanism_node_ids: list[str] = Field(default_factory=list)
    outcome_node_ids: list[str] = Field(default_factory=list)


class ExplorationDraft(StrictDraftModel):
    """Minimal LLM-owned representation for Graph Explorer v2.5.1.

    Scientific bookkeeping that can be derived from the frozen packet (paper IDs,
    verification propagation, path type, alignment/reverse flags, mechanism node/
    edge membership, stable IDs) is intentionally absent.  The compiler owns it.
    """

    statements: list[ExplorerStatementDraft]
    direct_finding_local_ids: list[str] = Field(default_factory=list)
    mechanism_routes: list[MechanismRouteDraft] = Field(default_factory=list)
    recurring_mechanistic_motifs: list[MechanisticMotifDraft] = Field(default_factory=list)
    cross_paper_connections: list[CrossPaperConnectionDraft] = Field(default_factory=list)
    evidence_tensions: list[EvidenceTensionDraft] = Field(default_factory=list)
    unresolved_connections: list[UnresolvedConnectionDraft] = Field(default_factory=list)
    reported_design_levers: list[ReportedDesignLeverDraft] = Field(default_factory=list)
