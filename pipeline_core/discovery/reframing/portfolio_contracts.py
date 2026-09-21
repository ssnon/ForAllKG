from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.critic_contracts import (
    CRITIC_DIMENSIONS,
    ReframeCriticDimension,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    ImplementedReframeOperatorId,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


PortfolioSlotId = Literal[
    "A_RELATIONAL_EVIDENCE_NEAR",
    "B_STRUCTURAL_REFRAME",
    "C_MEASUREMENT_REGIME_CHALLENGE",
    "D_HIGH_RISK_EXPLORATORY",
]

ParetoStatus = Literal[
    "non_dominated",
    "dominated",
    "not_evaluable",
]

PortfolioRouteKind = Literal[
    "structural_reframe",
    "measurement_regime_challenge",
    "high_risk_exploratory",
    "not_evaluable",
]

RiskFlag = Literal[
    "incomplete_critic_vector",
    "invalid_operator_shape",
    "ungrounded_premise_reference",
    "ungrounded_gap_reference",
    "low_operator_validity",
    "low_premise_fidelity",
    "low_over_specificity_safety",
    "low_reframe_depth",
]


class PortfolioVectorEntry(StrictModel):
    dimension: ReframeCriticDimension
    rating: int | None = Field(default=None, ge=0, le=3)


class ReframePortfolioRouteSignals(StrictModel):
    structural_followup_signal: bool = False
    measurement_regime_signal: bool = False
    high_risk_signal: bool = False


class ReframePortfolioAssessment(StrictModel):
    candidate_id: str = Field(min_length=1)
    operator_id: ImplementedReframeOperatorId
    review_id: str | None = None
    vector: list[PortfolioVectorEntry]
    vector_complete: bool
    pareto_status: ParetoStatus
    dominated_by_candidate_ids: list[str] = Field(default_factory=list)
    route_kind: PortfolioRouteKind
    assigned_slot_id: PortfolioSlotId | None = None
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    route_signals: ReframePortfolioRouteSignals
    routing_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_vector(self) -> "ReframePortfolioAssessment":
        dimensions = tuple(row.dimension for row in self.vector)
        if dimensions != CRITIC_DIMENSIONS:
            raise ValueError("portfolio vector must use canonical critic dimension order")
        if self.vector_complete != all(row.rating is not None for row in self.vector):
            raise ValueError("vector_complete must match vector ratings")
        if self.pareto_status == "not_evaluable" and self.dominated_by_candidate_ids:
            raise ValueError("not-evaluable candidates cannot have Pareto dominators")
        if self.route_kind == "not_evaluable" and self.assigned_slot_id is not None:
            raise ValueError("not-evaluable candidates must remain unassigned")
        return self


class ReframePortfolioSlot(StrictModel):
    slot_id: PortfolioSlotId
    label: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    candidate_ids: list[str] = Field(default_factory=list)
    empty_reason: str | None = None

    @model_validator(mode="after")
    def validate_empty_reason(self) -> "ReframePortfolioSlot":
        if self.candidate_ids and self.empty_reason is not None:
            raise ValueError("non-empty portfolio slots must not carry empty_reason")
        if not self.candidate_ids and not (self.empty_reason or "").strip():
            raise ValueError("empty portfolio slots require empty_reason")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("portfolio slot candidate IDs must be unique")
        return self


class ScientificReframeShadowPortfolio(StrictModel):
    schema_version: Literal[
        "scientific-reframe-shadow-portfolio-v1"
    ] = "scientific-reframe-shadow-portfolio-v1"

    portfolio_id: str = Field(min_length=1)
    source_shadow_report_id: str = Field(min_length=1)
    source_critic_report_id: str = Field(min_length=1)
    source_task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)

    assessments: list[ReframePortfolioAssessment]
    slots: list[ReframePortfolioSlot]
    non_dominated_candidate_ids: list[str] = Field(default_factory=list)
    dominated_candidate_ids: list[str] = Field(default_factory=list)
    not_evaluable_candidate_ids: list[str] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    pareto_comparison_performed: Literal[True] = True
    slot_routing_performed: Literal[True] = True
    route_signals_evaluated: Literal[True] = True
    relational_lane_integrated: Literal[False] = False
    candidate_ranking_performed: Literal[False] = False
    overall_score_computed: Literal[False] = False
    winner_selected: Literal[False] = False
    production_selection_changed: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    n10_run: Literal[False] = False
    legacy_portfolio_mutated: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_partition(self) -> "ScientificReframeShadowPortfolio":
        candidate_ids = [row.candidate_id for row in self.assessments]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("portfolio assessment candidate IDs must be unique")

        expected_slots: tuple[PortfolioSlotId, ...] = (
            "A_RELATIONAL_EVIDENCE_NEAR",
            "B_STRUCTURAL_REFRAME",
            "C_MEASUREMENT_REGIME_CHALLENGE",
            "D_HIGH_RISK_EXPLORATORY",
        )
        if tuple(row.slot_id for row in self.slots) != expected_slots:
            raise ValueError("portfolio slots must use canonical A/B/C/D order")

        classified = (
            set(self.non_dominated_candidate_ids)
            | set(self.dominated_candidate_ids)
            | set(self.not_evaluable_candidate_ids)
        )
        if classified != set(candidate_ids):
            raise ValueError("Pareto status lists must partition all assessments")
        if (
            set(self.non_dominated_candidate_ids)
            & set(self.dominated_candidate_ids)
            or set(self.non_dominated_candidate_ids)
            & set(self.not_evaluable_candidate_ids)
            or set(self.dominated_candidate_ids)
            & set(self.not_evaluable_candidate_ids)
        ):
            raise ValueError("Pareto status lists must be disjoint")

        slot_members = {
            candidate_id
            for slot in self.slots
            for candidate_id in slot.candidate_ids
        }
        assigned = {
            row.candidate_id
            for row in self.assessments
            if row.assigned_slot_id is not None
        }
        if slot_members != assigned:
            raise ValueError("slot membership must match assigned assessments")
        return self


def stable_shadow_portfolio_id(
    *,
    shadow_report_id: str,
    critic_report_id: str,
    assessment_ids: list[str],
) -> str:
    payload = json.dumps(
        [shadow_report_id, critic_report_id, *sorted(assessment_ids)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"reframe_shadow_portfolio:{digest}"
