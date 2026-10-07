from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.research_idea_active_realization import (
    ActiveRealizationSearchReport,
    BoundaryEscalationTrace,
    ExactFeedbackAttachmentReport,
)
from pipeline_core.discovery.research_idea_closed_loop import RealizationLifecycleReport
from pipeline_core.discovery.research_idea_family_calibration import (
    FamilyCalibrationReport,
    ProgramFamilyComputeReport,
)
from pipeline_core.discovery.research_idea_offspring_execution import OffspringExecutionReport
from pipeline_core.discovery.research_idea_population_semantics import ScientificFeedbackFacetObservation


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ContinuityOutcomeV2 = Literal[
    "RESCUED_WITHIN_IDEA",
    "RESCUED_BY_CHILD",
    "PRODUCTIVE_LINEAGE",
    "UNRESOLVED",
    "NOT_OBSERVED",
]


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical(parts).encode('utf-8')).hexdigest()[:20]}"


def _usable(row: ScientificFeedbackFacetObservation) -> bool:
    # v2.6 credit only treats a realization as usable after prospective
    # evaluation completed with intact lineage. NOT_EVALUATED is neither
    # success nor failure and therefore must not inflate rescue/productivity.
    return bool(
        row.materialization_status == "MATERIALIZED"
        and row.prospective_status == "COMPLETE"
        and row.prospective_contract_integrity_passed is True
        and row.prospective_identifiability != "NOT_OPERATIONALIZABLE"
    )


class CreditContinuityTraceV2(StrictModel):
    trace_id: str
    generation2_idea_id: str
    generation2_usable_before_active_search: bool
    generation2_rescued_by_active_local_search: bool
    generation2_usable_after_active_search: bool
    generation3_child_idea_ids: list[str] = Field(default_factory=list)
    generation3_usable_child_idea_ids: list[str] = Field(default_factory=list)
    generation3_locally_rescued_child_idea_ids: list[str] = Field(default_factory=list)
    outcome: ContinuityOutcomeV2
    reason_codes: list[str] = Field(default_factory=list)
    active_search_causality_not_assumed_for_existing_children: Literal[True] = True
    credit_continuity_is_diagnostic_not_truth: Literal[True] = True


class CreditContinuityReportV2(StrictModel):
    schema_version: Literal[
        "research-idea-credit-continuity-shadow-v2"
    ] = "research-idea-credit-continuity-shadow-v2"
    report_id: str
    report_sha256: str
    traces: list[CreditContinuityTraceV2] = Field(default_factory=list)
    outcome_counts: dict[str, int] = Field(default_factory=dict)
    generation2_same_idea_rescue_count: int = Field(ge=0)
    generation3_same_idea_rescue_count: int = Field(ge=0)
    total_same_idea_rescue_count: int = Field(ge=0)
    child_rescue_count: int = Field(ge=0)
    productive_lineage_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    local_rescue_precedes_global_escalation: Literal[True] = True
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_rescues(self) -> "CreditContinuityReportV2":
        if self.total_same_idea_rescue_count != (
            self.generation2_same_idea_rescue_count
            + self.generation3_same_idea_rescue_count
        ):
            raise ValueError("total_same_idea_rescue_count mismatch")
        return self


class ActiveClosedLoopCaseSummary(StrictModel):
    schema_version: Literal[
        "sis-v2-6-family-active-realization-case-summary-v1"
    ] = "sis-v2-6-family-active-realization-case-summary-v1"
    summary_id: str
    summary_sha256: str
    total_research_idea_nodes: int = Field(ge=0)
    tight_neighborhood_count: int = Field(ge=0)
    program_family_count: int = Field(ge=0)
    family_fragmentation_ratio: float = Field(ge=0.0, le=1.0)
    program_family_compute_hhi: float = Field(ge=0.0, le=1.0)
    largest_program_family_compute_share: float = Field(ge=0.0, le=1.0)
    overconcentrated_program_family_count: int = Field(ge=0)
    generation2_active_target_count: int = Field(ge=0)
    generation3_active_target_count: int = Field(ge=0)
    generation2_active_same_idea_rescue_count: int = Field(ge=0)
    generation3_active_same_idea_rescue_count: int = Field(ge=0)
    total_active_same_idea_rescue_count: int = Field(ge=0)
    active_local_action_counts: dict[str, int] = Field(default_factory=dict)
    boundary_escalation_resolution_counts: dict[str, int] = Field(default_factory=dict)
    prospective_feedback_evaluated_count: int = Field(ge=0)
    prospective_feedback_not_evaluated_count: int = Field(ge=0)
    residual_feedback_evaluated_count: int = Field(ge=0)
    residual_feedback_not_evaluated_count: int = Field(ge=0)
    credit_continuity_outcome_counts: dict[str, int] = Field(default_factory=dict)
    child_rescue_count: int = Field(ge=0)
    productive_lineage_count: int = Field(ge=0)
    unresolved_credit_count: int = Field(ge=0)
    local_llm_call_count: int = Field(ge=0)
    prospective_audit_llm_call_count: int = Field(ge=0)
    not_evaluated_is_not_negative_feedback: Literal[True] = True
    family_is_soft_and_recomputable: Literal[True] = True
    family_is_not_hard_gate: Literal[True] = True
    axis_mutation_is_boundary_sensitive: Literal[True] = True
    no_g4_generation: Literal[True] = True
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class ActiveClosedLoopCaseReport(StrictModel):
    family_calibration: FamilyCalibrationReport
    family_compute: ProgramFamilyComputeReport
    generation2_feedback: ExactFeedbackAttachmentReport
    generation3_feedback: ExactFeedbackAttachmentReport
    generation2_active: ActiveRealizationSearchReport
    generation3_active: ActiveRealizationSearchReport
    boundary_traces: list[BoundaryEscalationTrace] = Field(default_factory=list)
    credit_continuity: CreditContinuityReportV2
    summary: ActiveClosedLoopCaseSummary


def _usable_by_idea(
    *,
    lifecycle: RealizationLifecycleReport,
    active: ActiveRealizationSearchReport,
) -> dict[str, bool]:
    out: dict[str, bool] = defaultdict(bool)
    for row in [*lifecycle.observations, *active.new_observations]:
        out[row.idea_id] = out[row.idea_id] or _usable(row)
    return dict(out)


def build_credit_continuity_v2(
    *,
    generation2_execution: OffspringExecutionReport,
    generation2_lifecycle: RealizationLifecycleReport,
    generation2_active: ActiveRealizationSearchReport,
    generation3_execution: OffspringExecutionReport,
    generation3_lifecycle: RealizationLifecycleReport,
    generation3_active: ActiveRealizationSearchReport,
) -> CreditContinuityReportV2:
    g2_before = {
        row.idea_id: row.usable_grounded_realization_count > 0
        for row in generation2_lifecycle.local_states
    }
    g2_after = _usable_by_idea(lifecycle=generation2_lifecycle, active=generation2_active)
    g3_after = _usable_by_idea(lifecycle=generation3_lifecycle, active=generation3_active)
    g2_rescued = set(generation2_active.rescued_within_same_idea_ids)
    g3_rescued = set(generation3_active.rescued_within_same_idea_ids)

    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for child in generation3_execution.offspring_nodes:
        for parent_id in child.parent_idea_ids:
            children_by_parent[parent_id].append(child.idea_id)

    g2_ids = {row.idea_id for row in generation2_execution.offspring_nodes}
    traces: list[CreditContinuityTraceV2] = []
    for idea_id in sorted(g2_ids):
        children = sorted(set(children_by_parent.get(idea_id, [])))
        usable_children = [child_id for child_id in children if g3_after.get(child_id, False)]
        locally_rescued_children = [child_id for child_id in children if child_id in g3_rescued]
        before = g2_before.get(idea_id, False)
        after = g2_after.get(idea_id, before)
        local_rescue = idea_id in g2_rescued
        reasons: list[str] = []
        if local_rescue:
            outcome: ContinuityOutcomeV2 = "RESCUED_WITHIN_IDEA"
            reasons.append("G2_ACTIVE_LOCAL_SEARCH_RECOVERED_SAME_RESEARCH_IDEA")
        elif not after and usable_children:
            outcome = "RESCUED_BY_CHILD"
            reasons.append("G2_UNRESOLVED_BUT_EXISTING_G3_CHILD_HAS_USABLE_REALIZATION")
        elif after and usable_children:
            outcome = "PRODUCTIVE_LINEAGE"
            reasons.append("G2_AND_G3_CHILD_BOTH_HAVE_USABLE_REALIZATIONS")
        elif idea_id not in g2_after and idea_id not in g2_before:
            outcome = "NOT_OBSERVED"
            reasons.append("G2_IDEA_NOT_REALIZED_UNDER_CURRENT_BUDGET")
        else:
            outcome = "UNRESOLVED"
            reasons.append("NO_USABLE_LOCAL_OR_CHILD_REALIZATION_OBSERVED")
        traces.append(
            CreditContinuityTraceV2(
                trace_id=_stable_id("credit_continuity_v2", idea_id, children, outcome),
                generation2_idea_id=idea_id,
                generation2_usable_before_active_search=before,
                generation2_rescued_by_active_local_search=local_rescue,
                generation2_usable_after_active_search=after,
                generation3_child_idea_ids=children,
                generation3_usable_child_idea_ids=usable_children,
                generation3_locally_rescued_child_idea_ids=locally_rescued_children,
                outcome=outcome,
                reason_codes=reasons,
            )
        )

    counts = Counter(row.outcome for row in traces)
    provisional = CreditContinuityReportV2(
        report_id="pending",
        report_sha256="pending",
        traces=traces,
        outcome_counts=dict(sorted(counts.items())),
        generation2_same_idea_rescue_count=len(g2_rescued),
        generation3_same_idea_rescue_count=len(g3_rescued),
        total_same_idea_rescue_count=len(g2_rescued) + len(g3_rescued),
        child_rescue_count=counts["RESCUED_BY_CHILD"],
        productive_lineage_count=counts["PRODUCTIVE_LINEAGE"],
        unresolved_count=counts["UNRESOLVED"],
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"credit_continuity_v2:{digest[:20]}",
            "report_sha256": digest,
        }
    )


def build_active_closed_loop_case_report(
    *,
    family_calibration: FamilyCalibrationReport,
    family_compute: ProgramFamilyComputeReport,
    generation2_feedback: ExactFeedbackAttachmentReport,
    generation3_feedback: ExactFeedbackAttachmentReport,
    generation2_active: ActiveRealizationSearchReport,
    generation3_active: ActiveRealizationSearchReport,
    boundary_traces: Sequence[BoundaryEscalationTrace],
    credit_continuity: CreditContinuityReportV2,
) -> ActiveClosedLoopCaseReport:
    action_counts = Counter()
    for report in (generation2_active, generation3_active):
        action_counts.update(report.action_counts)
    boundary_counts = Counter(row.resolution for row in boundary_traces)
    idea_count = family_calibration.idea_count
    fragmentation = (
        family_calibration.program_family_count / idea_count if idea_count else 0.0
    )
    summary_provisional = ActiveClosedLoopCaseSummary(
        summary_id="pending",
        summary_sha256="pending",
        total_research_idea_nodes=idea_count,
        tight_neighborhood_count=family_calibration.tight_neighborhood_count,
        program_family_count=family_calibration.program_family_count,
        family_fragmentation_ratio=fragmentation,
        program_family_compute_hhi=family_compute.family_compute_hhi,
        largest_program_family_compute_share=family_compute.largest_family_compute_share,
        overconcentrated_program_family_count=len(family_compute.overconcentrated_family_keys),
        generation2_active_target_count=len(generation2_active.target_idea_ids),
        generation3_active_target_count=len(generation3_active.target_idea_ids),
        generation2_active_same_idea_rescue_count=len(
            generation2_active.rescued_within_same_idea_ids
        ),
        generation3_active_same_idea_rescue_count=len(
            generation3_active.rescued_within_same_idea_ids
        ),
        total_active_same_idea_rescue_count=credit_continuity.total_same_idea_rescue_count,
        active_local_action_counts=dict(sorted(action_counts.items())),
        boundary_escalation_resolution_counts=dict(sorted(boundary_counts.items())),
        prospective_feedback_evaluated_count=(
            generation2_feedback.prospective_evaluated_count
            + generation3_feedback.prospective_evaluated_count
        ),
        prospective_feedback_not_evaluated_count=(
            generation2_feedback.prospective_not_evaluated_count
            + generation3_feedback.prospective_not_evaluated_count
        ),
        residual_feedback_evaluated_count=(
            generation2_feedback.residual_evaluated_count
            + generation3_feedback.residual_evaluated_count
        ),
        residual_feedback_not_evaluated_count=(
            generation2_feedback.residual_not_evaluated_count
            + generation3_feedback.residual_not_evaluated_count
        ),
        credit_continuity_outcome_counts=credit_continuity.outcome_counts,
        child_rescue_count=credit_continuity.child_rescue_count,
        productive_lineage_count=credit_continuity.productive_lineage_count,
        unresolved_credit_count=credit_continuity.unresolved_count,
        local_llm_call_count=(
            generation2_active.local_llm_call_count
            + generation3_active.local_llm_call_count
        ),
        prospective_audit_llm_call_count=(
            generation2_active.prospective_audit_llm_call_count
            + generation3_active.prospective_audit_llm_call_count
        ),
    )
    payload = summary_provisional.model_dump(mode="json")
    payload.pop("summary_id", None)
    payload.pop("summary_sha256", None)
    digest = _sha(payload)
    summary = summary_provisional.model_copy(
        update={
            "summary_id": f"sis_v2_6_case:{digest[:20]}",
            "summary_sha256": digest,
        }
    )
    return ActiveClosedLoopCaseReport(
        family_calibration=family_calibration,
        family_compute=family_compute,
        generation2_feedback=generation2_feedback,
        generation3_feedback=generation3_feedback,
        generation2_active=generation2_active,
        generation3_active=generation3_active,
        boundary_traces=list(boundary_traces),
        credit_continuity=credit_continuity,
        summary=summary,
    )


__all__ = [
    "ActiveClosedLoopCaseReport",
    "ActiveClosedLoopCaseSummary",
    "CreditContinuityReportV2",
    "CreditContinuityTraceV2",
    "build_active_closed_loop_case_report",
    "build_credit_continuity_v2",
]
