from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.research_idea_active_realization import (
    ActiveRealizationAction,
    ActiveRealizationSearchReport,
    ExactFeedbackAttachmentReport,
    FeedbackAttachmentRecord,
)
from pipeline_core.discovery.research_idea_closed_loop import RealizationLifecycleReport
from pipeline_core.discovery.research_idea_closed_loop_v2_6 import CreditContinuityReportV2
from pipeline_core.discovery.research_idea_feedback_resolution import resolve_feedback_artifacts
from pipeline_core.discovery.research_idea_offspring_execution import OffspringExecutionReport
from pipeline_core.discovery.research_idea_population_semantics import (
    IdeaLocalSearchState,
    ScientificFeedbackFacetObservation,
)
from pipeline_core.discovery.research_idea_program_families import ScientificProgramFamilyReport


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical(parts).encode('utf-8')).hexdigest()[:20]}"


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _usable(row: ScientificFeedbackFacetObservation | None) -> bool:
    return bool(
        row is not None
        and row.materialization_status == "MATERIALIZED"
        and row.prospective_status == "COMPLETE"
        and row.prospective_contract_integrity_passed is True
        and row.prospective_identifiability != "NOT_OPERATIONALIZABLE"
    )


FreshVerificationStatus = Literal[
    "COMPLETE",
    "SKIPPED_EMPTY_PORTFOLIO",
    "FAILED_OPERATIONAL",
    "PAUSED_PROVIDER_BUDGET_EXHAUSTED",
]
MutationScope = Literal[
    "OPERATIONALIZATION_ESCAPE",
    "KNOWN_REGION_ESCAPE",
    "EVIDENCE_GAP_ESCAPE",
    "TOPOLOGY_ESCAPE",
    "GENERAL_LOCAL_EXHAUSTION",
]


class FreshResidualVerificationReport(StrictModel):
    schema_version: Literal[
        "sis-v2-7-fresh-residual-verification-v1"
    ] = "sis-v2-7-fresh-residual-verification-v1"
    report_id: str
    report_sha256: str
    generation_index: int = Field(ge=2)
    round_index: int = Field(ge=0)
    source_portfolio_id: str
    hypothesis_count: int = Field(ge=0)
    status: FreshVerificationStatus
    residual_state_counts: dict[str, int] = Field(default_factory=dict)
    epistemic_state_path: str | None = None
    external_report_path: str | None = None
    query_plan_path: str | None = None
    aggregation_path: str | None = None
    provider_plan_path: str | None = None
    stage_return_codes: dict[str, int] = Field(default_factory=dict)
    verification_subprocess_call_count: int = Field(ge=0)
    fresh_exact_lineage_residual_created: bool = False
    operational_failure_is_not_scientific_failure: Literal[True] = True
    residual_is_not_truth_authority: Literal[True] = True
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class VerifiedLocalRoundRecord(StrictModel):
    generation_index: int = Field(ge=2)
    round_index: int = Field(ge=1)
    source_lifecycle_report_id: str
    source_portfolio_id: str
    feedback_report_id: str
    active_report_id: str
    output_lifecycle_report_id: str
    output_portfolio_id: str
    action_counts: dict[str, int] = Field(default_factory=dict)
    generated_realization_count: int = Field(ge=0)
    same_idea_rescue_count: int = Field(ge=0)
    local_llm_call_count: int = Field(ge=0)
    prospective_audit_llm_call_count: int = Field(ge=0)
    fresh_verification_report_id: str | None = None


class FailureObservationSnapshot(StrictModel):
    hypothesis_id: str | None = None
    prospective_identifiability: str | None = None
    residual_epistemic_state: str | None = None
    residual_state_reason: str | None = None
    current_evidence_status: str | None = None


class IdeaEvolutionRequest(StrictModel):
    schema_version: Literal[
        "idea-evolution-request-v1"
    ] = "idea-evolution-request-v1"
    request_id: str
    source_idea_id: str
    source_generation_index: int = Field(ge=2)
    source_realization_ids: list[str] = Field(default_factory=list)
    exhausted_local_actions: list[ActiveRealizationAction] = Field(default_factory=list)
    typed_failure_observations: list[FailureObservationSnapshot] = Field(default_factory=list)
    requested_mutation_scope: MutationScope
    recommended_channels: list[Literal["TRANSFORM", "EXPLORE", "WILDCARD"]] = Field(default_factory=list)
    nonbinding_operator_hints: list[str] = Field(default_factory=list)
    program_family_key: str | None = None
    program_family_size: int | None = Field(default=None, ge=1)
    reason_codes: list[str] = Field(default_factory=list)
    local_realization_space_was_attempted_first: Literal[True] = True
    operator_hint_is_not_idea_identity_authority: Literal[True] = True
    request_executes_g4_generation: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class IdeaEvolutionRequestReport(StrictModel):
    schema_version: Literal[
        "idea-evolution-request-report-v1"
    ] = "idea-evolution-request-report-v1"
    report_id: str
    report_sha256: str
    requests: list[IdeaEvolutionRequest] = Field(default_factory=list)
    request_count: int = Field(ge=0)
    scope_counts: dict[str, int] = Field(default_factory=dict)
    source_generation_counts: dict[str, int] = Field(default_factory=dict)
    g4_generation_executed: Literal[False] = False
    production_generation_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_count(self) -> "IdeaEvolutionRequestReport":
        if self.request_count != len(self.requests):
            raise ValueError("request_count mismatch")
        return self


class ResidualActionOutcome(StrictModel):
    generation_index: int = Field(ge=2)
    idea_id: str
    pre_residual_state: str | None = None
    action: str
    post_residual_state: str | None = None
    usable_after_action: bool
    changed_residual_state: bool


class CostAwareClosedLoopAudit(StrictModel):
    schema_version: Literal[
        "sis-v2-7-cost-aware-closed-loop-audit-v1"
    ] = "sis-v2-7-cost-aware-closed-loop-audit-v1"
    local_llm_calls: int = Field(ge=0)
    prospective_audit_llm_calls: int = Field(ge=0)
    fresh_verification_barrier_runs: int = Field(ge=0)
    fresh_verification_subprocess_calls: int = Field(ge=0)
    same_idea_rescue_count: int = Field(ge=0)
    same_idea_rescue_per_local_llm_call: float = Field(ge=0.0)
    existing_child_rescue_count: int = Field(ge=0)
    g3_offspring_llm_calls: int = Field(ge=0)
    existing_child_rescue_per_g3_offspring_llm_call: float = Field(ge=0.0)
    unresolved_before: int = Field(ge=0)
    unresolved_after: int = Field(ge=0)
    unresolved_reduction: int
    program_family_count: int = Field(ge=0)
    family_fragmentation_ratio: float = Field(ge=0.0, le=1.0)
    residual_action_outcomes: list[ResidualActionOutcome] = Field(default_factory=list)
    scalar_fitness_used: Literal[False] = False
    cost_metrics_are_diagnostic_only: Literal[True] = True
    production_selection_authority: Literal[False] = False


class VerifiedActiveSearchCaseSummary(StrictModel):
    schema_version: Literal[
        "sis-v2-7-verified-active-search-case-summary-v1"
    ] = "sis-v2-7-verified-active-search-case-summary-v1"
    summary_id: str
    summary_sha256: str
    research_idea_count: int = Field(ge=0)
    scientific_program_family_count: int = Field(ge=0)
    family_fragmentation_ratio: float = Field(ge=0.0, le=1.0)
    family_reference_absolute_error: int | None = Field(default=None, ge=0)
    generation2_verified_round_count: int = Field(ge=0)
    generation3_verified_round_count: int = Field(ge=0)
    generation2_same_idea_rescue_count: int = Field(ge=0)
    generation3_same_idea_rescue_count: int = Field(ge=0)
    total_same_idea_rescue_count: int = Field(ge=0)
    initial_residual_evaluated_count: int = Field(ge=0)
    final_residual_evaluated_count: int = Field(ge=0)
    idea_evolution_request_count: int = Field(ge=0)
    idea_evolution_request_scope_counts: dict[str, int] = Field(default_factory=dict)
    cost_audit: CostAwareClosedLoopAudit
    g4_generation_executed: Literal[False] = False
    family_is_hard_gate: Literal[False] = False
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


def _strict_local_states(
    lifecycle: RealizationLifecycleReport,
    *,
    local_search_budget: int,
) -> list[IdeaLocalSearchState]:
    links_by_idea: dict[str, list[Any]] = defaultdict(list)
    for link in lifecycle.links:
        links_by_idea[link.idea_id].append(link)
    obs_by_realization = {row.realization_id: row for row in lifecycle.observations}
    out: list[IdeaLocalSearchState] = []
    for idea_id in lifecycle.target_idea_ids:
        links = sorted(links_by_idea.get(idea_id, []), key=lambda row: (row.attempt_index, row.realization_id))
        observations = [obs_by_realization.get(link.realization_id) for link in links]
        usable_flags = [_usable(row) for row in observations]
        evaluated_flags = [
            bool(row is not None and row.prospective_status == "COMPLETE")
            for row in observations
        ]
        usable = sum(usable_flags)
        materialized = sum(link.materialization_status == "MATERIALIZED" for link in links)
        not_op = sum(
            bool(row is not None and row.prospective_identifiability == "NOT_OPERATIONALIZABLE")
            for row in observations
        )
        failed = sum(link.materialization_status != "MATERIALIZED" for link in links)
        rescued = bool(usable and not (usable_flags[0] if usable_flags else False))
        used = len(links)
        exhausted = bool(
            usable == 0
            and used >= local_search_budget
            and (all(evaluated_flags) if evaluated_flags else True)
        )
        out.append(
            IdeaLocalSearchState(
                idea_id=idea_id,
                realization_ids=[row.realization_id for row in links],
                realization_count=len(links),
                materialized_count=materialized,
                usable_grounded_realization_count=usable,
                not_operationalizable_count=not_op,
                failed_or_abstained_count=failed,
                adaptive_event_ids=[],
                local_search_budget=max(local_search_budget, used, 1),
                local_search_budget_used=used,
                local_search_exhausted=exhausted,
                rescued_within_same_idea=rescued,
            )
        )
    return out


def normalize_lifecycle_strict(
    lifecycle: RealizationLifecycleReport,
    *,
    extra_local_budget: int = 0,
) -> RealizationLifecycleReport:
    max_links = max(
        (sum(link.idea_id == idea_id for link in lifecycle.links) for idea_id in lifecycle.target_idea_ids),
        default=0,
    )
    budget = max(lifecycle.max_realizations_per_idea + extra_local_budget, max_links, 1)
    states = _strict_local_states(lifecycle, local_search_budget=budget)
    provisional = lifecycle.model_copy(
        update={
            "report_id": "pending",
            "report_sha256": "pending",
            "local_states": states,
            "usable_grounded_realization_count": sum(row.usable_grounded_realization_count for row in states),
            "rescued_within_same_idea_count": sum(row.rescued_within_same_idea for row in states),
            "local_search_exhausted_count": sum(row.local_search_exhausted for row in states),
            "prospective_not_operationalizable_count": sum(row.not_operationalizable_count for row in states),
            "max_realizations_per_idea": budget,
        }
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"g{lifecycle.generation_index}_strict_lifecycle:{digest[:20]}",
            "report_sha256": digest,
        }
    )


def merge_active_into_lifecycle(
    *,
    lifecycle: RealizationLifecycleReport,
    active: ActiveRealizationSearchReport,
    output_portfolio: HypothesisPortfolio,
    remaining_round_budget: int = 0,
) -> RealizationLifecycleReport:
    if active.source_lifecycle_report_id != lifecycle.report_id:
        raise ValueError("active report / lifecycle lineage mismatch")
    links = [*lifecycle.links, *active.new_links]
    observations = [*lifecycle.observations, *active.new_observations]
    max_links = max(
        (sum(link.idea_id == idea_id for link in links) for idea_id in lifecycle.target_idea_ids),
        default=0,
    )
    budget = max(lifecycle.max_realizations_per_idea, max_links + remaining_round_budget, 1)
    template = lifecycle.model_copy(
        update={
            "report_id": "pending",
            "report_sha256": "pending",
            "links": links,
            "observations": observations,
            "output_portfolio_id": output_portfolio.portfolio_id,
            "materialized_hypothesis_count": sum(row.materialization_status == "MATERIALIZED" for row in links),
            "realization_llm_call_count": lifecycle.realization_llm_call_count + active.local_llm_call_count,
            "prospective_audit_llm_call_count": (
                lifecycle.prospective_audit_llm_call_count + active.prospective_audit_llm_call_count
            ),
            "prospective_audit_count": lifecycle.prospective_audit_count + active.prospective_audit_llm_call_count,
            "max_realizations_per_idea": budget,
        }
    )
    states = _strict_local_states(template, local_search_budget=budget)
    provisional = template.model_copy(
        update={
            "local_states": states,
            "usable_grounded_realization_count": sum(row.usable_grounded_realization_count for row in states),
            "rescued_within_same_idea_count": sum(row.rescued_within_same_idea for row in states),
            "local_search_exhausted_count": sum(row.local_search_exhausted for row in states),
            "prospective_not_operationalizable_count": sum(row.not_operationalizable_count for row in states),
        }
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"g{lifecycle.generation_index}_verified_active_lifecycle:{digest[:20]}",
            "report_sha256": digest,
        }
    )


def _materialization_adapter(lifecycle: RealizationLifecycleReport) -> Any:
    records = [
        SimpleNamespace(status=link.materialization_status, hypothesis_id=link.hypothesis_id)
        for link in lifecycle.links
        if link.hypothesis_id is not None
    ]
    return SimpleNamespace(
        report_id=lifecycle.report_id,
        output_portfolio_id=lifecycle.output_portfolio_id,
        source_context_id=lifecycle.source_context_id,
        records=records,
    )


def attach_feedback_with_fresh_residual(
    *,
    lifecycle: RealizationLifecycleReport,
    fresh_residual_state: Mapping[str, Any] | None,
    prospective_search_roots: Sequence[Path],
) -> ExactFeedbackAttachmentReport:
    if fresh_residual_state is not None:
        source = str(fresh_residual_state.get("source_portfolio_id") or "")
        if source != lifecycle.output_portfolio_id:
            raise ValueError("fresh residual state / lifecycle portfolio lineage mismatch")
    resolved = resolve_feedback_artifacts(
        materialization=_materialization_adapter(lifecycle),
        search_roots=prospective_search_roots,
    )
    idea_by_hypothesis = {
        str(link.hypothesis_id): link.idea_id
        for link in lifecycle.links
        if link.hypothesis_id is not None and link.materialization_status == "MATERIALIZED"
    }
    residual_by_h = {
        str(row.get("hypothesis_id")): row
        for row in (fresh_residual_state or {}).get("hypotheses", [])
        if isinstance(row, Mapping) and row.get("hypothesis_id")
    }
    ambiguous_prospective = set(resolved.report.prospective_ambiguous_hypothesis_ids)
    records: list[FeedbackAttachmentRecord] = []
    for hypothesis_id, idea_id in sorted(idea_by_hypothesis.items()):
        prospective = resolved.prospective_by_hypothesis.get(hypothesis_id)
        p_state = (
            "AMBIGUOUS"
            if hypothesis_id in ambiguous_prospective
            else "EVALUATED"
            if prospective is not None
            else "NOT_EVALUATED"
        )
        residual_row = residual_by_h.get(hypothesis_id)
        r_state = "EVALUATED" if fresh_residual_state is not None else "NOT_EVALUATED"
        records.append(
            FeedbackAttachmentRecord(
                hypothesis_id=hypothesis_id,
                idea_id=idea_id,
                prospective_evaluation_state=p_state,
                residual_evaluation_state=r_state,
                current_evidence_status=(
                    str(prospective.get("current_evidence_status"))
                    if prospective and prospective.get("current_evidence_status") is not None
                    else None
                ),
                prospective_identifiability=(
                    str(prospective.get("prospective_identifiability"))
                    if prospective and prospective.get("prospective_identifiability") is not None
                    else None
                ),
                directionality_mode=(
                    str(prospective.get("directionality_mode"))
                    if prospective and prospective.get("directionality_mode") is not None
                    else None
                ),
                measurement_compatibility_mode=(
                    str(prospective.get("measurement_compatibility_mode"))
                    if prospective and prospective.get("measurement_compatibility_mode") is not None
                    else None
                ),
                residual_epistemic_state=(
                    str(residual_row.get("final_epistemic_state"))
                    if residual_row and residual_row.get("final_epistemic_state") is not None
                    else None
                ),
                residual_state_reason=(
                    str(residual_row.get("state_reason"))
                    if residual_row and residual_row.get("state_reason") is not None
                    else None
                ),
            )
        )
    counts_p = Counter(row.prospective_evaluation_state for row in records)
    counts_r = Counter(row.residual_evaluation_state for row in records)
    provisional = ExactFeedbackAttachmentReport(
        report_id="pending",
        report_sha256="pending",
        source_lifecycle_report_id=lifecycle.report_id,
        search_roots=[str(path.expanduser().resolve()) for path in prospective_search_roots],
        records=records,
        prospective_evaluated_count=counts_p["EVALUATED"],
        prospective_not_evaluated_count=counts_p["NOT_EVALUATED"],
        prospective_ambiguous_count=counts_p["AMBIGUOUS"],
        residual_evaluated_count=counts_r["EVALUATED"],
        residual_not_evaluated_count=counts_r["NOT_EVALUATED"],
        residual_ambiguous_count=0,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"verified_exact_feedback:{digest[:20]}",
            "report_sha256": digest,
        }
    )


def prior_action_map(reports: Sequence[ActiveRealizationSearchReport]) -> dict[str, list[ActiveRealizationAction]]:
    out: dict[str, list[ActiveRealizationAction]] = defaultdict(list)
    for report in reports:
        for row in report.action_records:
            if row.action not in out[row.idea_id]:
                out[row.idea_id].append(row.action)
    return dict(out)


def _family_context(families: ScientificProgramFamilyReport) -> dict[str, tuple[str, int]]:
    return {
        row.idea_id: (row.program_family_key, row.program_family_size)
        for row in families.assignments
    }


def build_idea_evolution_requests(
    *,
    generation_index: int,
    lifecycle: RealizationLifecycleReport,
    final_feedback: ExactFeedbackAttachmentReport,
    action_reports: Sequence[ActiveRealizationSearchReport],
    families: ScientificProgramFamilyReport,
) -> IdeaEvolutionRequestReport:
    actions = prior_action_map(action_reports)
    feedback_by_h = {row.hypothesis_id: row for row in final_feedback.records}
    links_by_idea: dict[str, list[Any]] = defaultdict(list)
    for link in lifecycle.links:
        links_by_idea[link.idea_id].append(link)
    obs_by_realization = {row.realization_id: row for row in lifecycle.observations}
    family_by_idea = _family_context(families)
    requests: list[IdeaEvolutionRequest] = []

    for state in lifecycle.local_states:
        if state.usable_grounded_realization_count > 0:
            continue
        idea_id = state.idea_id
        links = sorted(links_by_idea.get(idea_id, []), key=lambda row: (row.attempt_index, row.realization_id))
        latest = links[-1] if links else None
        latest_obs = obs_by_realization.get(latest.realization_id) if latest is not None else None
        feedback_link = next(
            (
                row
                for row in reversed(links)
                if row.hypothesis_id is not None
                and str(row.hypothesis_id) in feedback_by_h
            ),
            None,
        )
        feedback = (
            feedback_by_h.get(str(feedback_link.hypothesis_id))
            if feedback_link is not None
            else None
        )
        attempted = actions.get(idea_id, [])
        scope: MutationScope | None = None
        reasons: list[str] = []
        hints: list[str] = []
        channels: list[str] = []

        if (
            latest_obs is not None
            and latest_obs.prospective_status == "COMPLETE"
            and latest_obs.prospective_contract_integrity_passed is True
            and latest_obs.prospective_identifiability == "NOT_OPERATIONALIZABLE"
            and any(action in attempted for action in ("EVIDENCE_REAXIS", "SAME_PREMISE_SHARPEN"))
        ):
            scope = "OPERATIONALIZATION_ESCAPE"
            reasons.append("SAME_IDEA_OPERATIONALIZATION_SEARCH_DID_NOT_RECOVER_USABLE_REALIZATION")
            hints = ["AXIS_MUTATION", "REGIME_BOUNDARY", "PROXY_CHALLENGE"]
            channels = ["TRANSFORM", "EXPLORE"]
        elif feedback is not None and feedback.residual_epistemic_state == "PRIOR_ART_BACKED_OR_NO_RESIDUAL":
            scope = "KNOWN_REGION_ESCAPE"
            reasons.append("FRESH_RESIDUAL_PLACES_CURRENT_REALIZATION_IN_KNOWN_REGION")
            hints = ["AXIS_MUTATION", "REGIME_BOUNDARY", "CANDIDATE_INTERPRETATION"]
            channels = ["TRANSFORM", "EXPLORE"]
        elif feedback is not None and feedback.residual_epistemic_state == "UNRESOLVED_TOPOLOGY_GAP":
            scope = "TOPOLOGY_ESCAPE"
            reasons.append("FRESH_RESIDUAL_TOPOLOGY_GAP_REMAINS_AFTER_LOCAL_SEARCH")
            hints = ["BACKBONE_MUTATION", "LATENT_VARIABLE", "AXIS_MUTATION"]
            channels = ["TRANSFORM", "EXPLORE"]
        elif feedback is not None and feedback.residual_epistemic_state == "UNRESOLVED_EVIDENCE_GAP":
            if "EVIDENCE_REAXIS" in attempted or "RETRIEVE_MORE" in attempted:
                scope = "EVIDENCE_GAP_ESCAPE"
                reasons.append("FRESH_RESIDUAL_EVIDENCE_GAP_REMAINS_AFTER_LOCAL_EVIDENCE_SEARCH")
                hints = ["AXIS_MUTATION", "CROSS_SOURCE_BRIDGE", "REGIME_BOUNDARY"]
                channels = ["TRANSFORM", "EXPLORE"]
        elif len(attempted) >= 2 and state.realization_count > 0:
            scope = "GENERAL_LOCAL_EXHAUSTION"
            reasons.append("BOUNDED_SAME_IDEA_SEARCH_EXHAUSTED_WITHOUT_USABLE_REALIZATION")
            hints = ["AXIS_MUTATION", "BACKBONE_MUTATION"]
            channels = ["TRANSFORM", "EXPLORE", "WILDCARD"]

        if scope is None:
            continue
        family_key, family_size = family_by_idea.get(idea_id, (None, None))
        snapshot = FailureObservationSnapshot(
            hypothesis_id=(
                str(latest.hypothesis_id)
                if latest is not None and latest.hypothesis_id is not None
                else str(feedback_link.hypothesis_id)
                if feedback_link is not None
                else None
            ),
            prospective_identifiability=(latest_obs.prospective_identifiability if latest_obs is not None else None),
            residual_epistemic_state=(feedback.residual_epistemic_state if feedback is not None else None),
            residual_state_reason=(feedback.residual_state_reason if feedback is not None else None),
            current_evidence_status=(feedback.current_evidence_status if feedback is not None else None),
        )
        requests.append(
            IdeaEvolutionRequest(
                request_id=_stable_id("idea_evolution_request", generation_index, idea_id, scope, attempted),
                source_idea_id=idea_id,
                source_generation_index=generation_index,
                source_realization_ids=[row.realization_id for row in links],
                exhausted_local_actions=attempted,
                typed_failure_observations=[snapshot],
                requested_mutation_scope=scope,
                recommended_channels=channels,
                nonbinding_operator_hints=hints,
                program_family_key=family_key,
                program_family_size=family_size,
                reason_codes=reasons,
            )
        )

    scope_counts = Counter(row.requested_mutation_scope for row in requests)
    gen_counts = Counter(f"G{row.source_generation_index}" for row in requests)
    provisional = IdeaEvolutionRequestReport(
        report_id="pending",
        report_sha256="pending",
        requests=requests,
        request_count=len(requests),
        scope_counts=dict(sorted(scope_counts.items())),
        source_generation_counts=dict(sorted(gen_counts.items())),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"idea_evolution_requests:{digest[:20]}",
            "report_sha256": digest,
        }
    )



def combine_active_reports(
    reports: Sequence[ActiveRealizationSearchReport],
    *,
    generation_index: int,
) -> ActiveRealizationSearchReport:
    if not reports:
        raise ValueError("at least one active report is required")
    if any(row.generation_index != generation_index for row in reports):
        raise ValueError("active report generation mismatch")
    action_records = [item for report in reports for item in report.action_records]
    new_links = [item for report in reports for item in report.new_links]
    new_observations = [item for report in reports for item in report.new_observations]
    rescued = sorted({item for report in reports for item in report.rescued_within_same_idea_ids})
    resolved = sorted({item for report in reports for item in report.resolved_usable_idea_ids})
    boundary = sorted({item for report in reports for item in report.boundary_escalation_requested_idea_ids})
    retrieval = sorted({item for report in reports for item in report.deferred_retrieval_idea_ids})
    graph = sorted({item for report in reports for item in report.deferred_graph_retraversal_idea_ids})
    targets = sorted({item for report in reports for item in report.target_idea_ids})
    action_counts = Counter(row.action for row in action_records)
    status_counts = Counter(row.execution_status for row in action_records)
    provisional = ActiveRealizationSearchReport(
        report_id="pending",
        report_sha256="pending",
        generation_index=generation_index,
        source_lifecycle_report_id=reports[0].source_lifecycle_report_id,
        source_offspring_execution_report_id=reports[0].source_offspring_execution_report_id,
        source_feedback_attachment_report_id=reports[0].source_feedback_attachment_report_id,
        target_idea_ids=targets,
        action_records=action_records,
        new_links=new_links,
        new_observations=new_observations,
        resolved_usable_idea_ids=resolved,
        rescued_within_same_idea_ids=rescued,
        unresolved_idea_ids=list(reports[-1].unresolved_idea_ids),
        boundary_escalation_requested_idea_ids=boundary,
        deferred_retrieval_idea_ids=retrieval,
        deferred_graph_retraversal_idea_ids=graph,
        action_counts=dict(sorted(action_counts.items())),
        execution_status_counts=dict(sorted(status_counts.items())),
        local_llm_call_count=sum(row.local_llm_call_count for row in reports),
        prospective_audit_llm_call_count=sum(row.prospective_audit_llm_call_count for row in reports),
        max_actions_per_idea=sum(row.max_actions_per_idea for row in reports),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"g{generation_index}_combined_active:{digest[:20]}",
            "report_sha256": digest,
        }
    )

def build_cost_audit(
    *,
    family_report: ScientificProgramFamilyReport,
    active_reports: Sequence[ActiveRealizationSearchReport],
    fresh_verification_reports: Sequence[FreshResidualVerificationReport],
    initial_credit: CreditContinuityReportV2,
    final_credit: CreditContinuityReportV2,
    generation3_execution: OffspringExecutionReport,
    residual_action_outcomes: Sequence[ResidualActionOutcome] = (),
) -> CostAwareClosedLoopAudit:
    local_calls = sum(row.local_llm_call_count for row in active_reports)
    prospective_calls = sum(row.prospective_audit_llm_call_count for row in active_reports)
    rescued = len({
        idea_id
        for report in active_reports
        for idea_id in report.rescued_within_same_idea_ids
    })
    child_rescue = final_credit.child_rescue_count
    g3_calls = generation3_execution.llm_call_count
    unresolved_before = initial_credit.unresolved_count
    unresolved_after = final_credit.unresolved_count
    return CostAwareClosedLoopAudit(
        local_llm_calls=local_calls,
        prospective_audit_llm_calls=prospective_calls,
        fresh_verification_barrier_runs=len(fresh_verification_reports),
        fresh_verification_subprocess_calls=sum(row.verification_subprocess_call_count for row in fresh_verification_reports),
        same_idea_rescue_count=rescued,
        same_idea_rescue_per_local_llm_call=(rescued / local_calls if local_calls else 0.0),
        existing_child_rescue_count=child_rescue,
        g3_offspring_llm_calls=g3_calls,
        existing_child_rescue_per_g3_offspring_llm_call=(child_rescue / g3_calls if g3_calls else 0.0),
        unresolved_before=unresolved_before,
        unresolved_after=unresolved_after,
        unresolved_reduction=unresolved_before - unresolved_after,
        program_family_count=family_report.program_family_count,
        family_fragmentation_ratio=family_report.fragmentation_ratio,
        residual_action_outcomes=list(residual_action_outcomes),
    )


def build_case_summary(
    *,
    idea_count: int,
    family_report: ScientificProgramFamilyReport,
    family_reference_absolute_error: int | None,
    generation2_rounds: Sequence[VerifiedLocalRoundRecord],
    generation3_rounds: Sequence[VerifiedLocalRoundRecord],
    initial_feedback_reports: Sequence[ExactFeedbackAttachmentReport],
    final_feedback_reports: Sequence[ExactFeedbackAttachmentReport],
    evolution_requests: IdeaEvolutionRequestReport,
    cost_audit: CostAwareClosedLoopAudit,
) -> VerifiedActiveSearchCaseSummary:
    g2_rescue = len({row.output_lifecycle_report_id for row in generation2_rounds if row.same_idea_rescue_count > 0})
    # Count actual rescued ideas, not merely rounds, from round records' totals.
    g2_rescue = sum(row.same_idea_rescue_count for row in generation2_rounds)
    g3_rescue = sum(row.same_idea_rescue_count for row in generation3_rounds)
    initial_residual = sum(row.residual_evaluated_count for row in initial_feedback_reports)
    final_residual = sum(row.residual_evaluated_count for row in final_feedback_reports)
    provisional = VerifiedActiveSearchCaseSummary(
        summary_id="pending",
        summary_sha256="pending",
        research_idea_count=idea_count,
        scientific_program_family_count=family_report.program_family_count,
        family_fragmentation_ratio=family_report.fragmentation_ratio,
        family_reference_absolute_error=family_reference_absolute_error,
        generation2_verified_round_count=len(generation2_rounds),
        generation3_verified_round_count=len(generation3_rounds),
        generation2_same_idea_rescue_count=g2_rescue,
        generation3_same_idea_rescue_count=g3_rescue,
        total_same_idea_rescue_count=g2_rescue + g3_rescue,
        initial_residual_evaluated_count=initial_residual,
        final_residual_evaluated_count=final_residual,
        idea_evolution_request_count=evolution_requests.request_count,
        idea_evolution_request_scope_counts=evolution_requests.scope_counts,
        cost_audit=cost_audit,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("summary_id", None)
    payload.pop("summary_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "summary_id": f"sis_v2_7_case:{digest[:20]}",
            "summary_sha256": digest,
        }
    )


__all__ = [
    "CostAwareClosedLoopAudit",
    "FreshResidualVerificationReport",
    "IdeaEvolutionRequest",
    "IdeaEvolutionRequestReport",
    "ResidualActionOutcome",
    "VerifiedActiveSearchCaseSummary",
    "VerifiedLocalRoundRecord",
    "attach_feedback_with_fresh_residual",
    "build_case_summary",
    "build_cost_audit",
    "combine_active_reports",
    "build_idea_evolution_requests",
    "merge_active_into_lifecycle",
    "normalize_lifecycle_strict",
    "prior_action_map",
]
