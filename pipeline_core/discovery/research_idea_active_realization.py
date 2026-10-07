from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.hypothesis_compiler import HypothesisCompileError, HypothesisCompiler
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
)
from pipeline_core.discovery.hypothesis_llm import HypothesisDraftBackend
from pipeline_core.discovery.hypothesis_prompt import HypothesisPrompt, HypothesisPromptAssembler
from pipeline_core.discovery.hypothesis_validation import HypothesisValidator
from pipeline_core.discovery.prospective_identification_materialization_shadow import (
    ProspectiveIdentificationShadowArtifact,
)
from pipeline_core.discovery.research_idea_closed_loop import RealizationLifecycleReport
from pipeline_core.discovery.research_idea_feedback_resolution import resolve_feedback_artifacts
from pipeline_core.discovery.research_idea_offspring_execution import OffspringExecutionReport
from pipeline_core.discovery.research_idea_population_semantics import (
    IdeaRealizationLink,
    ScientificFeedbackFacetObservation,
)



def _safe_premise_ids(context: HypothesisContext) -> list[str]:
    return sorted(
        str(row.statement_id)
        for row in context.evidence_statements
        if (
            row.eligible_as_premise
            and not row.requires_verification
            and not row.premise_restrictions
            and row.epistemic_role in {"reported", "evidence_synthesis"}
        )
    )


def _safe_unused_premise_ids(context: HypothesisContext, card: Any) -> list[str]:
    used = set(map(str, card.premise_statement_ids))
    return [statement_id for statement_id in _safe_premise_ids(context) if statement_id not in used]

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ActiveRealizationAction = Literal[
    "KEEP",
    "KEEP_UNTIL_EVALUATED",
    "SAME_PREMISE_SHARPEN",
    "EVIDENCE_REAXIS",
    "RETRIEVE_MORE",
    "AXIS_MUTATION",
    "REQUEST_GRAPH_RETRAVERSAL",
    "STOP",
]
ActionExecutionStatus = Literal[
    "NO_ACTION_NEEDED",
    "GENERATED",
    "ABSTAINED",
    "GENERATION_FAILED",
    "COMPILE_REJECTED",
    "VALIDATION_REJECTED",
    "CARDINALITY_REJECTED",
    "GROUNDING_DRIFT_REJECTED",
    "DUPLICATE_REALIZATION_SUPPRESSED",
    "DEFERRED_EXTERNAL_ACTION",
    "BOUNDARY_ESCALATION_REQUESTED",
    "LOCAL_SEARCH_EXHAUSTED",
]
FeedbackEvaluationState = Literal["EVALUATED", "NOT_EVALUATED", "AMBIGUOUS"]
BoundaryResolution = Literal[
    "CHILD_IDEA_OBSERVED",
    "SAME_IDEA_OBSERVED",
    "INDETERMINATE_CHILD_OBSERVED",
    "NO_NEXT_GENERATION_CHILD_OBSERVED",
    "NOT_REQUESTED",
]


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical(parts).encode('utf-8')).hexdigest()[:20]}"


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


class FeedbackAttachmentRecord(StrictModel):
    hypothesis_id: str = Field(min_length=1)
    idea_id: str = Field(min_length=1)
    prospective_evaluation_state: FeedbackEvaluationState
    residual_evaluation_state: FeedbackEvaluationState
    current_evidence_status: str | None = None
    prospective_identifiability: str | None = None
    directionality_mode: str | None = None
    measurement_compatibility_mode: str | None = None
    residual_epistemic_state: str | None = None
    residual_state_reason: str | None = None
    exact_lineage_only: Literal[True] = True
    not_evaluated_is_not_negative_feedback: Literal[True] = True


class ExactFeedbackAttachmentReport(StrictModel):
    schema_version: Literal[
        "research-idea-exact-feedback-attachment-shadow-v1"
    ] = "research-idea-exact-feedback-attachment-shadow-v1"
    report_id: str
    report_sha256: str
    source_lifecycle_report_id: str
    search_roots: list[str] = Field(default_factory=list)
    records: list[FeedbackAttachmentRecord] = Field(default_factory=list)
    prospective_evaluated_count: int = Field(ge=0)
    prospective_not_evaluated_count: int = Field(ge=0)
    prospective_ambiguous_count: int = Field(ge=0)
    residual_evaluated_count: int = Field(ge=0)
    residual_not_evaluated_count: int = Field(ge=0)
    residual_ambiguous_count: int = Field(ge=0)
    exact_lineage_only: Literal[True] = True
    not_evaluated_is_not_failure: Literal[True] = True
    production_selection_authority: Literal[False] = False


class ActiveRealizationActionRecord(StrictModel):
    action_id: str
    idea_id: str
    generation_index: int = Field(ge=2)
    source_realization_id: str | None = None
    source_hypothesis_id: str | None = None
    action: ActiveRealizationAction
    action_index: int = Field(ge=1)
    execution_status: ActionExecutionStatus
    generated_realization_id: str | None = None
    generated_hypothesis_id: str | None = None
    issue_codes: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    llm_call_count: int = Field(default=0, ge=0)
    requires_research_idea_semantic_comparison: bool = False
    local_action_cannot_change_idea_identity: bool = True
    external_action_executed: Literal[False] = False
    production_selection_authority: Literal[False] = False


class ActiveRealizationSearchReport(StrictModel):
    schema_version: Literal[
        "active-realization-search-shadow-v1"
    ] = "active-realization-search-shadow-v1"
    report_id: str
    report_sha256: str
    generation_index: int = Field(ge=2)
    source_lifecycle_report_id: str
    source_offspring_execution_report_id: str
    source_feedback_attachment_report_id: str
    target_idea_ids: list[str] = Field(default_factory=list)
    action_records: list[ActiveRealizationActionRecord] = Field(default_factory=list)
    new_links: list[IdeaRealizationLink] = Field(default_factory=list)
    new_observations: list[ScientificFeedbackFacetObservation] = Field(default_factory=list)
    resolved_usable_idea_ids: list[str] = Field(default_factory=list)
    rescued_within_same_idea_ids: list[str] = Field(default_factory=list)
    unresolved_idea_ids: list[str] = Field(default_factory=list)
    boundary_escalation_requested_idea_ids: list[str] = Field(default_factory=list)
    deferred_retrieval_idea_ids: list[str] = Field(default_factory=list)
    deferred_graph_retraversal_idea_ids: list[str] = Field(default_factory=list)
    action_counts: dict[str, int] = Field(default_factory=dict)
    execution_status_counts: dict[str, int] = Field(default_factory=dict)
    local_llm_call_count: int = Field(ge=0)
    prospective_audit_llm_call_count: int = Field(ge=0)
    max_actions_per_idea: int = Field(ge=1)
    local_search_executes_before_global_axis_mutation: Literal[True] = True
    not_evaluated_is_not_negative_feedback: Literal[True] = True
    axis_mutation_is_boundary_sensitive_not_local_identity_authority: Literal[True] = True
    no_g4_generation: Literal[True] = True
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class BoundaryEscalationTrace(StrictModel):
    trace_id: str
    source_idea_id: str
    source_generation_index: int = Field(ge=2)
    action: Literal["AXIS_MUTATION"] = "AXIS_MUTATION"
    observed_next_generation_child_idea_ids: list[str] = Field(default_factory=list)
    observed_identity_relations: list[str] = Field(default_factory=list)
    resolution: BoundaryResolution
    diagnostic_only: Literal[True] = True
    existing_next_generation_is_not_causally_attributed_to_v2_6_action: Literal[True] = True
    idea_identity_decided_by_operator_name: Literal[False] = False
    production_generation_authority: Literal[False] = False


class ActiveRealizationPortfolioResult(StrictModel):
    schema_version: Literal[
        "active-realization-portfolio-result-v1"
    ] = "active-realization-portfolio-result-v1"
    portfolio: HypothesisPortfolio
    report: ActiveRealizationSearchReport


class ProspectiveRunner(Protocol):
    def __call__(
        self,
        *,
        context: HypothesisContext,
        candidate: Any,
        source_stage: str,
        output_prefix: Path,
    ) -> ProspectiveIdentificationShadowArtifact: ...


class _CompiledAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    portfolio: HypothesisPortfolio | None = None
    status: str
    issue_codes: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


def _compile_single(
    *,
    context: HypothesisContext,
    draft: HypothesisPortfolioDraft,
) -> _CompiledAttempt:
    if not draft.hypotheses:
        return _CompiledAttempt(
            status="ABSTAINED",
            issues=[str(draft.abstention_reason or "model abstained")],
        )
    if len(draft.hypotheses) != 1:
        return _CompiledAttempt(
            status="CARDINALITY_REJECTED",
            issue_codes=["EXPECTED_EXACTLY_ONE_HYPOTHESIS"],
            issues=["Expected exactly one hypothesis realization."],
        )
    try:
        portfolio = HypothesisCompiler().compile(context, draft)
    except HypothesisCompileError as exc:
        return _CompiledAttempt(
            status="COMPILE_REJECTED",
            issue_codes=[row.code for row in exc.issues],
            issues=[f"{row.code}:{row.location}:{row.message}" for row in exc.issues],
        )
    except Exception as exc:
        return _CompiledAttempt(
            status="COMPILE_REJECTED",
            issue_codes=[type(exc).__name__],
            issues=[str(exc)],
        )
    validation = HypothesisValidator().validate(context, portfolio)
    if not validation.passes:
        issues = [
            f"{row.code}:{row.location}:{row.message}"
            for row in validation.issues
            if row.severity == "error"
        ]
        return _CompiledAttempt(
            status="VALIDATION_REJECTED",
            issue_codes=_dedupe([row.split(":", 1)[0] for row in issues]),
            issues=issues,
        )
    return _CompiledAttempt(portfolio=portfolio, status="MATERIALIZED")


def _observation_usable(row: ScientificFeedbackFacetObservation) -> bool:
    return bool(
        row.materialization_status == "MATERIALIZED"
        and row.prospective_status == "COMPLETE"
        and row.prospective_contract_integrity_passed is True
        and row.prospective_identifiability != "NOT_OPERATIONALIZABLE"
    )


def _observation_from_attempt(
    *,
    link: IdeaRealizationLink,
    prospective: ProspectiveIdentificationShadowArtifact | None,
) -> ScientificFeedbackFacetObservation:
    systems = ["SIS_V2_6_ACTIVE_REALIZATION_SEARCH"]
    versions = [link.schema_version]
    refs = list(link.source_report_ids)
    if prospective is not None:
        systems.append("PROSPECTIVE_IDENTIFICATION_MATERIALIZATION_SHADOW")
        versions.append(prospective.schema_version)
        if prospective.artifact_path:
            refs.append(str(prospective.artifact_path))
    return ScientificFeedbackFacetObservation(
        observation_id=_stable_id(
            "v2_6_feedback_observation",
            link.realization_id,
            prospective.model_dump(mode="json") if prospective is not None else None,
        ),
        idea_id=link.idea_id,
        realization_id=link.realization_id,
        hypothesis_id=link.hypothesis_id,
        materialization_status=link.materialization_status,
        grounding_integrity=(
            "PASSED"
            if link.materialization_status == "MATERIALIZED"
            else "FAILED"
            if link.materialization_status in {
                "COMPILE_REJECTED", "VALIDATION_REJECTED", "CARDINALITY_REJECTED"
            }
            else "NOT_ASSESSED"
        ),
        materialization_issue_codes=list(link.issue_codes),
        prospective_status=(prospective.status if prospective is not None else None),
        current_evidence_status=(prospective.current_evidence_status if prospective is not None else None),
        prospective_identifiability=(
            prospective.prospective_identifiability if prospective is not None else None
        ),
        directionality_mode=(prospective.directionality_mode if prospective is not None else None),
        measurement_compatibility_mode=(
            prospective.measurement_compatibility_mode if prospective is not None else None
        ),
        prospective_contract_integrity_passed=(
            prospective.contract_integrity_passed if prospective is not None else None
        ),
        source_systems=systems,
        source_versions=versions,
        source_artifact_refs=_dedupe(refs),
    )


def _lifecycle_materialization_adapter(lifecycle: RealizationLifecycleReport) -> Any:
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


def attach_exact_feedback(
    *,
    lifecycle: RealizationLifecycleReport,
    search_roots: Sequence[Path],
) -> ExactFeedbackAttachmentReport:
    resolved = resolve_feedback_artifacts(
        materialization=_lifecycle_materialization_adapter(lifecycle),
        search_roots=search_roots,
    )
    idea_by_hypothesis = {
        str(link.hypothesis_id): link.idea_id
        for link in lifecycle.links
        if link.hypothesis_id is not None
        and link.materialization_status == "MATERIALIZED"
    }
    residual_by_hypothesis: dict[str, Mapping[str, Any]] = {}
    if resolved.residual_state is not None:
        residual_by_hypothesis = {
            str(row.get("hypothesis_id")): row
            for row in resolved.residual_state.get("hypotheses", [])
            if isinstance(row, Mapping) and row.get("hypothesis_id")
        }

    records: list[FeedbackAttachmentRecord] = []
    ambiguous_prospective = set(resolved.report.prospective_ambiguous_hypothesis_ids)
    for hypothesis_id, idea_id in sorted(idea_by_hypothesis.items()):
        prospective = resolved.prospective_by_hypothesis.get(hypothesis_id)
        if hypothesis_id in ambiguous_prospective:
            p_state: FeedbackEvaluationState = "AMBIGUOUS"
        elif prospective is not None:
            p_state = "EVALUATED"
        else:
            p_state = "NOT_EVALUATED"

        residual_row = residual_by_hypothesis.get(hypothesis_id)
        if resolved.report.residual_ambiguous:
            r_state: FeedbackEvaluationState = "AMBIGUOUS"
        elif resolved.residual_state is not None:
            r_state = "EVALUATED"
        else:
            r_state = "NOT_EVALUATED"

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
        search_roots=[str(path.expanduser().resolve()) for path in search_roots],
        records=records,
        prospective_evaluated_count=counts_p["EVALUATED"],
        prospective_not_evaluated_count=counts_p["NOT_EVALUATED"],
        prospective_ambiguous_count=counts_p["AMBIGUOUS"],
        residual_evaluated_count=counts_r["EVALUATED"],
        residual_not_evaluated_count=counts_r["NOT_EVALUATED"],
        residual_ambiguous_count=counts_r["AMBIGUOUS"],
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"exact_feedback_attachment:{digest[:20]}",
            "report_sha256": digest,
        }
    )


def _latest_link_and_observation(
    *,
    idea_id: str,
    links: Sequence[IdeaRealizationLink],
    observations: Sequence[ScientificFeedbackFacetObservation],
) -> tuple[IdeaRealizationLink | None, ScientificFeedbackFacetObservation | None]:
    idea_links = [row for row in links if row.idea_id == idea_id]
    if not idea_links:
        return None, None
    idea_links.sort(key=lambda row: (row.attempt_index, row.realization_id))
    link = idea_links[-1]
    by_realization = {row.realization_id: row for row in observations}
    return link, by_realization.get(link.realization_id)


def choose_active_action(
    *,
    context: HypothesisContext,
    current_card: Any | None,
    current_observation: ScientificFeedbackFacetObservation | None,
    feedback: FeedbackAttachmentRecord | None,
    prior_actions: Sequence[ActiveRealizationAction],
) -> tuple[ActiveRealizationAction, list[str]]:
    tried = set(prior_actions)
    if current_observation is not None and current_observation.materialization_status == "MATERIALIZED":
        attached_prospective_state = (
            feedback.prospective_evaluation_state if feedback is not None else "NOT_EVALUATED"
        )
        if current_observation.prospective_identifiability is None and attached_prospective_state == "NOT_EVALUATED":
            return "KEEP_UNTIL_EVALUATED", ["PROSPECTIVE_NOT_EVALUATED_IS_NOT_FAILURE"]

    if current_observation is not None and _observation_usable(current_observation):
        return "KEEP", ["CURRENT_REALIZATION_USABLE"]

    if current_card is not None:
        safe_unused = _safe_unused_premise_ids(context, current_card)
    else:
        safe_unused = _safe_premise_ids(context)

    not_operationalizable = bool(
        current_observation is not None
        and current_observation.prospective_status == "COMPLETE"
        and current_observation.prospective_contract_integrity_passed is True
        and current_observation.prospective_identifiability == "NOT_OPERATIONALIZABLE"
    )
    if not_operationalizable:
        if safe_unused and "EVIDENCE_REAXIS" not in tried:
            return "EVIDENCE_REAXIS", ["NOT_OPERATIONALIZABLE_TRY_NEW_SAFE_GROUNDED_AXIS"]
        if current_card is not None and "SAME_PREMISE_SHARPEN" not in tried:
            return "SAME_PREMISE_SHARPEN", ["NOT_OPERATIONALIZABLE_TRY_SAME_PREMISE_REFORMULATION"]
        return "AXIS_MUTATION", ["NOT_OPERATIONALIZABLE_LOCAL_SEARCH_EXHAUSTED"]

    residual_state = feedback.residual_epistemic_state if feedback is not None else None
    if residual_state == "UNRESOLVED_EVIDENCE_GAP":
        if safe_unused and "EVIDENCE_REAXIS" not in tried:
            return "EVIDENCE_REAXIS", ["RESIDUAL_EVIDENCE_GAP_TRY_SAFE_REAXIS"]
        if "RETRIEVE_MORE" not in tried:
            return "RETRIEVE_MORE", ["RESIDUAL_EVIDENCE_GAP_REQUEST_MORE_EVIDENCE"]
        return "AXIS_MUTATION", ["RESIDUAL_EVIDENCE_GAP_LOCAL_OPTIONS_EXHAUSTED"]
    if residual_state == "UNRESOLVED_TOPOLOGY_GAP":
        if current_card is not None and "SAME_PREMISE_SHARPEN" not in tried:
            return "SAME_PREMISE_SHARPEN", ["RESIDUAL_TOPOLOGY_GAP_TRY_LOCAL_SHARPEN"]
        return "REQUEST_GRAPH_RETRAVERSAL", ["RESIDUAL_TOPOLOGY_GAP_REQUEST_CONTEXT_RESET"]
    if residual_state == "PRIOR_ART_BACKED_OR_NO_RESIDUAL":
        return "AXIS_MUTATION", ["KNOWN_REGION_RAISES_TRANSFORMATION_PRESSURE"]

    if current_observation is None or current_observation.materialization_status != "MATERIALIZED":
        if safe_unused and "EVIDENCE_REAXIS" not in tried:
            return "EVIDENCE_REAXIS", ["FAILED_REALIZATION_TRY_GROUNDED_REAXIS"]
        if current_card is not None and "SAME_PREMISE_SHARPEN" not in tried:
            return "SAME_PREMISE_SHARPEN", ["FAILED_REALIZATION_TRY_LOCAL_REPAIR"]
        return "AXIS_MUTATION", ["FAILED_REALIZATION_LOCAL_SEARCH_EXHAUSTED"]

    return "KEEP_UNTIL_EVALUATED", ["NO_AUTHORITATIVE_NEGATIVE_LOCAL_SIGNAL"]


def _action_prompt(
    *,
    action: Literal["SAME_PREMISE_SHARPEN", "EVIDENCE_REAXIS"],
    context: HypothesisContext,
    node: Any,
    current_card: Any | None,
    action_index: int,
) -> HypothesisPrompt:
    base = HypothesisPromptAssembler(max_hypotheses=1).build(context)
    safe_ids = _safe_premise_ids(context)
    current_ids = list(current_card.premise_statement_ids) if current_card is not None else []
    unused = [value for value in safe_ids if value not in set(current_ids)]

    if action == "SAME_PREMISE_SHARPEN":
        system_extra = """
SIS-v2.6 ACTIVE LOCAL SEARCH — SAME PREMISE SHARPEN
===================================================
Generate exactly ONE alternate grounded realization of the SAME ResearchIdea.
Preserve the current hypothesis positive premise IDs EXACTLY and preserve the ResearchIdea kernel.
You may sharpen the inferential bridge, observable framing, qualitative prediction, or falsifier.
Do not change the mechanism/causal/relation commitments merely to make grounding easier.
Return one corrected realization or abstain.
"""
        requirement = {
            "required_exact_premise_ids": current_ids,
            "new_premise_required": False,
        }
    else:
        system_extra = """
SIS-v2.6 ACTIVE LOCAL SEARCH — EVIDENCE RE-AXIS
===============================================
Generate exactly ONE alternate grounded realization of the SAME ResearchIdea using the supplied grounded context.
Preserve the ResearchIdea kernel. Change the grounded evidence axis/formulation, not the abstract scientific program.
Use only SAFE GROUNDED PREMISE IDS listed below and include at least one NEW SAFE premise not used by the current realization.
Do not invent evidence, citations, or novelty claims. Return one realization or abstain.
"""
        requirement = {
            "safe_grounded_premise_ids": safe_ids,
            "current_premise_ids": current_ids,
            "required_new_premise_ids_one_of": unused,
            "new_premise_required": True,
        }
    payload = {
        "action": action,
        "action_index": action_index,
        "research_idea_id": node.idea_id,
        "research_idea_kernel_must_be_preserved": node.kernel.model_dump(mode="json"),
        "current_hypothesis": (
            current_card.model_dump(mode="json") if current_card is not None else None
        ),
        "grounding_requirements": requirement,
    }
    user = (
        base.user_prompt
        + "\n\nSIS-v2.6 ACTIVE LOCAL REALIZATION SEARCH\n"
        + "==========================================\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    )
    return HypothesisPrompt.create(
        system_prompt=base.system_prompt + system_extra,
        user_prompt=user,
    )


def _validate_action_premises(
    *,
    action: str,
    context: HypothesisContext,
    current_card: Any | None,
    candidate: Any,
) -> list[str]:
    candidate_ids = list(map(str, candidate.premise_statement_ids))
    current_ids = list(map(str, current_card.premise_statement_ids)) if current_card is not None else []
    if action == "SAME_PREMISE_SHARPEN":
        if candidate_ids != current_ids:
            return ["SAME_PREMISE_SHARPEN_CHANGED_PREMISE_IDENTITY"]
        return []
    safe = set(_safe_premise_ids(context))
    if not set(candidate_ids).issubset(safe):
        return ["EVIDENCE_REAXIS_USED_NONSAFE_POSITIVE_PREMISE"]
    unused = safe - set(current_ids)
    if unused and not (set(candidate_ids) & unused):
        return ["EVIDENCE_REAXIS_DID_NOT_USE_NEW_SAFE_PREMISE"]
    if not unused:
        return ["EVIDENCE_REAXIS_NO_UNUSED_SAFE_PREMISE_AVAILABLE"]
    return []


def run_active_realization_search(
    *,
    execution: OffspringExecutionReport,
    lifecycle: RealizationLifecycleReport,
    portfolio: HypothesisPortfolio,
    context: HypothesisContext,
    feedback: ExactFeedbackAttachmentReport,
    backend: HypothesisDraftBackend,
    output_dir: Path,
    prospective_runner: ProspectiveRunner | None = None,
    max_active_ideas: int = 8,
    max_actions_per_idea: int = 2,
    max_repair_attempts: int = 1,
    max_prospective_audits: int = 12,
    prior_actions_by_idea: Mapping[str, Sequence[ActiveRealizationAction]] | None = None,
) -> ActiveRealizationPortfolioResult:
    if max_active_ideas < 1 or max_actions_per_idea < 1:
        raise ValueError("active-search bounds must be >= 1")
    node_by_id = {row.idea_id: row for row in execution.offspring_nodes}
    cards_by_id = {str(card.hypothesis_id): card for card in portfolio.hypotheses}
    feedback_by_hypothesis = {row.hypothesis_id: row for row in feedback.records}
    old_links = list(lifecycle.links)
    old_observations = list(lifecycle.observations)
    links = list(old_links)
    observations = list(old_observations)
    cards = list(portfolio.hypotheses)
    seen_hypothesis_ids = {str(card.hypothesis_id) for card in cards}

    state_by_idea = {row.idea_id: row for row in lifecycle.local_states}
    target_ids = [
        idea_id
        for idea_id in lifecycle.target_idea_ids
        if idea_id in node_by_id
        and state_by_idea.get(idea_id) is not None
        and state_by_idea[idea_id].usable_grounded_realization_count == 0
    ][:max_active_ideas]

    action_records: list[ActiveRealizationActionRecord] = []
    rescued: list[str] = []
    resolved: list[str] = sorted({
        row.idea_id for row in old_observations if _observation_usable(row)
    })
    boundary: list[str] = []
    deferred_retrieval: list[str] = []
    deferred_graph: list[str] = []
    prospective_calls = 0

    for idea_id in target_ids:
        node = node_by_id[idea_id]
        prior_actions: list[ActiveRealizationAction] = list(
            (prior_actions_by_idea or {}).get(idea_id, ())
        )
        had_usable_before = any(
            row.idea_id == idea_id and _observation_usable(row)
            for row in old_observations
        )

        for action_index in range(1, max_actions_per_idea + 1):
            current_link, current_obs = _latest_link_and_observation(
                idea_id=idea_id,
                links=links,
                observations=observations,
            )
            current_card = (
                cards_by_id.get(str(current_link.hypothesis_id))
                if current_link is not None and current_link.hypothesis_id is not None
                else None
            )
            feedback_row = (
                feedback_by_hypothesis.get(str(current_link.hypothesis_id))
                if current_link is not None and current_link.hypothesis_id is not None
                else None
            )
            action, reasons = choose_active_action(
                context=context,
                current_card=current_card,
                current_observation=current_obs,
                feedback=feedback_row,
                prior_actions=prior_actions,
            )
            prior_actions.append(action)
            action_id = _stable_id(
                "active_realization_action",
                lifecycle.report_id,
                idea_id,
                action_index,
                action,
            )

            if action in {"KEEP", "KEEP_UNTIL_EVALUATED"}:
                action_records.append(
                    ActiveRealizationActionRecord(
                        action_id=action_id,
                        idea_id=idea_id,
                        generation_index=execution.generation_index,
                        source_realization_id=(current_link.realization_id if current_link else None),
                        source_hypothesis_id=(current_link.hypothesis_id if current_link else None),
                        action=action,
                        action_index=action_index,
                        execution_status="NO_ACTION_NEEDED",
                        reason_codes=reasons,
                    )
                )
                if current_obs is not None and _observation_usable(current_obs):
                    resolved.append(idea_id)
                break

            if action == "RETRIEVE_MORE":
                deferred_retrieval.append(idea_id)
                action_records.append(
                    ActiveRealizationActionRecord(
                        action_id=action_id,
                        idea_id=idea_id,
                        generation_index=execution.generation_index,
                        source_realization_id=(current_link.realization_id if current_link else None),
                        source_hypothesis_id=(current_link.hypothesis_id if current_link else None),
                        action=action,
                        action_index=action_index,
                        execution_status="DEFERRED_EXTERNAL_ACTION",
                        reason_codes=[*reasons, "RETRIEVAL_NOT_EXECUTED_BY_SIS_V2_6"],
                    )
                )
                break

            if action == "REQUEST_GRAPH_RETRAVERSAL":
                deferred_graph.append(idea_id)
                action_records.append(
                    ActiveRealizationActionRecord(
                        action_id=action_id,
                        idea_id=idea_id,
                        generation_index=execution.generation_index,
                        source_realization_id=(current_link.realization_id if current_link else None),
                        source_hypothesis_id=(current_link.hypothesis_id if current_link else None),
                        action=action,
                        action_index=action_index,
                        execution_status="DEFERRED_EXTERNAL_ACTION",
                        reason_codes=[*reasons, "GRAPH_RETRAVERSAL_NOT_EXECUTED_BY_SIS_V2_6"],
                    )
                )
                break

            if action == "AXIS_MUTATION":
                boundary.append(idea_id)
                action_records.append(
                    ActiveRealizationActionRecord(
                        action_id=action_id,
                        idea_id=idea_id,
                        generation_index=execution.generation_index,
                        source_realization_id=(current_link.realization_id if current_link else None),
                        source_hypothesis_id=(current_link.hypothesis_id if current_link else None),
                        action=action,
                        action_index=action_index,
                        execution_status="BOUNDARY_ESCALATION_REQUESTED",
                        reason_codes=reasons,
                        requires_research_idea_semantic_comparison=True,
                        local_action_cannot_change_idea_identity=False,
                    )
                )
                break

            if action == "STOP":
                action_records.append(
                    ActiveRealizationActionRecord(
                        action_id=action_id,
                        idea_id=idea_id,
                        generation_index=execution.generation_index,
                        source_realization_id=(current_link.realization_id if current_link else None),
                        source_hypothesis_id=(current_link.hypothesis_id if current_link else None),
                        action=action,
                        action_index=action_index,
                        execution_status="LOCAL_SEARCH_EXHAUSTED",
                        reason_codes=reasons,
                    )
                )
                break

            assert action in {"SAME_PREMISE_SHARPEN", "EVIDENCE_REAXIS"}
            prompt = _action_prompt(
                action=action,
                context=context,
                node=node,
                current_card=current_card,
                action_index=action_index,
            )
            calls = 0
            repairs = 0
            compiled = _CompiledAttempt(status="GENERATION_FAILED")
            draft: HypothesisPortfolioDraft | None = None
            try:
                generation = backend.generate(prompt)
                calls += 1
                draft = generation.draft
                compiled = _compile_single(context=context, draft=draft)
            except Exception as exc:
                compiled = _CompiledAttempt(
                    status="GENERATION_FAILED",
                    issue_codes=[type(exc).__name__],
                    issues=[str(exc)],
                )

            while (
                draft is not None
                and compiled.status in {"COMPILE_REJECTED", "VALIDATION_REJECTED", "CARDINALITY_REJECTED"}
                and repairs < max_repair_attempts
            ):
                repairs += 1
                feedback_text = "\n".join(
                    [
                        f"SIS-v2.6 {action} repair.",
                        "Remain under the SAME ResearchIdea kernel.",
                        "Use only grounded eligible premise IDs.",
                        "Issues:",
                        *[f"- {value}" for value in compiled.issues],
                        "Return one corrected realization or abstain.",
                    ]
                )
                try:
                    repaired = backend.repair(prompt, draft, feedback_text)
                    calls += 1
                    draft = repaired.draft
                    compiled = _compile_single(context=context, draft=draft)
                except Exception as exc:
                    compiled = _CompiledAttempt(
                        status="GENERATION_FAILED",
                        issue_codes=[type(exc).__name__],
                        issues=[str(exc)],
                    )
                    break

            card = None
            hypothesis_id = None
            issue_codes = list(compiled.issue_codes)
            status = compiled.status
            if compiled.portfolio is not None and compiled.portfolio.hypotheses:
                candidate = compiled.portfolio.hypotheses[0]
                premise_issues = _validate_action_premises(
                    action=action,
                    context=context,
                    current_card=current_card,
                    candidate=candidate,
                )
                if premise_issues:
                    status = "GROUNDING_DRIFT_REJECTED"
                    issue_codes = _dedupe([*issue_codes, *premise_issues])
                else:
                    hypothesis_id = str(candidate.hypothesis_id)
                    if hypothesis_id in seen_hypothesis_ids:
                        status = "DUPLICATE_REALIZATION_SUPPRESSED"
                        issue_codes = _dedupe([*issue_codes, "DUPLICATE_HYPOTHESIS_ID"])
                        hypothesis_id = None
                    else:
                        card = candidate
                        seen_hypothesis_ids.add(hypothesis_id)
                        cards.append(candidate)
                        cards_by_id[hypothesis_id] = candidate

            realization_id = _stable_id(
                "v2_6_realization",
                lifecycle.report_id,
                idea_id,
                action_index,
                action,
                hypothesis_id,
                status,
            )
            link = IdeaRealizationLink(
                realization_id=realization_id,
                idea_id=idea_id,
                generation_index=execution.generation_index,
                hypothesis_id=hypothesis_id,
                source_context_id=context.context_id,
                realization_kind=("REPAIR" if action == "SAME_PREMISE_SHARPEN" else "EVIDENCE_REAXIS"),
                attempt_index=max(
                    [row.attempt_index for row in links if row.idea_id == idea_id] or [0]
                ) + 1,
                parent_realization_id=(current_link.realization_id if current_link else None),
                materialization_status=status,
                issue_codes=issue_codes,
                repair_attempt_count=repairs,
                llm_call_count=calls,
                source_report_ids=[lifecycle.report_id],
            )
            links.append(link)
            prospective = None
            if (
                card is not None
                and prospective_runner is not None
                and prospective_calls < max_prospective_audits
            ):
                prefix = output_dir / "prospective" / realization_id.replace(":", "_")
                prospective = prospective_runner(
                    context=context,
                    candidate=card,
                    source_stage=f"SIS_V2_6_G{execution.generation_index}_{action}",
                    output_prefix=prefix,
                )
                prospective_calls += 1
            observation = _observation_from_attempt(link=link, prospective=prospective)
            observations.append(observation)
            action_records.append(
                ActiveRealizationActionRecord(
                    action_id=action_id,
                    idea_id=idea_id,
                    generation_index=execution.generation_index,
                    source_realization_id=(current_link.realization_id if current_link else None),
                    source_hypothesis_id=(current_link.hypothesis_id if current_link else None),
                    action=action,
                    action_index=action_index,
                    execution_status=(status if status in {
                        "ABSTAINED", "GENERATION_FAILED", "COMPILE_REJECTED",
                        "VALIDATION_REJECTED", "CARDINALITY_REJECTED",
                        "GROUNDING_DRIFT_REJECTED", "DUPLICATE_REALIZATION_SUPPRESSED"
                    } else "GENERATED"),
                    generated_realization_id=realization_id,
                    generated_hypothesis_id=hypothesis_id,
                    issue_codes=issue_codes,
                    reason_codes=reasons,
                    llm_call_count=calls,
                )
            )
            if _observation_usable(observation):
                resolved.append(idea_id)
                if not had_usable_before:
                    rescued.append(idea_id)
                break

    unresolved = [
        idea_id
        for idea_id in lifecycle.target_idea_ids
        if not any(row.idea_id == idea_id and _observation_usable(row) for row in observations)
    ]
    action_counts = Counter(row.action for row in action_records)
    status_counts = Counter(row.execution_status for row in action_records)
    provisional = ActiveRealizationSearchReport(
        report_id="pending",
        report_sha256="pending",
        generation_index=execution.generation_index,
        source_lifecycle_report_id=lifecycle.report_id,
        source_offspring_execution_report_id=execution.report_id,
        source_feedback_attachment_report_id=feedback.report_id,
        target_idea_ids=target_ids,
        action_records=action_records,
        new_links=[row for row in links if row.realization_id not in {x.realization_id for x in old_links}],
        new_observations=[
            row for row in observations if row.observation_id not in {x.observation_id for x in old_observations}
        ],
        resolved_usable_idea_ids=sorted(set(resolved)),
        rescued_within_same_idea_ids=sorted(set(rescued)),
        unresolved_idea_ids=sorted(set(unresolved)),
        boundary_escalation_requested_idea_ids=sorted(set(boundary)),
        deferred_retrieval_idea_ids=sorted(set(deferred_retrieval)),
        deferred_graph_retraversal_idea_ids=sorted(set(deferred_graph)),
        action_counts=dict(sorted(action_counts.items())),
        execution_status_counts=dict(sorted(status_counts.items())),
        local_llm_call_count=sum(row.llm_call_count for row in action_records),
        prospective_audit_llm_call_count=prospective_calls,
        max_actions_per_idea=max_actions_per_idea,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    report = provisional.model_copy(
        update={
            "report_id": f"g{execution.generation_index}_active_realization:{digest[:20]}",
            "report_sha256": digest,
        }
    )
    combined = portfolio.model_copy(
        update={
            "portfolio_id": _stable_id(
                f"g{execution.generation_index}_v2_6_active_portfolio",
                portfolio.portfolio_id,
                [str(card.hypothesis_id) for card in cards],
            ),
            "hypotheses": cards,
            "abstention_reason": None if cards else portfolio.abstention_reason,
        }
    )
    return ActiveRealizationPortfolioResult(portfolio=combined, report=report)


def build_boundary_escalation_traces(
    *,
    generation2_active: ActiveRealizationSearchReport,
    generation3_active: ActiveRealizationSearchReport,
    generation3_execution: OffspringExecutionReport,
) -> list[BoundaryEscalationTrace]:
    semantic_by_child = {row.idea_id: row for row in generation3_execution.semantic_records}
    children_by_parent: dict[str, list[Any]] = {}
    for row in generation3_execution.semantic_records:
        for parent_id in row.parent_idea_ids:
            children_by_parent.setdefault(parent_id, []).append(row)

    traces: list[BoundaryEscalationTrace] = []
    for active in (generation2_active, generation3_active):
        for record in active.action_records:
            if record.action != "AXIS_MUTATION":
                continue
            children = (
                children_by_parent.get(record.idea_id, [])
                if active.generation_index == 2
                else []
            )
            identities = [row.transition.identity_relation for row in children]
            if any(value == "DIFFERENT_IDEA" for value in identities):
                resolution: BoundaryResolution = "CHILD_IDEA_OBSERVED"
            elif any(value == "SAME_IDEA" for value in identities):
                resolution = "SAME_IDEA_OBSERVED"
            elif children:
                resolution = "INDETERMINATE_CHILD_OBSERVED"
            else:
                resolution = "NO_NEXT_GENERATION_CHILD_OBSERVED"
            traces.append(
                BoundaryEscalationTrace(
                    trace_id=_stable_id(
                        "boundary_escalation",
                        active.report_id,
                        record.idea_id,
                        record.action_index,
                    ),
                    source_idea_id=record.idea_id,
                    source_generation_index=active.generation_index,
                    observed_next_generation_child_idea_ids=[row.idea_id for row in children],
                    observed_identity_relations=identities,
                    resolution=resolution,
                )
            )
    return traces


__all__ = [
    "ActiveRealizationActionRecord",
    "ActiveRealizationPortfolioResult",
    "ActiveRealizationSearchReport",
    "BoundaryEscalationTrace",
    "ExactFeedbackAttachmentReport",
    "FeedbackAttachmentRecord",
    "attach_exact_feedback",
    "build_boundary_escalation_traces",
    "choose_active_action",
    "run_active_realization_search",
]
