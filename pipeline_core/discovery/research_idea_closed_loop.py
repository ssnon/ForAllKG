from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.hypothesis_compiler import HypothesisCompileError, HypothesisCompiler
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
)
from pipeline_core.discovery.hypothesis_llm import HypothesisDraftBackend
from pipeline_core.discovery.hypothesis_prompt import HypothesisPrompt
from pipeline_core.discovery.hypothesis_validation import HypothesisValidator
from pipeline_core.discovery.prospective_identification_materialization_shadow import (
    ProspectiveIdentificationShadowArtifact,
)
from pipeline_core.discovery.research_idea_contracts import ResearchIdeaNode
from pipeline_core.discovery.research_idea_multigeneration import ConceptualDeltaAuditReport
from pipeline_core.discovery.research_idea_offspring_execution import (
    OffspringExecutionReport,
    OffspringRealizationReport,
    build_realization_prompt,
    select_offspring_for_realization,
)
from pipeline_core.discovery.research_idea_population_semantics import (
    AdaptiveLocalSearchEvent,
    CreditContinuityTrace,
    FamilyPopulationReport,
    IdeaLearningDecision,
    IdeaLocalSearchState,
    IdeaRealizationLink,
    ScientificFeedbackFacetObservation,
    SemanticCalibrationSample,
    VerificationFacetIntegrationReport,
    adapt_adaptive_controller_plans,
    assign_conceptual_families,
    build_credit_continuity_traces,
    build_family_population_report,
    build_idea_learning_decisions,
    build_local_search_states,
    build_semantic_calibration_sample,
    enrich_observations_from_residual_reports,
)
from pipeline_core.discovery.research_idea_search_contracts import GenerationalIdeaSearchShadowReport


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


class RealizationLifecycleReport(StrictModel):
    schema_version: Literal[
        "research-idea-realization-lifecycle-shadow-v1"
    ] = "research-idea-realization-lifecycle-shadow-v1"
    report_id: str
    report_sha256: str
    generation_index: int = Field(ge=2)
    source_offspring_execution_report_id: str
    source_context_id: str
    source_context_sha256: str
    target_idea_ids: list[str] = Field(default_factory=list)
    links: list[IdeaRealizationLink] = Field(default_factory=list)
    observations: list[ScientificFeedbackFacetObservation] = Field(default_factory=list)
    local_states: list[IdeaLocalSearchState] = Field(default_factory=list)
    output_portfolio_id: str
    materialized_hypothesis_count: int = Field(ge=0)
    usable_grounded_realization_count: int = Field(ge=0)
    rescued_within_same_idea_count: int = Field(ge=0)
    local_search_exhausted_count: int = Field(ge=0)
    prospective_audit_count: int = Field(ge=0)
    prospective_not_operationalizable_count: int = Field(ge=0)
    realization_llm_call_count: int = Field(ge=0)
    prospective_audit_llm_call_count: int = Field(ge=0)
    max_realizations_per_idea: int = Field(ge=1)
    max_repair_attempts: int = Field(ge=0)
    same_idea_local_search_precedes_idea_escalation: Literal[True] = True
    not_operationalizable_realization_is_not_idea_failure: Literal[True] = True
    grounded_premise_boundary_enforced: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "RealizationLifecycleReport":
        if self.materialized_hypothesis_count != sum(
            row.materialization_status == "MATERIALIZED" for row in self.links
        ):
            raise ValueError("materialized_hypothesis_count mismatch")
        return self


class PopulationClosedLoopPolicyReport(StrictModel):
    schema_version: Literal[
        "research-idea-population-closed-loop-policy-shadow-v1"
    ] = "research-idea-population-closed-loop-policy-shadow-v1"
    report_id: str
    report_sha256: str
    decisions: list[IdeaLearningDecision] = Field(default_factory=list)
    decision_counts: dict[str, int] = Field(default_factory=dict)
    local_search_exhausted_count: int = Field(ge=0)
    transform_escalation_count: int = Field(ge=0)
    family_overconcentration_pressure_count: int = Field(ge=0)
    observation_policy_separated: Literal[True] = True
    single_scalar_fitness_used: Literal[False] = False
    family_is_soft_pressure_not_gate: Literal[True] = True
    verification_is_not_idea_truth_authority: Literal[True] = True
    idea_termination_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class CreditContinuityReport(StrictModel):
    schema_version: Literal[
        "research-idea-credit-continuity-shadow-v1"
    ] = "research-idea-credit-continuity-shadow-v1"
    report_id: str
    report_sha256: str
    traces: list[CreditContinuityTrace] = Field(default_factory=list)
    outcome_counts: dict[str, int] = Field(default_factory=dict)
    same_idea_rescue_count: int = Field(ge=0)
    child_rescue_count: int = Field(ge=0)
    productive_lineage_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    credit_continuity_is_not_truth_authority: Literal[True] = True
    production_selection_authority: Literal[False] = False


class PopulationClosedLoopCaseSummary(StrictModel):
    schema_version: Literal[
        "sis-v2-5-population-closed-loop-case-summary-v1"
    ] = "sis-v2-5-population-closed-loop-case-summary-v1"
    summary_id: str
    summary_sha256: str
    total_research_idea_nodes: int = Field(ge=0)
    conceptual_family_count: int = Field(ge=0)
    generation_family_counts: dict[str, int] = Field(default_factory=dict)
    family_birth_count_by_generation: dict[str, int] = Field(default_factory=dict)
    family_compute_hhi: float = Field(ge=0.0, le=1.0)
    largest_family_compute_share: float = Field(ge=0.0, le=1.0)
    generation2_realization_count: int = Field(ge=0)
    generation2_usable_grounded_realization_count: int = Field(ge=0)
    generation2_same_idea_rescue_count: int = Field(ge=0)
    generation3_realization_count: int = Field(ge=0)
    generation3_usable_grounded_realization_count: int = Field(ge=0)
    generation3_same_idea_rescue_count: int = Field(ge=0)
    credit_continuity_outcome_counts: dict[str, int] = Field(default_factory=dict)
    adaptive_local_event_count: int = Field(ge=0)
    adaptive_boundary_sensitive_event_count: int = Field(ge=0)
    policy_decision_counts: dict[str, int] = Field(default_factory=dict)
    semantic_calibration_sample_count: int = Field(ge=0)
    prospective_observation_count: int = Field(ge=0)
    residual_matched_hypothesis_count: int = Field(ge=0)
    local_realization_llm_calls: int = Field(ge=0)
    prospective_audit_llm_calls: int = Field(ge=0)
    conceptual_family_is_soft_and_recomputable: Literal[True] = True
    realization_failure_is_not_idea_failure: Literal[True] = True
    observation_policy_separated: Literal[True] = True
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False


class PopulationClosedLoopCaseArtifacts(StrictModel):
    family_report: FamilyPopulationReport
    generation2_lifecycle: RealizationLifecycleReport
    generation3_lifecycle: RealizationLifecycleReport
    policy_report: PopulationClosedLoopPolicyReport
    credit_report: CreditContinuityReport
    semantic_sample: SemanticCalibrationSample
    verification_facets: VerificationFacetIntegrationReport
    summary: PopulationClosedLoopCaseSummary


class ProspectiveRunner(Protocol):
    def __call__(
        self,
        *,
        context: HypothesisContext,
        candidate: Any,
        source_stage: str,
        output_prefix: Path,
    ) -> ProspectiveIdentificationShadowArtifact: ...


@dataclass(frozen=True)
class _CompiledAttempt:
    portfolio: HypothesisPortfolio | None
    status: str
    issue_codes: list[str]
    issues: list[str]


def _compile_single(
    *,
    context: HypothesisContext,
    draft: HypothesisPortfolioDraft,
) -> _CompiledAttempt:
    if not draft.hypotheses:
        return _CompiledAttempt(
            portfolio=None,
            status="ABSTAINED",
            issue_codes=[],
            issues=[str(draft.abstention_reason or "model abstained")],
        )
    if len(draft.hypotheses) != 1:
        return _CompiledAttempt(
            portfolio=None,
            status="CARDINALITY_REJECTED",
            issue_codes=["EXPECTED_EXACTLY_ONE_HYPOTHESIS"],
            issues=["Expected exactly one hypothesis realization."],
        )
    try:
        portfolio = HypothesisCompiler().compile(context, draft)
    except HypothesisCompileError as exc:
        return _CompiledAttempt(
            portfolio=None,
            status="COMPILE_REJECTED",
            issue_codes=[row.code for row in exc.issues],
            issues=[f"{row.code}:{row.location}:{row.message}" for row in exc.issues],
        )
    except Exception as exc:
        return _CompiledAttempt(
            portfolio=None,
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
            portfolio=None,
            status="VALIDATION_REJECTED",
            issue_codes=_dedupe([row.split(":", 1)[0] for row in issues]),
            issues=issues,
        )
    return _CompiledAttempt(portfolio=portfolio, status="MATERIALIZED", issue_codes=[], issues=[])


def _alternate_prompt(
    *,
    base: HypothesisPrompt,
    node: ResearchIdeaNode,
    prior_observations: Sequence[ScientificFeedbackFacetObservation],
    attempt_index: int,
) -> HypothesisPrompt:
    status_lines = [
        {
            "materialization_status": row.materialization_status,
            "prospective_identifiability": row.prospective_identifiability,
            "issue_codes": row.materialization_issue_codes,
        }
        for row in prior_observations
    ]
    system = base.system_prompt + """

SIS-v2.5 SAME-IDEA LOCAL REALIZATION SEARCH
===========================================
This is a local realization search under the SAME ResearchIdea, not an idea-evolution step.
Do not change the ResearchIdea kernel to make grounding easier.
Seek an alternate grounded formulation, eligible premise combination, observable framing, or falsifier while preserving the same scientific commitments.
A failed or NOT_OPERATIONALIZABLE prior realization is evidence about that realization only.
If no coherent alternate realization exists under the supplied grounded context, abstain.
"""
    payload = {
        "same_research_idea_id": node.idea_id,
        "attempt_index": attempt_index,
        "kernel_must_be_preserved": node.kernel.model_dump(mode="json"),
        "prior_realization_observations": status_lines,
    }
    user = (
        base.user_prompt
        + "\n\nSIS-v2.5 LOCAL REALIZATION SEARCH\n"
        + "=================================\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\nReturn exactly one alternate grounded realization of the SAME ResearchIdea or abstain."
    )
    return HypothesisPrompt.create(system_prompt=system, user_prompt=user)


def _grounding_integrity(status: str) -> str:
    if status == "MATERIALIZED":
        return "PASSED"
    if status in {"COMPILE_REJECTED", "VALIDATION_REJECTED", "CARDINALITY_REJECTED"}:
        return "FAILED"
    return "NOT_ASSESSED"


def _observation_from_attempt(
    *,
    link: IdeaRealizationLink,
    prospective: ProspectiveIdentificationShadowArtifact | None,
) -> ScientificFeedbackFacetObservation:
    systems = ["SIS_V2_5_REALIZATION_LIFECYCLE"]
    versions = [link.schema_version]
    refs = list(link.source_report_ids)
    if prospective is not None:
        systems.append("PROSPECTIVE_IDENTIFICATION_MATERIALIZATION_SHADOW")
        versions.append(prospective.schema_version)
        if prospective.artifact_path:
            refs.append(str(prospective.artifact_path))
    return ScientificFeedbackFacetObservation(
        observation_id=_stable_id(
            "scientific_feedback_observation",
            link.realization_id,
            link.materialization_status,
            prospective.model_dump(mode="json") if prospective is not None else None,
        ),
        idea_id=link.idea_id,
        realization_id=link.realization_id,
        hypothesis_id=link.hypothesis_id,
        materialization_status=link.materialization_status,
        grounding_integrity=_grounding_integrity(link.materialization_status),
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


def _is_usable(observation: ScientificFeedbackFacetObservation) -> bool:
    return bool(
        observation.materialization_status == "MATERIALIZED"
        and not (
            observation.prospective_status == "COMPLETE"
            and observation.prospective_contract_integrity_passed is True
            and observation.prospective_identifiability == "NOT_OPERATIONALIZABLE"
        )
    )


def run_realization_lifecycle(
    *,
    execution: OffspringExecutionReport,
    context: HypothesisContext,
    backend: HypothesisDraftBackend,
    output_dir: Path,
    max_ideas: int = 8,
    max_realizations_per_idea: int = 2,
    max_per_parent: int = 2,
    max_repair_attempts: int = 1,
    prospective_runner: ProspectiveRunner | None = None,
    max_prospective_audits: int = 12,
    seed_realization_report: OffspringRealizationReport | None = None,
    seed_portfolio: HypothesisPortfolio | None = None,
) -> tuple[RealizationLifecycleReport, HypothesisPortfolio]:
    if max_realizations_per_idea < 1:
        raise ValueError("max_realizations_per_idea must be >= 1")
    if max_repair_attempts < 0:
        raise ValueError("max_repair_attempts must be >= 0")
    if max_prospective_audits < 0:
        raise ValueError("max_prospective_audits must be >= 0")

    node_by_id = {row.idea_id: row for row in execution.offspring_nodes}
    semantic_by_id = {row.idea_id: row for row in execution.semantic_records}
    target_ids = select_offspring_for_realization(
        execution,
        max_realizations=max_ideas,
        max_per_parent=max_per_parent,
    )
    seed_record_by_idea = {
        row.idea_id: row for row in (seed_realization_report.records if seed_realization_report else [])
    }
    seed_card_by_id = {
        card.hypothesis_id: card for card in (seed_portfolio.hypotheses if seed_portfolio else [])
    }

    links: list[IdeaRealizationLink] = []
    observations: list[ScientificFeedbackFacetObservation] = []
    cards: list[Any] = []
    seen_hypothesis_ids: set[str] = set()
    prospective_count = 0
    prospective_calls_by_idea: Counter[str] = Counter()

    def attach_prospective(
        *,
        link: IdeaRealizationLink,
        card: Any | None,
    ) -> ScientificFeedbackFacetObservation:
        nonlocal prospective_count
        artifact = None
        if (
            card is not None
            and prospective_runner is not None
            and prospective_count < max_prospective_audits
        ):
            prefix = output_dir / "prospective" / f"{link.realization_id.replace(':', '_')}"
            artifact = prospective_runner(
                context=context,
                candidate=card,
                source_stage=f"SIS_V2_5_G{execution.generation_index}_LOCAL_REALIZATION",
                output_prefix=prefix,
            )
            prospective_count += 1
            prospective_calls_by_idea[link.idea_id] += 1
        obs = _observation_from_attempt(link=link, prospective=artifact)
        observations.append(obs)
        return obs

    for idea_id in target_ids:
        node = node_by_id[idea_id]
        semantic = semantic_by_id[idea_id]
        idea_links: list[IdeaRealizationLink] = []
        idea_observations: list[ScientificFeedbackFacetObservation] = []

        seed = seed_record_by_idea.get(idea_id)
        if seed is not None:
            rid = _stable_id("realization", execution.report_id, idea_id, 1, "IMPORTED_EXISTING")
            link = IdeaRealizationLink(
                realization_id=rid,
                idea_id=idea_id,
                generation_index=execution.generation_index,
                hypothesis_id=seed.hypothesis_id,
                source_context_id=context.context_id,
                realization_kind="IMPORTED_EXISTING",
                attempt_index=1,
                materialization_status=seed.status,
                issue_codes=list(seed.issue_codes),
                repair_attempt_count=int(seed.repair_attempt_count or 0),
                llm_call_count=int(seed.generation_attempt_count or 0),
                source_report_ids=[seed_realization_report.report_id],
            )
            links.append(link)
            idea_links.append(link)
            card = seed_card_by_id.get(seed.hypothesis_id) if seed.hypothesis_id else None
            if seed.status == "MATERIALIZED" and card is None:
                link = link.model_copy(
                    update={
                        "materialization_status": "COMPILE_REJECTED",
                        "hypothesis_id": None,
                        "issue_codes": _dedupe([*link.issue_codes, "MISSING_SEED_HYPOTHESIS_CARD"]),
                    }
                )
                links[-1] = link
                idea_links[-1] = link
            if card is not None and card.hypothesis_id not in seen_hypothesis_ids:
                cards.append(card)
                seen_hypothesis_ids.add(card.hypothesis_id)
            obs = attach_prospective(link=link, card=card)
            idea_observations.append(obs)

        while len(idea_links) < max_realizations_per_idea:
            if idea_observations and any(_is_usable(row) for row in idea_observations):
                break
            attempt_index = len(idea_links) + 1
            base_prompt = build_realization_prompt(context=context, node=node, semantic=semantic)
            prompt = (
                base_prompt
                if not idea_links
                else _alternate_prompt(
                    base=base_prompt,
                    node=node,
                    prior_observations=idea_observations,
                    attempt_index=attempt_index,
                )
            )
            kind: Literal["INITIAL", "ALTERNATE"] = "INITIAL" if not idea_links else "ALTERNATE"
            calls = 0
            repairs = 0
            issue_codes: list[str] = []
            issues: list[str] = []
            status = "GENERATION_FAILED"
            portfolio = None
            draft = None
            try:
                generation = backend.generate(prompt)
                calls += 1
                draft = generation.draft
                compiled = _compile_single(context=context, draft=draft)
                portfolio = compiled.portfolio
                status = compiled.status
                issue_codes = compiled.issue_codes
                issues = compiled.issues
            except Exception as exc:
                status = "GENERATION_FAILED"
                issue_codes = [type(exc).__name__]
                issues = [str(exc)]

            while (
                draft is not None
                and status in {"COMPILE_REJECTED", "VALIDATION_REJECTED", "CARDINALITY_REJECTED"}
                and repairs < max_repair_attempts
            ):
                repairs += 1
                feedback = "\n".join(
                    [
                        "SIS-v2.5 local realization repair under the SAME ResearchIdea.",
                        "Do not change the ResearchIdea kernel.",
                        "Use only eligible grounded premise IDs from the supplied context.",
                        "Return one corrected realization or abstain.",
                        "Issues:",
                        *[f"- {row}" for row in issues],
                    ]
                )
                try:
                    repaired = backend.repair(prompt, draft, feedback)
                    calls += 1
                    draft = repaired.draft
                    compiled = _compile_single(context=context, draft=draft)
                    portfolio = compiled.portfolio
                    status = compiled.status
                    issue_codes = compiled.issue_codes
                    issues = compiled.issues
                except Exception as exc:
                    status = "GENERATION_FAILED"
                    issue_codes = [type(exc).__name__]
                    issues = [str(exc)]
                    portfolio = None
                    break

            card = None
            hypothesis_id = None
            if portfolio is not None and portfolio.hypotheses:
                card = portfolio.hypotheses[0]
                hypothesis_id = str(card.hypothesis_id)
                if hypothesis_id in seen_hypothesis_ids:
                    status = "COMPILE_REJECTED"
                    issue_codes = _dedupe([*issue_codes, "DUPLICATE_HYPOTHESIS_ID_ACROSS_REALIZATIONS"])
                    hypothesis_id = None
                    card = None
                else:
                    seen_hypothesis_ids.add(hypothesis_id)
                    cards.append(card)

            rid = _stable_id(
                "realization",
                execution.report_id,
                idea_id,
                attempt_index,
                kind,
                hypothesis_id,
                status,
            )
            link = IdeaRealizationLink(
                realization_id=rid,
                idea_id=idea_id,
                generation_index=execution.generation_index,
                hypothesis_id=hypothesis_id,
                source_context_id=context.context_id,
                realization_kind=kind,
                attempt_index=attempt_index,
                parent_realization_id=(idea_links[-1].realization_id if idea_links else None),
                materialization_status=status,
                issue_codes=_dedupe(issue_codes),
                repair_attempt_count=repairs,
                llm_call_count=calls,
                source_report_ids=[execution.report_id],
            )
            links.append(link)
            idea_links.append(link)
            obs = attach_prospective(link=link, card=card)
            idea_observations.append(obs)

    portfolio_id = _stable_id(
        f"g{execution.generation_index}_v2_5_lifecycle_portfolio",
        execution.report_id,
        context.context_id,
        [card.hypothesis_id for card in cards],
    )
    combined = HypothesisPortfolio(
        portfolio_id=portfolio_id,
        domain_profile_id=context.domain_profile_id,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        source_report_id=context.source_report_id,
        source_report_sha256=context.source_report_sha256,
        hypotheses=cards,
        abstention_reason=(
            None
            if cards
            else f"No G{execution.generation_index} ResearchIdea realization materialized under the SIS-v2.5 bounded lifecycle."
        ),
    )
    local_states = build_local_search_states(
        idea_ids=target_ids,
        links=links,
        observations=observations,
        adaptive_events=[],
        local_search_budget=max_realizations_per_idea,
    )
    usable = sum(row.usable_grounded_realization_count for row in local_states)
    not_op = sum(row.not_operationalizable_count for row in local_states)
    provisional = RealizationLifecycleReport(
        report_id="pending",
        report_sha256="pending",
        generation_index=execution.generation_index,
        source_offspring_execution_report_id=execution.report_id,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        target_idea_ids=target_ids,
        links=links,
        observations=observations,
        local_states=local_states,
        output_portfolio_id=combined.portfolio_id,
        materialized_hypothesis_count=sum(
            row.materialization_status == "MATERIALIZED" for row in links
        ),
        usable_grounded_realization_count=usable,
        rescued_within_same_idea_count=sum(row.rescued_within_same_idea for row in local_states),
        local_search_exhausted_count=sum(row.local_search_exhausted for row in local_states),
        prospective_audit_count=prospective_count,
        prospective_not_operationalizable_count=not_op,
        realization_llm_call_count=sum(row.llm_call_count for row in links if row.realization_kind != "IMPORTED_EXISTING"),
        prospective_audit_llm_call_count=sum(prospective_calls_by_idea.values()),
        max_realizations_per_idea=max_realizations_per_idea,
        max_repair_attempts=max_repair_attempts,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return (
        provisional.model_copy(
            update={
                "report_id": f"g{execution.generation_index}_realization_lifecycle:{digest[:20]}",
                "report_sha256": digest,
            }
        ),
        combined,
    )


def _rebuild_local_states_with_adaptive_events(
    *,
    lifecycle: RealizationLifecycleReport,
    adaptive_events: Sequence[AdaptiveLocalSearchEvent],
) -> list[IdeaLocalSearchState]:
    return build_local_search_states(
        idea_ids=lifecycle.target_idea_ids,
        links=lifecycle.links,
        observations=lifecycle.observations,
        adaptive_events=adaptive_events,
        local_search_budget=lifecycle.max_realizations_per_idea,
    )


def build_population_closed_loop_case(
    *,
    source_generational: GenerationalIdeaSearchShadowReport,
    generation2_execution: OffspringExecutionReport,
    generation2_lifecycle: RealizationLifecycleReport,
    generation2_audit: ConceptualDeltaAuditReport,
    generation3_execution: OffspringExecutionReport,
    generation3_lifecycle: RealizationLifecycleReport,
    generation3_audit: ConceptualDeltaAuditReport,
    adaptive_raw_plans: Sequence[Mapping[str, Any]] = (),
    adaptive_source_artifacts: Sequence[str] = (),
    residual_reports: Sequence[Mapping[str, Any]] = (),
    additional_residual_source_portfolio_ids: Sequence[str] = (),
    family_overconcentration_threshold: float = 0.35,
    semantic_sample_size: int = 24,
) -> PopulationClosedLoopCaseArtifacts:
    if not 0.0 <= family_overconcentration_threshold <= 1.0:
        raise ValueError("family_overconcentration_threshold must be in [0, 1]")
    nodes = [
        *source_generational.research_ideas,
        *generation2_execution.offspring_nodes,
        *generation3_execution.offspring_nodes,
    ]
    node_by_id = {row.idea_id: row for row in nodes}
    if len(node_by_id) != len(nodes):
        # Exact duplicate node IDs across reports should represent the same object.
        nodes = list(node_by_id.values())
    assignments = assign_conceptual_families(nodes)

    all_links = [*generation2_lifecycle.links, *generation3_lifecycle.links]
    all_observations = [
        *generation2_lifecycle.observations,
        *generation3_lifecycle.observations,
    ]
    hypothesis_to_idea = {
        row.hypothesis_id: row.idea_id
        for row in all_links
        if row.hypothesis_id is not None
    }
    adaptive_events = adapt_adaptive_controller_plans(
        adaptive_raw_plans,
        hypothesis_to_idea=hypothesis_to_idea,
        source_artifacts=adaptive_source_artifacts,
    )

    enriched_observations, verification_facets = enrich_observations_from_residual_reports(
        all_observations,
        residual_reports=residual_reports,
        allowed_source_portfolio_ids=[
            generation2_lifecycle.output_portfolio_id,
            generation3_lifecycle.output_portfolio_id,
            *additional_residual_source_portfolio_ids,
        ],
        adaptive_events=adaptive_events,
    )
    g2_realization_ids = {row.realization_id for row in generation2_lifecycle.links}
    g3_realization_ids = {row.realization_id for row in generation3_lifecycle.links}
    g2_observations = [row for row in enriched_observations if row.realization_id in g2_realization_ids]
    g3_observations = [row for row in enriched_observations if row.realization_id in g3_realization_ids]
    generation2_lifecycle = generation2_lifecycle.model_copy(update={"observations": g2_observations})
    generation3_lifecycle = generation3_lifecycle.model_copy(update={"observations": g3_observations})

    g2_states = _rebuild_local_states_with_adaptive_events(
        lifecycle=generation2_lifecycle,
        adaptive_events=adaptive_events,
    )
    g3_states = _rebuild_local_states_with_adaptive_events(
        lifecycle=generation3_lifecycle,
        adaptive_events=adaptive_events,
    )
    generation2_lifecycle = generation2_lifecycle.model_copy(update={"local_states": g2_states})
    generation3_lifecycle = generation3_lifecycle.model_copy(update={"local_states": g3_states})

    prospective_calls_by_idea: Counter[str] = Counter()
    for obs in enriched_observations:
        if "PROSPECTIVE_IDENTIFICATION_MATERIALIZATION_SHADOW" in obs.source_systems:
            prospective_calls_by_idea[obs.idea_id] += 1

    family_report = build_family_population_report(
        nodes=nodes,
        assignments=assignments,
        realization_links=all_links,
        observations=enriched_observations,
        adaptive_events=adaptive_events,
        executions=[generation2_execution, generation3_execution],
        audits=[generation2_audit, generation3_audit],
        prospective_audit_llm_calls_by_idea=prospective_calls_by_idea,
    )
    decisions = build_idea_learning_decisions(
        local_states=[*g2_states, *g3_states],
        observations=enriched_observations,
        adaptive_events=adaptive_events,
        family_report=family_report,
        overconcentrated_family_share=family_overconcentration_threshold,
    )
    decision_counts = Counter(row.disposition for row in decisions)
    policy_provisional = PopulationClosedLoopPolicyReport(
        report_id="pending",
        report_sha256="pending",
        decisions=decisions,
        decision_counts=dict(sorted(decision_counts.items())),
        local_search_exhausted_count=sum(row.local_search_exhausted for row in [*g2_states, *g3_states]),
        transform_escalation_count=decision_counts.get("ESCALATE_TRANSFORM", 0),
        family_overconcentration_pressure_count=sum(
            "OVERCONCENTRATED_FAMILY_ADDS_EXPLORATION_PRESSURE_NOT_HARD_GATE" in row.reason_codes
            for row in decisions
        ),
    )
    payload = policy_provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    policy_report = policy_provisional.model_copy(
        update={
            "report_id": f"population_closed_loop_policy:{digest[:20]}",
            "report_sha256": digest,
        }
    )

    g2_nodes = [row for row in generation2_execution.offspring_nodes]
    g3_nodes = [row for row in generation3_execution.offspring_nodes]
    traces = build_credit_continuity_traces(
        generation2_nodes=g2_nodes,
        generation3_nodes=g3_nodes,
        g2_states=g2_states,
        g3_states=g3_states,
    )
    outcome_counts = Counter(row.outcome for row in traces)
    credit_provisional = CreditContinuityReport(
        report_id="pending",
        report_sha256="pending",
        traces=traces,
        outcome_counts=dict(sorted(outcome_counts.items())),
        same_idea_rescue_count=outcome_counts.get("RESCUED_WITHIN_IDEA", 0),
        child_rescue_count=outcome_counts.get("RESCUED_BY_CHILD", 0),
        productive_lineage_count=outcome_counts.get("PRODUCTIVE_LINEAGE", 0),
        unresolved_count=outcome_counts.get("UNRESOLVED", 0),
    )
    payload = credit_provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    credit_report = credit_provisional.model_copy(
        update={
            "report_id": f"credit_continuity:{digest[:20]}",
            "report_sha256": digest,
        }
    )

    semantic_sample = build_semantic_calibration_sample(
        nodes=nodes,
        audits=[generation2_audit, generation3_audit],
        max_items=semantic_sample_size,
    )
    summary_provisional = PopulationClosedLoopCaseSummary(
        summary_id="pending",
        summary_sha256="pending",
        total_research_idea_nodes=len(nodes),
        conceptual_family_count=family_report.family_count,
        generation_family_counts=family_report.generation_family_counts,
        family_birth_count_by_generation=family_report.family_birth_count_by_generation,
        family_compute_hhi=family_report.family_compute_hhi,
        largest_family_compute_share=family_report.largest_family_compute_share,
        generation2_realization_count=len(generation2_lifecycle.links),
        generation2_usable_grounded_realization_count=sum(
            row.usable_grounded_realization_count for row in g2_states
        ),
        generation2_same_idea_rescue_count=sum(row.rescued_within_same_idea for row in g2_states),
        generation3_realization_count=len(generation3_lifecycle.links),
        generation3_usable_grounded_realization_count=sum(
            row.usable_grounded_realization_count for row in g3_states
        ),
        generation3_same_idea_rescue_count=sum(row.rescued_within_same_idea for row in g3_states),
        credit_continuity_outcome_counts=credit_report.outcome_counts,
        adaptive_local_event_count=len(adaptive_events),
        adaptive_boundary_sensitive_event_count=sum(
            row.interpreted_scope == "BOUNDARY_SENSITIVE" for row in adaptive_events
        ),
        policy_decision_counts=policy_report.decision_counts,
        semantic_calibration_sample_count=len(semantic_sample.items),
        prospective_observation_count=verification_facets.prospective_observation_count,
        residual_matched_hypothesis_count=verification_facets.residual_matched_hypothesis_count,
        local_realization_llm_calls=(
            generation2_lifecycle.realization_llm_call_count
            + generation3_lifecycle.realization_llm_call_count
        ),
        prospective_audit_llm_calls=(
            generation2_lifecycle.prospective_audit_llm_call_count
            + generation3_lifecycle.prospective_audit_llm_call_count
        ),
    )
    payload = summary_provisional.model_dump(mode="json")
    payload.pop("summary_id", None)
    payload.pop("summary_sha256", None)
    digest = _sha(payload)
    summary = summary_provisional.model_copy(
        update={
            "summary_id": f"sis_v2_5_case:{digest[:20]}",
            "summary_sha256": digest,
        }
    )
    return PopulationClosedLoopCaseArtifacts(
        family_report=family_report,
        generation2_lifecycle=generation2_lifecycle,
        generation3_lifecycle=generation3_lifecycle,
        policy_report=policy_report,
        credit_report=credit_report,
        semantic_sample=semantic_sample,
        verification_facets=verification_facets,
        summary=summary,
    )


__all__ = [
    "CreditContinuityReport",
    "PopulationClosedLoopCaseArtifacts",
    "PopulationClosedLoopCaseSummary",
    "PopulationClosedLoopPolicyReport",
    "RealizationLifecycleReport",
    "build_population_closed_loop_case",
    "run_realization_lifecycle",
]
