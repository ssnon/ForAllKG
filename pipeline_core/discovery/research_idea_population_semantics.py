from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.research_idea_contracts import ResearchIdeaNode
from pipeline_core.discovery.research_idea_multigeneration import ConceptualDeltaAuditReport
from pipeline_core.discovery.research_idea_offspring_execution import OffspringExecutionReport
from pipeline_core.discovery.research_idea_semantics import assess_conceptual_family


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


FamilyAssignmentRelation = Literal[
    "REPRESENTATIVE",
    "SAME_FAMILY",
    "NEW_FAMILY_ADJACENT_TO_EXISTING",
    "NEW_DISTINCT_FAMILY",
]
RealizationKind = Literal[
    "IMPORTED_EXISTING",
    "INITIAL",
    "ALTERNATE",
    "REPAIR",
    "EVIDENCE_REAXIS",
    "GRAPH_RETRAVERSAL",
]
LocalSearchScope = Literal[
    "LOCAL_KEEP",
    "LOCAL_EVIDENCE",
    "LOCAL_REPAIR",
    "LOCAL_REAXIS",
    "BOUNDARY_SENSITIVE",
    "LOCAL_CONTEXT_RESET",
    "LOCAL_EXHAUSTED",
]
GroundingIntegrity = Literal["PASSED", "FAILED", "NOT_ASSESSED"]
IdeaLearningDisposition = Literal[
    "RETAIN_EXPLOIT",
    "RETAIN_EXPLORE",
    "ESCALATE_TRANSFORM",
    "OBSERVE_UNDEREXPLORED",
]
CreditContinuityOutcome = Literal[
    "RESCUED_WITHIN_IDEA",
    "RESCUED_BY_CHILD",
    "PRODUCTIVE_LINEAGE",
    "UNRESOLVED",
    "NOT_OBSERVED",
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


class ConceptualFamilyAssignment(StrictModel):
    """Soft, versioned population abstraction; never ResearchIdea identity."""

    schema_version: Literal[
        "conceptual-family-assignment-v1"
    ] = "conceptual-family-assignment-v1"
    idea_id: str = Field(min_length=1)
    family_key: str = Field(min_length=1)
    representative_idea_id: str = Field(min_length=1)
    relation_to_representative: FamilyAssignmentRelation
    similarity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    nearest_existing_family_key: str | None = None
    nearest_existing_relation: str | None = None
    assignment_method: Literal[
        "GREEDY_REPRESENTATIVE_WEIGHTED_TOKEN_JACCARD_V1"
    ] = "GREEDY_REPRESENTATIVE_WEIGHTED_TOKEN_JACCARD_V1"
    assignment_method_version: Literal[
        "sis-v2.5-conceptual-family-v1"
    ] = "sis-v2.5-conceptual-family-v1"
    recomputable: Literal[True] = True
    immutable_idea_identity_field: Literal[False] = False
    search_policy_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class ConceptualFamilyTransition(StrictModel):
    child_idea_id: str
    child_family_key: str
    parent_idea_ids: list[str] = Field(default_factory=list)
    parent_family_keys: list[str] = Field(default_factory=list)
    transition_kind: Literal[
        "WITHIN_FAMILY",
        "ADJACENT_FAMILY_BIRTH",
        "DISTINCT_FAMILY_BIRTH",
        "INDETERMINATE_FAMILY_SHIFT",
    ]
    operator_id: str | None = None
    generation_index: int = Field(ge=1)
    diagnostic_only: Literal[True] = True


class IdeaRealizationLink(StrictModel):
    """Explicit 1:N association without modifying HypothesisCard identity."""

    schema_version: Literal[
        "idea-realization-link-v1"
    ] = "idea-realization-link-v1"
    realization_id: str = Field(min_length=1)
    idea_id: str = Field(min_length=1)
    generation_index: int = Field(ge=0)
    hypothesis_id: str | None = None
    source_context_id: str = Field(min_length=1)
    realization_kind: RealizationKind
    attempt_index: int = Field(ge=1)
    parent_realization_id: str | None = None
    materialization_status: str = Field(min_length=1)
    issue_codes: list[str] = Field(default_factory=list)
    repair_attempt_count: int = Field(ge=0, default=0)
    llm_call_count: int = Field(ge=0, default=0)
    source_report_ids: list[str] = Field(default_factory=list)
    same_research_idea_required: Literal[True] = True
    realization_identity_is_not_idea_identity: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _normalize(self) -> "IdeaRealizationLink":
        self.issue_codes = _dedupe(self.issue_codes)
        self.source_report_ids = _dedupe(self.source_report_ids)
        return self


class ScientificFeedbackFacetObservation(StrictModel):
    """Policy-free realization observation compiled from conservative checks."""

    schema_version: Literal[
        "scientific-feedback-facet-observation-v1"
    ] = "scientific-feedback-facet-observation-v1"
    observation_id: str = Field(min_length=1)
    idea_id: str = Field(min_length=1)
    realization_id: str = Field(min_length=1)
    hypothesis_id: str | None = None
    target_scope: Literal["REALIZATION"] = "REALIZATION"

    materialization_status: str = Field(min_length=1)
    grounding_integrity: GroundingIntegrity
    materialization_issue_codes: list[str] = Field(default_factory=list)

    prospective_status: str | None = None
    current_evidence_status: str | None = None
    prospective_identifiability: str | None = None
    directionality_mode: str | None = None
    measurement_compatibility_mode: str | None = None
    prospective_contract_integrity_passed: bool | None = None

    relation_validity: str | None = None
    residual_epistemic_state: str | None = None
    residual_state_reason: str | None = None
    external_novelty_status: str | None = None

    source_systems: list[str] = Field(default_factory=list)
    source_versions: list[str] = Field(default_factory=list)
    source_artifact_refs: list[str] = Field(default_factory=list)

    observation_only: Literal[True] = True
    search_policy_authority: Literal[False] = False
    idea_termination_authority: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class AdaptiveLocalSearchEvent(StrictModel):
    event_id: str = Field(min_length=1)
    idea_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    root_hypothesis_id: str | None = None
    action: str = Field(min_length=1)
    interpreted_scope: LocalSearchScope
    action_scope_rank: int | None = None
    requires_research_idea_semantic_comparison: bool = False
    source_plan_id: str | None = None
    source_artifact: str | None = None
    observation_or_local_search_trace_only: Literal[True] = True
    search_policy_authority: Literal[False] = False
    idea_identity_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False




class VerificationFacetIntegrationReport(StrictModel):
    schema_version: Literal[
        "scientific-feedback-facet-integration-report-v1"
    ] = "scientific-feedback-facet-integration-report-v1"
    report_id: str
    report_sha256: str
    observation_count: int = Field(ge=0)
    prospective_observation_count: int = Field(ge=0)
    residual_report_count: int = Field(ge=0)
    residual_matched_hypothesis_count: int = Field(ge=0)
    residual_unmatched_hypothesis_ids: list[str] = Field(default_factory=list)
    adaptive_local_event_count: int = Field(ge=0)
    adaptive_boundary_sensitive_event_count: int = Field(ge=0)
    exact_hypothesis_identity_required: Literal[True] = True
    exact_portfolio_lineage_required_for_residual: Literal[True] = True
    observation_policy_separated: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

class IdeaLocalSearchState(StrictModel):
    idea_id: str
    realization_ids: list[str] = Field(default_factory=list)
    realization_count: int = Field(ge=0)
    materialized_count: int = Field(ge=0)
    usable_grounded_realization_count: int = Field(ge=0)
    not_operationalizable_count: int = Field(ge=0)
    failed_or_abstained_count: int = Field(ge=0)
    adaptive_event_ids: list[str] = Field(default_factory=list)
    local_search_budget: int = Field(ge=1)
    local_search_budget_used: int = Field(ge=0)
    local_search_exhausted: bool
    rescued_within_same_idea: bool
    one_realization_failure_is_not_idea_failure: Literal[True] = True
    idea_termination_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "IdeaLocalSearchState":
        if self.realization_count != len(self.realization_ids):
            raise ValueError("realization_count mismatch")
        if self.local_search_budget_used > self.local_search_budget:
            raise ValueError("local search budget exceeded")
        return self


class FamilySearchState(StrictModel):
    family_key: str
    representative_idea_id: str
    member_idea_ids: list[str] = Field(default_factory=list)
    idea_count: int = Field(ge=0)
    generation_counts: dict[str, int] = Field(default_factory=dict)
    exact_kernel_unique_count: int = Field(ge=0)
    realization_count: int = Field(ge=0)
    materialized_realization_count: int = Field(ge=0)
    usable_grounded_realization_count: int = Field(ge=0)
    not_operationalizable_realization_count: int = Field(ge=0)
    adaptive_local_event_count: int = Field(ge=0)
    offspring_generation_llm_calls: int = Field(ge=0)
    realization_llm_calls: int = Field(ge=0)
    prospective_audit_llm_calls: int = Field(ge=0)
    total_tracked_llm_calls: int = Field(ge=0)
    compute_share: float = Field(ge=0.0, le=1.0)
    productive_offspring_count: int = Field(ge=0)
    family_expansion_event_count: int = Field(ge=0)
    productive_operator_counts: dict[str, int] = Field(default_factory=dict)
    mutable_search_abstraction: Literal[True] = True
    permanent_scientific_ontology: Literal[False] = False


class FamilyPopulationReport(StrictModel):
    schema_version: Literal[
        "conceptual-family-population-shadow-v1"
    ] = "conceptual-family-population-shadow-v1"
    report_id: str
    report_sha256: str
    assignments: list[ConceptualFamilyAssignment] = Field(default_factory=list)
    family_states: list[FamilySearchState] = Field(default_factory=list)
    transitions: list[ConceptualFamilyTransition] = Field(default_factory=list)
    idea_count: int = Field(ge=0)
    family_count: int = Field(ge=0)
    generation_family_counts: dict[str, int] = Field(default_factory=dict)
    family_birth_count_by_generation: dict[str, int] = Field(default_factory=dict)
    family_compute_hhi: float = Field(ge=0.0, le=1.0)
    largest_family_compute_share: float = Field(ge=0.0, le=1.0)
    family_assignment_is_soft_and_recomputable: Literal[True] = True
    family_is_not_research_idea_identity: Literal[True] = True
    family_is_not_hard_selection_gate: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class IdeaLearningDecision(StrictModel):
    idea_id: str
    family_key: str
    disposition: IdeaLearningDisposition
    recommended_channels: list[Literal[
        "EXPLOIT", "TRANSFORM", "EXPLORE", "WILDCARD"
    ]] = Field(default_factory=list)
    transformation_pressure: Literal["HIGH", "MEDIUM", "LOW"]
    exploration_pressure: Literal["HIGH", "MEDIUM", "LOW"]
    exploit_pressure: Literal["HIGH", "MEDIUM", "LOW"]
    reason_codes: list[str] = Field(default_factory=list)
    source_observation_ids: list[str] = Field(default_factory=list)
    source_adaptive_event_ids: list[str] = Field(default_factory=list)
    family_compute_share: float = Field(ge=0.0, le=1.0)
    local_search_exhausted: bool
    one_realization_failure_is_not_idea_failure: Literal[True] = True
    observation_policy_separated: Literal[True] = True
    family_is_soft_pressure_not_gate: Literal[True] = True
    idea_termination_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class CreditContinuityTrace(StrictModel):
    trace_id: str
    generation2_idea_id: str
    generation2_realization_ids: list[str] = Field(default_factory=list)
    generation2_had_usable_realization: bool
    generation2_rescued_within_same_idea: bool
    generation3_child_idea_ids: list[str] = Field(default_factory=list)
    generation3_usable_child_idea_ids: list[str] = Field(default_factory=list)
    outcome: CreditContinuityOutcome
    reason_codes: list[str] = Field(default_factory=list)
    credit_continuity_is_diagnostic_not_truth: Literal[True] = True


class SemanticCalibrationSampleItem(StrictModel):
    generation_index: int
    child_idea_id: str
    parent_idea_ids: list[str]
    deterministic_identity_relation: str
    audit_category: str
    audit_confidence: float
    comparator_audit_disagreement: bool
    parent_kernels: list[dict[str, Any]] = Field(default_factory=list)
    child_kernel: dict[str, Any]
    human_label: None = None
    human_annotation_pending: Literal[True] = True


class SemanticCalibrationSample(StrictModel):
    schema_version: Literal[
        "research-idea-semantic-calibration-sample-v1"
    ] = "research-idea-semantic-calibration-sample-v1"
    sample_id: str
    items: list[SemanticCalibrationSampleItem] = Field(default_factory=list)
    disagreement_items_first: Literal[True] = True
    no_automatic_gold_labels: Literal[True] = True


_ADAPTIVE_SCOPE: dict[str, LocalSearchScope] = {
    "KEEP": "LOCAL_KEEP",
    "RETRIEVE_MORE": "LOCAL_EVIDENCE",
    "SAME_PREMISE_SHARPEN": "LOCAL_REPAIR",
    "EVIDENCE_REAXIS": "LOCAL_REAXIS",
    "AXIS_MUTATION": "BOUNDARY_SENSITIVE",
    "REQUEST_GRAPH_RETRAVERSAL": "LOCAL_CONTEXT_RESET",
    "STOP": "LOCAL_EXHAUSTED",
}


def assign_conceptual_families(
    nodes: Sequence[ResearchIdeaNode],
) -> list[ConceptualFamilyAssignment]:
    """Greedy representative clustering using the existing soft family diagnostic.

    The output is intentionally a versioned search-time view. Adding new evidence or
    changing clustering policy may legitimately change assignments without changing
    ResearchIdeaNode identity.
    """

    ordered = sorted(nodes, key=lambda row: (row.generation_index, row.idea_id))
    representatives: list[tuple[str, ResearchIdeaNode]] = []
    out: list[ConceptualFamilyAssignment] = []

    for node in ordered:
        if not representatives:
            family_key = f"conceptual_family:{node.kernel_sha256[:20]}"
            representatives.append((family_key, node))
            out.append(
                ConceptualFamilyAssignment(
                    idea_id=node.idea_id,
                    family_key=family_key,
                    representative_idea_id=node.idea_id,
                    relation_to_representative="REPRESENTATIVE",
                    similarity=1.0,
                    confidence=1.0,
                )
            )
            continue

        assessed = []
        for family_key, rep in representatives:
            assessment = assess_conceptual_family(rep, node)
            assessed.append((assessment.similarity, family_key, rep, assessment))
        assessed.sort(key=lambda item: (-item[0], item[1]))
        _, nearest_key, nearest_rep, nearest = assessed[0]

        if nearest.relation == "SAME_FAMILY":
            out.append(
                ConceptualFamilyAssignment(
                    idea_id=node.idea_id,
                    family_key=nearest_key,
                    representative_idea_id=nearest_rep.idea_id,
                    relation_to_representative="SAME_FAMILY",
                    similarity=nearest.similarity,
                    confidence=nearest.confidence,
                    nearest_existing_family_key=nearest_key,
                    nearest_existing_relation=nearest.relation,
                )
            )
            continue

        family_key = f"conceptual_family:{node.kernel_sha256[:20]}"
        representatives.append((family_key, node))
        relation = (
            "NEW_FAMILY_ADJACENT_TO_EXISTING"
            if nearest.relation == "ADJACENT_FAMILY"
            else "NEW_DISTINCT_FAMILY"
        )
        out.append(
            ConceptualFamilyAssignment(
                idea_id=node.idea_id,
                family_key=family_key,
                representative_idea_id=node.idea_id,
                relation_to_representative=relation,
                similarity=1.0,
                confidence=nearest.confidence,
                nearest_existing_family_key=nearest_key,
                nearest_existing_relation=nearest.relation,
            )
        )
    return out


def adapt_adaptive_controller_plans(
    raw_plans: Sequence[Mapping[str, Any]],
    *,
    hypothesis_to_idea: Mapping[str, str],
    source_artifacts: Sequence[str] | None = None,
) -> list[AdaptiveLocalSearchEvent]:
    """Attach existing EPS/adaptive-controller traces without granting them idea authority."""

    artifacts = list(source_artifacts or [])
    events: list[AdaptiveLocalSearchEvent] = []
    for plan_index, plan in enumerate(raw_plans):
        if str(plan.get("schema_version") or "") != "adaptive-discovery-controller-plan-v1":
            continue
        plan_id = str(plan.get("plan_id") or "") or None
        source_artifact = artifacts[plan_index] if plan_index < len(artifacts) else None
        for decision in plan.get("decisions", []):
            if not isinstance(decision, Mapping):
                continue
            hypothesis_id = str(decision.get("current_hypothesis_id") or "")
            idea_id = hypothesis_to_idea.get(hypothesis_id)
            if not idea_id:
                continue
            action = str(decision.get("action") or "")
            scope = _ADAPTIVE_SCOPE.get(action)
            if scope is None:
                continue
            events.append(
                AdaptiveLocalSearchEvent(
                    event_id=_stable_id(
                        "adaptive_local_event",
                        plan_id,
                        hypothesis_id,
                        action,
                        plan.get("round_index"),
                    ),
                    idea_id=idea_id,
                    hypothesis_id=hypothesis_id,
                    root_hypothesis_id=(
                        str(decision.get("root_hypothesis_id"))
                        if decision.get("root_hypothesis_id")
                        else None
                    ),
                    action=action,
                    interpreted_scope=scope,
                    action_scope_rank=(
                        int(decision["action_scope_rank"])
                        if decision.get("action_scope_rank") is not None
                        else None
                    ),
                    requires_research_idea_semantic_comparison=(action == "AXIS_MUTATION"),
                    source_plan_id=plan_id,
                    source_artifact=source_artifact,
                )
            )
    return events


def build_family_transitions(
    *,
    nodes: Sequence[ResearchIdeaNode],
    assignments: Sequence[ConceptualFamilyAssignment],
) -> list[ConceptualFamilyTransition]:
    node_by_id = {row.idea_id: row for row in nodes}
    family_by_id = {row.idea_id: row.family_key for row in assignments}
    out: list[ConceptualFamilyTransition] = []
    for child in sorted(nodes, key=lambda row: (row.generation_index, row.idea_id)):
        if not child.parent_idea_ids:
            continue
        parent_ids = [pid for pid in child.parent_idea_ids if pid in node_by_id]
        if not parent_ids:
            continue
        child_family = family_by_id[child.idea_id]
        parent_families = _dedupe([family_by_id[pid] for pid in parent_ids])
        if child_family in set(parent_families):
            kind = "WITHIN_FAMILY"
        else:
            pair_relations = [
                assess_conceptual_family(node_by_id[pid], child).relation
                for pid in parent_ids
            ]
            if "ADJACENT_FAMILY" in pair_relations:
                kind = "ADJACENT_FAMILY_BIRTH"
            elif pair_relations and all(x == "DISTINCT_FAMILY" for x in pair_relations):
                kind = "DISTINCT_FAMILY_BIRTH"
            else:
                kind = "INDETERMINATE_FAMILY_SHIFT"
        out.append(
            ConceptualFamilyTransition(
                child_idea_id=child.idea_id,
                child_family_key=child_family,
                parent_idea_ids=parent_ids,
                parent_family_keys=parent_families,
                transition_kind=kind,
                operator_id=child.operator_id,
                generation_index=child.generation_index,
            )
        )
    return out


def build_family_population_report(
    *,
    nodes: Sequence[ResearchIdeaNode],
    assignments: Sequence[ConceptualFamilyAssignment],
    realization_links: Sequence[IdeaRealizationLink],
    observations: Sequence[ScientificFeedbackFacetObservation],
    adaptive_events: Sequence[AdaptiveLocalSearchEvent] = (),
    executions: Sequence[OffspringExecutionReport] = (),
    audits: Sequence[ConceptualDeltaAuditReport] = (),
    prospective_audit_llm_calls_by_idea: Mapping[str, int] | None = None,
) -> FamilyPopulationReport:
    node_by_id = {row.idea_id: row for row in nodes}
    assignment_by_id = {row.idea_id: row for row in assignments}
    prospective_calls = dict(prospective_audit_llm_calls_by_idea or {})
    transitions = build_family_transitions(nodes=nodes, assignments=assignments)

    family_members: dict[str, list[str]] = defaultdict(list)
    representative: dict[str, str] = {}
    for row in assignments:
        family_members[row.family_key].append(row.idea_id)
        representative.setdefault(row.family_key, row.representative_idea_id)

    links_by_family: dict[str, list[IdeaRealizationLink]] = defaultdict(list)
    for link in realization_links:
        assignment = assignment_by_id.get(link.idea_id)
        if assignment is not None:
            links_by_family[assignment.family_key].append(link)

    obs_by_realization = {row.realization_id: row for row in observations}
    events_by_family: Counter[str] = Counter()
    for event in adaptive_events:
        assignment = assignment_by_id.get(event.idea_id)
        if assignment is not None:
            events_by_family[assignment.family_key] += 1

    offspring_calls: Counter[str] = Counter()
    productive: Counter[str] = Counter()
    productive_ops: dict[str, Counter[str]] = defaultdict(Counter)
    audit_by_child: dict[str, Any] = {}
    for audit in audits:
        for row in audit.records:
            audit_by_child[row.child_idea_id] = row
    for execution in executions:
        semantics = {row.idea_id: row for row in execution.semantic_records}
        for run in execution.run_records:
            assignment = assignment_by_id.get(run.primary_parent_idea_id)
            if assignment is not None:
                offspring_calls[assignment.family_key] += int(run.llm_call_count or 0)
        for child_id, semantic in semantics.items():
            audit = audit_by_child.get(child_id)
            is_productive = bool(
                audit.strong_distinct_child
                if audit is not None
                else semantic.transition.identity_relation == "DIFFERENT_IDEA"
            )
            if not is_productive:
                continue
            for parent_id in semantic.parent_idea_ids:
                assignment = assignment_by_id.get(parent_id)
                if assignment is None:
                    continue
                productive[assignment.family_key] += 1
                productive_ops[assignment.family_key][semantic.chosen_operator_id] += 1

    expansion_by_parent_family: Counter[str] = Counter()
    for transition in transitions:
        if transition.transition_kind == "WITHIN_FAMILY":
            continue
        for family_key in transition.parent_family_keys:
            expansion_by_parent_family[family_key] += 1

    realization_calls: Counter[str] = Counter()
    prospective_family_calls: Counter[str] = Counter()
    for link in realization_links:
        assignment = assignment_by_id.get(link.idea_id)
        if assignment is None:
            continue
        realization_calls[assignment.family_key] += int(link.llm_call_count or 0)
    for idea_id, count in prospective_calls.items():
        assignment = assignment_by_id.get(idea_id)
        if assignment is not None:
            prospective_family_calls[assignment.family_key] += int(count)

    raw_compute: dict[str, int] = {}
    for family_key in family_members:
        raw_compute[family_key] = (
            offspring_calls[family_key]
            + realization_calls[family_key]
            + prospective_family_calls[family_key]
        )
    compute_total = sum(raw_compute.values())

    states: list[FamilySearchState] = []
    for family_key in sorted(family_members):
        members = sorted(family_members[family_key])
        generation_counts = Counter(
            f"G{node_by_id[idea_id].generation_index}" for idea_id in members
        )
        links = links_by_family.get(family_key, [])
        materialized = [row for row in links if row.materialization_status == "MATERIALIZED"]
        usable = 0
        not_operationalizable = 0
        for link in links:
            obs = obs_by_realization.get(link.realization_id)
            if obs is None:
                continue
            if obs.prospective_identifiability == "NOT_OPERATIONALIZABLE":
                not_operationalizable += 1
            if (
                link.materialization_status == "MATERIALIZED"
                and not (
                    obs.prospective_status == "COMPLETE"
                    and obs.prospective_contract_integrity_passed is True
                    and obs.prospective_identifiability == "NOT_OPERATIONALIZABLE"
                )
            ):
                usable += 1
        share = raw_compute[family_key] / compute_total if compute_total else 0.0
        states.append(
            FamilySearchState(
                family_key=family_key,
                representative_idea_id=representative[family_key],
                member_idea_ids=members,
                idea_count=len(members),
                generation_counts=dict(sorted(generation_counts.items())),
                exact_kernel_unique_count=len(
                    {node_by_id[idea_id].kernel_sha256 for idea_id in members}
                ),
                realization_count=len(links),
                materialized_realization_count=len(materialized),
                usable_grounded_realization_count=usable,
                not_operationalizable_realization_count=not_operationalizable,
                adaptive_local_event_count=events_by_family[family_key],
                offspring_generation_llm_calls=offspring_calls[family_key],
                realization_llm_calls=realization_calls[family_key],
                prospective_audit_llm_calls=prospective_family_calls[family_key],
                total_tracked_llm_calls=raw_compute[family_key],
                compute_share=share,
                productive_offspring_count=productive[family_key],
                family_expansion_event_count=expansion_by_parent_family[family_key],
                productive_operator_counts=dict(sorted(productive_ops[family_key].items())),
            )
        )

    generation_family: dict[str, set[str]] = defaultdict(set)
    births: Counter[str] = Counter()
    seen_families: set[str] = set()
    for node in sorted(nodes, key=lambda row: (row.generation_index, row.idea_id)):
        family_key = assignment_by_id[node.idea_id].family_key
        label = f"G{node.generation_index}"
        generation_family[label].add(family_key)
        if family_key not in seen_families:
            births[label] += 1
            seen_families.add(family_key)

    hhi = sum(row.compute_share ** 2 for row in states)
    largest = max((row.compute_share for row in states), default=0.0)
    provisional = FamilyPopulationReport(
        report_id="pending",
        report_sha256="pending",
        assignments=list(assignments),
        family_states=states,
        transitions=transitions,
        idea_count=len(nodes),
        family_count=len(states),
        generation_family_counts={
            key: len(value) for key, value in sorted(generation_family.items())
        },
        family_birth_count_by_generation=dict(sorted(births.items())),
        family_compute_hhi=hhi,
        largest_family_compute_share=largest,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"family_population:{digest[:20]}",
            "report_sha256": digest,
        }
    )




def enrich_observations_from_residual_reports(
    observations: Sequence[ScientificFeedbackFacetObservation],
    *,
    residual_reports: Sequence[Mapping[str, Any]],
    allowed_source_portfolio_ids: Sequence[str],
    adaptive_events: Sequence[AdaptiveLocalSearchEvent] = (),
) -> tuple[list[ScientificFeedbackFacetObservation], VerificationFacetIntegrationReport]:
    """Attach exact-lineage residual/novelty facets without converting them to policy."""

    allowed = set(_dedupe(allowed_source_portfolio_ids))
    residual_by_hypothesis: dict[str, Mapping[str, Any]] = {}
    accepted_report_count = 0
    for report in residual_reports:
        if str(report.get("schema_version") or "") != "scientific-portfolio-residual-epistemic-state-v1":
            continue
        source_portfolio_id = str(report.get("source_portfolio_id") or "")
        if source_portfolio_id not in allowed:
            continue
        accepted_report_count += 1
        for row in report.get("hypotheses", []):
            if not isinstance(row, Mapping):
                continue
            hid = str(row.get("hypothesis_id") or "")
            if hid:
                residual_by_hypothesis[hid] = row

    out: list[ScientificFeedbackFacetObservation] = []
    matched: set[str] = set()
    materialized_ids = {
        str(row.hypothesis_id)
        for row in observations
        if row.hypothesis_id is not None
    }
    for observation in observations:
        row = residual_by_hypothesis.get(str(observation.hypothesis_id or ""))
        if row is None:
            out.append(observation)
            continue
        matched.add(str(observation.hypothesis_id))
        out.append(
            observation.model_copy(
                update={
                    "residual_epistemic_state": str(row.get("final_epistemic_state") or "") or None,
                    "residual_state_reason": str(row.get("state_reason") or "") or None,
                    "external_novelty_status": str(row.get("fresh_external_status") or "") or None,
                    "source_systems": _dedupe([*observation.source_systems, "SCIENTIFIC_PORTFOLIO_RESIDUAL_EPISTEMIC_STATE"]),
                    "source_versions": _dedupe([*observation.source_versions, "scientific-portfolio-residual-epistemic-state-v1"]),
                }
            )
        )

    provisional = VerificationFacetIntegrationReport(
        report_id="pending",
        report_sha256="pending",
        observation_count=len(out),
        prospective_observation_count=sum(
            "PROSPECTIVE_IDENTIFICATION_MATERIALIZATION_SHADOW" in row.source_systems
            for row in out
        ),
        residual_report_count=accepted_report_count,
        residual_matched_hypothesis_count=len(matched),
        residual_unmatched_hypothesis_ids=sorted(materialized_ids - matched),
        adaptive_local_event_count=len(adaptive_events),
        adaptive_boundary_sensitive_event_count=sum(
            row.interpreted_scope == "BOUNDARY_SENSITIVE" for row in adaptive_events
        ),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return out, provisional.model_copy(
        update={
            "report_id": f"verification_facet_integration:{digest[:20]}",
            "report_sha256": digest,
        }
    )

def build_local_search_states(
    *,
    idea_ids: Sequence[str],
    links: Sequence[IdeaRealizationLink],
    observations: Sequence[ScientificFeedbackFacetObservation],
    adaptive_events: Sequence[AdaptiveLocalSearchEvent],
    local_search_budget: int,
) -> list[IdeaLocalSearchState]:
    links_by_idea: dict[str, list[IdeaRealizationLink]] = defaultdict(list)
    for row in links:
        links_by_idea[row.idea_id].append(row)
    obs_by_realization = {row.realization_id: row for row in observations}
    events_by_idea: dict[str, list[AdaptiveLocalSearchEvent]] = defaultdict(list)
    for row in adaptive_events:
        events_by_idea[row.idea_id].append(row)

    out: list[IdeaLocalSearchState] = []
    for idea_id in _dedupe(idea_ids):
        rows = sorted(links_by_idea.get(idea_id, []), key=lambda row: row.attempt_index)
        usable = 0
        materialized = 0
        not_op = 0
        failed = 0
        for row in rows:
            if row.materialization_status == "MATERIALIZED":
                materialized += 1
            else:
                failed += 1
            obs = obs_by_realization.get(row.realization_id)
            if obs is not None and obs.prospective_identifiability == "NOT_OPERATIONALIZABLE":
                not_op += 1
            if (
                row.materialization_status == "MATERIALIZED"
                and (obs is None or obs.prospective_identifiability != "NOT_OPERATIONALIZABLE")
                and (obs is None or obs.prospective_contract_integrity_passed is not False)
            ):
                usable += 1
        rescued = bool(
            usable
            and rows
            and rows[0].materialization_status != "MATERIALIZED"
        ) or bool(
            usable
            and any(
                obs_by_realization.get(row.realization_id) is not None
                and obs_by_realization[row.realization_id].prospective_identifiability
                == "NOT_OPERATIONALIZABLE"
                for row in rows[:-1]
            )
        )
        used = min(len(rows), local_search_budget)
        exhausted = used >= local_search_budget and usable == 0
        out.append(
            IdeaLocalSearchState(
                idea_id=idea_id,
                realization_ids=[row.realization_id for row in rows],
                realization_count=len(rows),
                materialized_count=materialized,
                usable_grounded_realization_count=usable,
                not_operationalizable_count=not_op,
                failed_or_abstained_count=failed,
                adaptive_event_ids=[row.event_id for row in events_by_idea.get(idea_id, [])],
                local_search_budget=local_search_budget,
                local_search_budget_used=used,
                local_search_exhausted=exhausted,
                rescued_within_same_idea=rescued,
            )
        )
    return out


def build_idea_learning_decisions(
    *,
    local_states: Sequence[IdeaLocalSearchState],
    observations: Sequence[ScientificFeedbackFacetObservation],
    adaptive_events: Sequence[AdaptiveLocalSearchEvent],
    family_report: FamilyPopulationReport,
    overconcentrated_family_share: float = 0.35,
) -> list[IdeaLearningDecision]:
    assignment_by_id = {row.idea_id: row for row in family_report.assignments}
    family_by_key = {row.family_key: row for row in family_report.family_states}
    obs_by_idea: dict[str, list[ScientificFeedbackFacetObservation]] = defaultdict(list)
    for row in observations:
        obs_by_idea[row.idea_id].append(row)
    events_by_idea: dict[str, list[AdaptiveLocalSearchEvent]] = defaultdict(list)
    for row in adaptive_events:
        events_by_idea[row.idea_id].append(row)

    out: list[IdeaLearningDecision] = []
    for state in local_states:
        assignment = assignment_by_id.get(state.idea_id)
        if assignment is None:
            continue
        family = family_by_key[assignment.family_key]
        exploit = "MEDIUM"
        transform = "MEDIUM"
        explore = "MEDIUM"
        channels: list[str] = []
        reasons: list[str] = []

        if state.usable_grounded_realization_count > 0:
            exploit = "HIGH"
            channels.extend(["EXPLOIT", "EXPLORE"])
            reasons.append("USABLE_GROUNDED_REALIZATION_RETAINED")
            disposition: IdeaLearningDisposition = "RETAIN_EXPLOIT"
            if state.rescued_within_same_idea:
                reasons.append("SAME_IDEA_LOCAL_REALIZATION_RESCUE_SUCCEEDED")
        elif state.local_search_exhausted:
            exploit = "LOW"
            transform = "HIGH"
            explore = "HIGH"
            channels.extend(["TRANSFORM", "EXPLORE"])
            reasons.append("LOCAL_REALIZATION_SPACE_EXHAUSTED_ESCALATES_IDEA_SEARCH")
            disposition = "ESCALATE_TRANSFORM"
        elif state.realization_count == 0:
            explore = "HIGH"
            channels.extend(["EXPLORE", "WILDCARD"])
            reasons.append("IDEA_NOT_YET_REALIZED_REMAINS_UNDEREXPLORED")
            disposition = "OBSERVE_UNDEREXPLORED"
        else:
            explore = "HIGH"
            channels.extend(["EXPLORE", "TRANSFORM"])
            reasons.append("LOCAL_SEARCH_STILL_OPEN_NO_IDEA_FAILURE_ASSERTED")
            disposition = "RETAIN_EXPLORE"

        if state.not_operationalizable_count > 0:
            transform = "HIGH"
            channels.insert(0, "TRANSFORM")
            reasons.append("NOT_OPERATIONALIZABLE_REALIZATION_RAISES_TRANSFORM_PRESSURE_ONLY")

        idea_events = events_by_idea.get(state.idea_id, [])
        if any(row.interpreted_scope == "BOUNDARY_SENSITIVE" for row in idea_events):
            transform = "HIGH"
            reasons.append("AXIS_MUTATION_REQUIRES_POSTHOC_IDEA_SEMANTIC_COMPARISON")
            channels.insert(0, "TRANSFORM")
        if any(row.interpreted_scope == "LOCAL_CONTEXT_RESET" for row in idea_events):
            explore = "HIGH"
            reasons.append("GRAPH_RETRAVERSAL_IS_CONTEXT_RESET_NOT_AUTOMATIC_CHILD_IDEA")
            channels.append("EXPLORE")

        if family.compute_share > overconcentrated_family_share:
            explore = "HIGH"
            channels.extend(["EXPLORE", "WILDCARD"])
            reasons.append("OVERCONCENTRATED_FAMILY_ADDS_EXPLORATION_PRESSURE_NOT_HARD_GATE")

        out.append(
            IdeaLearningDecision(
                idea_id=state.idea_id,
                family_key=assignment.family_key,
                disposition=disposition,
                recommended_channels=_dedupe(channels),
                transformation_pressure=transform,
                exploration_pressure=explore,
                exploit_pressure=exploit,
                reason_codes=_dedupe(reasons),
                source_observation_ids=[row.observation_id for row in obs_by_idea.get(state.idea_id, [])],
                source_adaptive_event_ids=[row.event_id for row in idea_events],
                family_compute_share=family.compute_share,
                local_search_exhausted=state.local_search_exhausted,
            )
        )
    return out


def build_credit_continuity_traces(
    *,
    generation2_nodes: Sequence[ResearchIdeaNode],
    generation3_nodes: Sequence[ResearchIdeaNode],
    g2_states: Sequence[IdeaLocalSearchState],
    g3_states: Sequence[IdeaLocalSearchState],
) -> list[CreditContinuityTrace]:
    g2_state = {row.idea_id: row for row in g2_states}
    g3_state = {row.idea_id: row for row in g3_states}
    child_ids_by_parent: dict[str, list[str]] = defaultdict(list)
    for child in generation3_nodes:
        for parent_id in child.parent_idea_ids:
            child_ids_by_parent[parent_id].append(child.idea_id)

    out: list[CreditContinuityTrace] = []
    for node in sorted(generation2_nodes, key=lambda row: row.idea_id):
        state = g2_state.get(node.idea_id)
        children = sorted(child_ids_by_parent.get(node.idea_id, []))
        usable_children = [
            child_id
            for child_id in children
            if g3_state.get(child_id) is not None
            and g3_state[child_id].usable_grounded_realization_count > 0
        ]
        had_usable = bool(state and state.usable_grounded_realization_count > 0)
        rescued_local = bool(state and state.rescued_within_same_idea)
        reasons: list[str] = []
        if rescued_local:
            outcome: CreditContinuityOutcome = "RESCUED_WITHIN_IDEA"
            reasons.append("FAILED_OR_STERILE_REALIZATION_RECOVERED_WITHOUT_CHANGING_IDEA")
        elif not had_usable and usable_children:
            outcome = "RESCUED_BY_CHILD"
            reasons.append("PARENT_REALIZATION_FAILED_BUT_DESCENDANT_IDEA_MATERIALIZED")
        elif had_usable and usable_children:
            outcome = "PRODUCTIVE_LINEAGE"
            reasons.append("PARENT_AND_DESCENDANT_BOTH_HAVE_USABLE_GROUNDED_REALIZATIONS")
        elif state is None or state.realization_count == 0:
            outcome = "NOT_OBSERVED"
            reasons.append("PARENT_REALIZATION_NOT_OBSERVED_UNDER_CURRENT_BUDGET")
        else:
            outcome = "UNRESOLVED"
            reasons.append("NO_USABLE_LOCAL_OR_DESCENDANT_REALIZATION_YET")
        out.append(
            CreditContinuityTrace(
                trace_id=_stable_id("credit_continuity", node.idea_id, children, outcome),
                generation2_idea_id=node.idea_id,
                generation2_realization_ids=(state.realization_ids if state is not None else []),
                generation2_had_usable_realization=had_usable,
                generation2_rescued_within_same_idea=rescued_local,
                generation3_child_idea_ids=children,
                generation3_usable_child_idea_ids=usable_children,
                outcome=outcome,
                reason_codes=reasons,
            )
        )
    return out


def build_semantic_calibration_sample(
    *,
    nodes: Sequence[ResearchIdeaNode],
    audits: Sequence[ConceptualDeltaAuditReport],
    max_items: int = 24,
) -> SemanticCalibrationSample:
    node_by_id = {row.idea_id: row for row in nodes}
    rows = []
    for audit in audits:
        rows.extend(audit.records)
    rows.sort(
        key=lambda row: (
            -int(row.comparator_audit_disagreement),
            0 if row.audit_category in {"SAME_CONCEPT", "REFINEMENT"} else 1,
            -row.audit_confidence,
            row.child_idea_id,
        )
    )
    items: list[SemanticCalibrationSampleItem] = []
    for row in rows[:max_items]:
        child = node_by_id.get(row.child_idea_id)
        if child is None:
            continue
        parents = [node_by_id[pid] for pid in row.parent_idea_ids if pid in node_by_id]
        items.append(
            SemanticCalibrationSampleItem(
                generation_index=row.generation_index,
                child_idea_id=row.child_idea_id,
                parent_idea_ids=list(row.parent_idea_ids),
                deterministic_identity_relation=row.deterministic_identity_relation,
                audit_category=row.audit_category,
                audit_confidence=row.audit_confidence,
                comparator_audit_disagreement=row.comparator_audit_disagreement,
                parent_kernels=[p.kernel.model_dump(mode="json") for p in parents],
                child_kernel=child.kernel.model_dump(mode="json"),
            )
        )
    return SemanticCalibrationSample(
        sample_id=_stable_id(
            "semantic_calibration_sample",
            [(row.child_idea_id, row.audit_category) for row in items],
        ),
        items=items,
    )


__all__ = [
    "AdaptiveLocalSearchEvent",
    "ConceptualFamilyAssignment",
    "ConceptualFamilyTransition",
    "CreditContinuityTrace",
    "FamilyPopulationReport",
    "FamilySearchState",
    "IdeaLearningDecision",
    "IdeaLocalSearchState",
    "IdeaRealizationLink",
    "ScientificFeedbackFacetObservation",
    "SemanticCalibrationSample",
    "VerificationFacetIntegrationReport",
    "adapt_adaptive_controller_plans",
    "assign_conceptual_families",
    "build_credit_continuity_traces",
    "enrich_observations_from_residual_reports",
    "build_family_population_report",
    "build_idea_learning_decisions",
    "build_local_search_states",
    "build_semantic_calibration_sample",
]
