from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.frontier_exploration_audit import FrontierExplorationAudit
from pipeline_core.discovery.frontier_idea_population import (
    FrontierIdea,
    FrontierIdeaPopulation,
)
from pipeline_core.discovery.idea_evolution import IdeaEvolutionIdea, IdeaEvolutionReport


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


AssessmentLevel = Literal["LOW", "MODERATE", "HIGH", "INDETERMINATE"]
PortfolioOrigin = Literal["FRONTIER", "EVOLUTION"]
PortfolioProfile = Literal[
    "TASK_NEAR_VALIDATION",
    "MECHANISM_FOCUSED",
    "DISCRIMINATING_TEST",
    "HIGH_INFORMATION",
    "EXPLORATORY_BRIDGE",
    "REFRAME",
]
VerificationBurden = Literal["LOW", "MODERATE", "HIGH"]


PROFILE_ORDER: tuple[PortfolioProfile, ...] = (
    "TASK_NEAR_VALIDATION",
    "MECHANISM_FOCUSED",
    "DISCRIMINATING_TEST",
    "HIGH_INFORMATION",
    "EXPLORATORY_BRIDGE",
    "REFRAME",
)

_LEVEL_RANK: dict[str, int] = {
    "LOW": 0,
    "INDETERMINATE": 1,
    "MODERATE": 2,
    "HIGH": 3,
}

_PROFILE_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "TASK_NEAR_VALIDATION": (
        "task_relevance",
        "falsifiability",
        "operationalizability",
        "information_gain",
    ),
    "MECHANISM_FOCUSED": (
        "mechanistic_coherence",
        "falsifiability",
        "discriminating_power",
    ),
    "DISCRIMINATING_TEST": (
        "discriminating_power",
        "falsifiability",
        "information_gain",
        "operationalizability",
    ),
    "HIGH_INFORMATION": (
        "information_gain",
        "discriminating_power",
        "falsifiability",
    ),
    "EXPLORATORY_BRIDGE": (
        "task_relevance",
        "mechanistic_coherence",
        "falsifiability",
        "discriminating_power",
        "information_gain",
    ),
    "REFRAME": (
        "task_relevance",
        "discriminating_power",
        "information_gain",
        "falsifiability",
    ),
}


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical_json(parts).encode('utf-8')).hexdigest()[:20]}"


def _rank(value: AssessmentLevel) -> int:
    return _LEVEL_RANK[value]


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(x) for x in values if str(x).strip()))


def _frontier_core_relations(idea: FrontierIdea) -> list[str]:
    if idea.relation_signature is not None:
        row = idea.relation_signature
        return [f"{row.subject} --{row.relation}--> {row.object}"]
    if idea.topology_signature is not None:
        rows = list(idea.topology_signature.backbone_relation_texts)
        modifier = str(idea.topology_signature.modifier_text or "").strip()
        if modifier:
            rows.append(f"modifier: {modifier}")
        return rows
    if idea.tension_signature is not None:
        return list(idea.tension_signature.basis_relation_texts)
    if idea.competing_explanation_signature is not None:
        return list(idea.competing_explanation_signature.basis_relation_texts)
    return []


def _frontier_family_signature(
    idea: FrontierIdea,
    backbone_family_by_idea: Mapping[str, str],
) -> str:
    if idea.idea_form == "HIGHER_ORDER_TOPOLOGY":
        value = backbone_family_by_idea.get(idea.idea_id)
        if value:
            return value
    return idea.exact_scientific_signature


def _frontier_candidate(
    idea: FrontierIdea,
    *,
    backbone_family_by_idea: Mapping[str, str],
) -> "ScientificPortfolioCandidate":
    external = any(x.external_literature_lineage for x in idea.source_lineage)
    unverified = any(x.candidate_or_unverified_lineage for x in idea.source_lineage)
    source_kinds = sorted({x.source_kind for x in idea.source_lineage})
    return ScientificPortfolioCandidate(
        candidate_id=_stable_id("scientific_portfolio_candidate", "FRONTIER", idea.idea_id),
        origin="FRONTIER",
        source_object_id=idea.idea_id,
        source_kind=idea.source_kind,
        operator_id=None,
        idea_form=idea.idea_form,
        title=idea.rendered_scientific_intent,
        scientific_intent=idea.rendered_scientific_intent,
        conceptual_change_summary="",
        core_relations=_frontier_core_relations(idea),
        differential_prediction="",
        falsification_condition="",
        discriminating_observation="",
        task_relation_mode=idea.task_relation_mode,
        conceptual_family_signature=_frontier_family_signature(
            idea,
            backbone_family_by_idea,
        ),
        parent_source_kinds=source_kinds,
        external_literature_lineage=external,
        candidate_or_unverified_lineage=unverified,
        cross_source_composition=False,
        source_artifact_refs=_unique(x.source_artifact for x in idea.source_lineage),
    )


def _evolution_candidate(idea: IdeaEvolutionIdea) -> "ScientificPortfolioCandidate":
    return ScientificPortfolioCandidate(
        candidate_id=_stable_id("scientific_portfolio_candidate", "EVOLUTION", idea.evolution_id),
        origin="EVOLUTION",
        source_object_id=idea.evolution_id,
        source_kind=None,
        operator_id=idea.operator_id,
        idea_form=idea.idea_form,
        title=idea.title,
        scientific_intent=idea.scientific_intent,
        conceptual_change_summary=idea.conceptual_change_summary,
        core_relations=list(idea.core_relations),
        differential_prediction=idea.differential_prediction,
        falsification_condition=idea.falsification_condition,
        discriminating_observation=idea.discriminating_observation,
        task_relation_mode=idea.task_relation_mode,
        conceptual_family_signature=idea.conceptual_family_signature,
        parent_source_kinds=list(idea.parent_source_kinds),
        external_literature_lineage=idea.inherited_external_literature_lineage,
        candidate_or_unverified_lineage=idea.inherited_candidate_or_unverified_lineage,
        cross_source_composition=idea.cross_source_composition,
        source_artifact_refs=_unique(x.source_artifact for x in idea.lineage_refs),
    )


class ScientificPortfolioCandidate(StrictModel):
    schema_version: Literal["scientific-portfolio-candidate-v1"] = (
        "scientific-portfolio-candidate-v1"
    )
    candidate_id: str
    origin: PortfolioOrigin
    source_object_id: str
    source_kind: str | None = None
    operator_id: str | None = None
    idea_form: str
    title: str
    scientific_intent: str
    conceptual_change_summary: str = ""
    core_relations: list[str] = Field(default_factory=list)
    differential_prediction: str = ""
    falsification_condition: str = ""
    discriminating_observation: str = ""
    task_relation_mode: str
    conceptual_family_signature: str
    parent_source_kinds: list[str] = Field(default_factory=list)
    external_literature_lineage: bool = False
    candidate_or_unverified_lineage: bool = False
    cross_source_composition: bool = False
    source_artifact_refs: list[str] = Field(default_factory=list)

    requires_verification: Literal[True] = True
    epistemic_status: Literal["INSPIRATION_ONLY"] = "INSPIRATION_ONLY"
    truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class ScientificPortfolioCandidatePool(StrictModel):
    schema_version: Literal["scientific-portfolio-candidate-pool-v1"] = (
        "scientific-portfolio-candidate-pool-v1"
    )
    pool_id: str
    pool_sha256: str
    source_population_id: str
    source_population_sha256: str
    source_evolution_report_id: str
    source_evolution_report_sha256: str
    source_context_id: str
    source_context_sha256: str
    research_question: str

    raw_frontier_idea_count: int = Field(ge=0)
    raw_evolution_idea_count: int = Field(ge=0)
    projected_candidate_count: int = Field(ge=0)
    candidate_count_by_origin: dict[str, int] = Field(default_factory=dict)
    candidate_count_by_form: dict[str, int] = Field(default_factory=dict)
    projected_topology_backbone_representative_count: int = Field(ge=0)
    projected_candidate_topology_supplement_count: int = Field(ge=0)
    raw_topology_variant_count_omitted_from_evaluation: int = Field(ge=0)
    evaluation_budget_cap: int = Field(ge=1)
    evaluation_budget_cap_reached: bool
    candidates: list[ScientificPortfolioCandidate] = Field(default_factory=list)

    structural_projection_only: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    candidate_deletion_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class ScientificPortfolioDimensionAssessment(StrictModel):
    level: AssessmentLevel
    rationale: str = Field(min_length=1)


class ScientificPortfolioNormalizedSketch(StrictModel):
    """Origin-blind common representation used only for portfolio evaluation.

    This is a hypothetical scientific sketch derived from the supplied idea
    content. It does not establish truth, novelty, premise eligibility, or
    production authority.
    """

    hypothesis_frame: str = Field(min_length=1)
    predicted_observation: str = Field(min_length=1)
    falsification_condition: str = Field(min_length=1)
    discriminating_observation: str = Field(min_length=1)


class ScientificPortfolioCandidateEvaluationDraft(StrictModel):
    candidate_id: str
    normalized_sketch: ScientificPortfolioNormalizedSketch
    task_relevance: ScientificPortfolioDimensionAssessment
    mechanistic_coherence: ScientificPortfolioDimensionAssessment
    falsifiability: ScientificPortfolioDimensionAssessment
    discriminating_power: ScientificPortfolioDimensionAssessment
    operationalizability: ScientificPortfolioDimensionAssessment
    information_gain: ScientificPortfolioDimensionAssessment
    overall_rationale: str = Field(min_length=1)


class ScientificPortfolioEvaluationBatchDraft(StrictModel):
    schema_version: Literal["scientific-portfolio-evaluation-batch-draft-v1"] = (
        "scientific-portfolio-evaluation-batch-draft-v1"
    )
    evaluations: list[ScientificPortfolioCandidateEvaluationDraft] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def _ids_unique(self) -> "ScientificPortfolioEvaluationBatchDraft":
        ids = [x.candidate_id for x in self.evaluations]
        if len(ids) != len(set(ids)):
            raise ValueError("evaluation candidate_id values must be unique")
        return self


class ScientificPortfolioCandidateEvaluation(StrictModel):
    candidate_id: str
    normalized_sketch: ScientificPortfolioNormalizedSketch
    task_relevance: ScientificPortfolioDimensionAssessment
    mechanistic_coherence: ScientificPortfolioDimensionAssessment
    falsifiability: ScientificPortfolioDimensionAssessment
    discriminating_power: ScientificPortfolioDimensionAssessment
    operationalizability: ScientificPortfolioDimensionAssessment
    information_gain: ScientificPortfolioDimensionAssessment
    overall_rationale: str
    verification_burden: VerificationBurden
    eligible_profiles: list[PortfolioProfile] = Field(default_factory=list)

    truth_evaluated: Literal[False] = False
    literature_novelty_evaluated: Literal[False] = False
    feasibility_certified: Literal[False] = False
    diagnostic_only: Literal[True] = True


class ScientificPortfolioEvaluationReport(StrictModel):
    schema_version: Literal["scientific-portfolio-evaluation-report-v1"] = (
        "scientific-portfolio-evaluation-report-v1"
    )
    report_id: str
    report_sha256: str
    source_pool_id: str
    source_pool_sha256: str
    evaluator_model: str
    evaluations: list[ScientificPortfolioCandidateEvaluation]
    evaluation_count: int = Field(ge=0)
    eligible_profile_counts: dict[str, int] = Field(default_factory=dict)
    verification_burden_counts: dict[str, int] = Field(default_factory=dict)
    level_counts_by_dimension: dict[str, dict[str, int]] = Field(default_factory=dict)

    evidence_scope: Literal[
        "supplied_idea_artifacts_only_no_external_truth_or_novelty_assessment"
    ] = "supplied_idea_artifacts_only_no_external_truth_or_novelty_assessment"
    profile_dimension_evaluation_performed: Literal[True] = True
    normalized_scientific_sketch_performed: Literal[True] = True
    selection_view_origin_blind: Literal[True] = True
    source_artifact_richness_not_used_as_profile_eligibility: Literal[True] = True
    global_scientific_quality_ranking_performed: Literal[False] = False
    overall_winner_selected: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class ScientificPortfolioSelectionEntry(StrictModel):
    candidate_id: str
    source_object_id: str
    origin: PortfolioOrigin
    assigned_profile: PortfolioProfile
    eligible_profiles: list[PortfolioProfile]
    conceptual_family_signature: str
    pareto_layer: int = Field(ge=1)
    verification_burden: VerificationBurden
    selection_reason_codes: list[str] = Field(default_factory=list)


class ScientificPortfolioSelectionReport(StrictModel):
    schema_version: Literal["scientific-portfolio-selection-shadow-v1"] = (
        "scientific-portfolio-selection-shadow-v1"
    )
    selection_id: str
    selection_sha256: str
    source_pool_id: str
    source_pool_sha256: str
    source_evaluation_report_id: str
    source_evaluation_report_sha256: str
    max_retained_candidates: int = Field(ge=1)
    max_retained_per_profile: int = Field(ge=1)
    retained_count: int = Field(ge=0)
    retained_unique_family_count: int = Field(ge=0)
    retained_count_by_profile: dict[str, int] = Field(default_factory=dict)
    retained_count_by_origin: dict[str, int] = Field(default_factory=dict)
    retained_candidate_ids: list[str] = Field(default_factory=list)
    entries: list[ScientificPortfolioSelectionEntry] = Field(default_factory=list)
    not_retained_candidate_ids: list[str] = Field(default_factory=list)

    selection_policy: Literal[
        "PROFILE_PARETO_DIVERSITY_RETENTION_V2_NORMALIZED_BLIND"
    ] = "PROFILE_PARETO_DIVERSITY_RETENTION_V2_NORMALIZED_BLIND"
    single_scalar_score_used: Literal[False] = False
    overall_winner_selected: Literal[False] = False
    family_diversity_enforced: Literal[True] = True
    scientific_quality_certification_performed: Literal[False] = False
    candidate_deletion_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    stage8_input_changed: Literal[False] = False


class ScientificPortfolioEvaluationPromptData(StrictModel):
    research_question: str
    task_source: str
    task_target: str
    candidates: list[dict[str, Any]]


def build_scientific_portfolio_candidate_pool(
    *,
    population: FrontierIdeaPopulation,
    exploration_audit: FrontierExplorationAudit,
    evolution_report: IdeaEvolutionReport,
    max_evaluation_candidates: int = 48,
    max_candidate_topology_supplements: int = 6,
) -> ScientificPortfolioCandidatePool:
    if max_evaluation_candidates < 1:
        raise ValueError("max_evaluation_candidates must be >= 1")
    if max_candidate_topology_supplements < 0:
        raise ValueError("max_candidate_topology_supplements must be >= 0")
    if exploration_audit.population_id != population.population_id:
        raise ValueError("Frontier audit/population lineage mismatch")
    if evolution_report.source_population_id != population.population_id:
        raise ValueError("Idea Evolution/Frontier population lineage mismatch")
    if evolution_report.source_context_id != population.source_context_id:
        raise ValueError("Idea Evolution/Frontier source context mismatch")

    frontier_by_id = {x.idea_id: x for x in population.ideas}
    backbone_family_by_idea: dict[str, str] = {}
    topology_rep_ids: list[str] = []
    for family in exploration_audit.topology_layer.backbone_families:
        family_ids = [
            idea_id
            for idea_id in family.topology_idea_ids
            if idea_id in frontier_by_id
        ]
        for idea_id in family_ids:
            backbone_family_by_idea[idea_id] = family.family_id

        if not family_ids:
            continue

        # Keep structural backbone representation and exploratory
        # candidate lineage as separate projection roles whenever
        # the family contains both.
        representative_id = next(
            (
                idea_id
                for idea_id in family_ids
                if not any(
                    lineage.candidate_or_unverified_lineage
                    for lineage in frontier_by_id[idea_id].source_lineage
                )
            ),
            family_ids[0],
        )
        topology_rep_ids.append(representative_id)

    selected_frontier_ids: list[str] = []
    for idea in population.ideas:
        if idea.idea_form != "HIGHER_ORDER_TOPOLOGY":
            selected_frontier_ids.append(idea.idea_id)
    selected_frontier_ids.extend(topology_rep_ids)

    candidate_topology_supplements = []
    for idea in population.ideas:
        if idea.idea_form != "HIGHER_ORDER_TOPOLOGY":
            continue
        if idea.idea_id in selected_frontier_ids:
            continue
        if any(x.candidate_or_unverified_lineage for x in idea.source_lineage):
            candidate_topology_supplements.append(idea.idea_id)
    candidate_topology_supplements = candidate_topology_supplements[
        :max_candidate_topology_supplements
    ]
    selected_frontier_ids.extend(candidate_topology_supplements)
    selected_frontier_ids = _unique(selected_frontier_ids)

    projected: list[ScientificPortfolioCandidate] = []
    for idea_id in selected_frontier_ids:
        idea = frontier_by_id.get(idea_id)
        if idea is None:
            continue
        projected.append(
            _frontier_candidate(
                idea,
                backbone_family_by_idea=backbone_family_by_idea,
            )
        )
    projected.extend(_evolution_candidate(x) for x in evolution_report.ideas)

    # The structural projection normally produces a compact pool. If a domain
    # still exceeds the explicit evaluation budget, apply a deterministic
    # round-robin over origin/form groups. This is a budget projection only;
    # it carries no scientific rejection authority.
    cap_reached = len(projected) > max_evaluation_candidates
    if cap_reached:
        groups: dict[tuple[str, str], list[ScientificPortfolioCandidate]] = defaultdict(list)
        for row in projected:
            groups[(row.origin, row.idea_form)].append(row)
        for values in groups.values():
            values.sort(key=lambda x: x.candidate_id)
        keys = sorted(groups)
        bounded: list[ScientificPortfolioCandidate] = []
        while len(bounded) < max_evaluation_candidates:
            added = False
            for key in keys:
                values = groups[key]
                if values and len(bounded) < max_evaluation_candidates:
                    bounded.append(values.pop(0))
                    added = True
            if not added:
                break
        projected = bounded

    raw_topology_count = sum(
        x.idea_form == "HIGHER_ORDER_TOPOLOGY" for x in population.ideas
    )
    projected_frontier_topology_ids = {
        row.source_object_id
        for row in projected
        if row.origin == "FRONTIER" and row.idea_form == "HIGHER_ORDER_TOPOLOGY"
    }
    omitted_topology_variants = max(0, raw_topology_count - len(projected_frontier_topology_ids))

    provisional = ScientificPortfolioCandidatePool(
        pool_id="pending",
        pool_sha256="pending",
        source_population_id=population.population_id,
        source_population_sha256=population.population_sha256,
        source_evolution_report_id=evolution_report.report_id,
        source_evolution_report_sha256=evolution_report.report_sha256,
        source_context_id=population.source_context_id,
        source_context_sha256=population.source_context_sha256,
        research_question=population.research_question,
        raw_frontier_idea_count=population.total_idea_count,
        raw_evolution_idea_count=evolution_report.idea_count,
        projected_candidate_count=len(projected),
        candidate_count_by_origin=dict(sorted(Counter(x.origin for x in projected).items())),
        candidate_count_by_form=dict(sorted(Counter(x.idea_form for x in projected).items())),
        projected_topology_backbone_representative_count=len(topology_rep_ids),
        projected_candidate_topology_supplement_count=len(candidate_topology_supplements),
        raw_topology_variant_count_omitted_from_evaluation=omitted_topology_variants,
        evaluation_budget_cap=max_evaluation_candidates,
        evaluation_budget_cap_reached=cap_reached,
        candidates=projected,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("pool_id", None)
    payload.pop("pool_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "pool_id": f"scientific_portfolio_pool:{digest[:20]}",
            "pool_sha256": digest,
        }
    )


def _verification_burden(candidate: ScientificPortfolioCandidate) -> VerificationBurden:
    burden = 0
    if candidate.external_literature_lineage:
        burden += 1
    if candidate.candidate_or_unverified_lineage:
        burden += 1
    if candidate.task_relation_mode in {"UNKNOWN", "REFRAME"}:
        burden += 1
    if candidate.origin == "EVOLUTION":
        burden += 1
    if burden >= 3:
        return "HIGH"
    if burden >= 1:
        return "MODERATE"
    return "LOW"


def eligible_profiles(
    candidate: ScientificPortfolioCandidate,
    draft: ScientificPortfolioCandidateEvaluationDraft,
) -> list[PortfolioProfile]:
    r = lambda name: _rank(getattr(draft, name).level)
    profiles: list[PortfolioProfile] = []
    if r("task_relevance") >= 2 and r("falsifiability") >= 2 and r("operationalizability") >= 2:
        profiles.append("TASK_NEAR_VALIDATION")
    if r("mechanistic_coherence") >= 3 and r("falsifiability") >= 2:
        profiles.append("MECHANISM_FOCUSED")
    if r("discriminating_power") >= 3 and r("falsifiability") >= 2:
        profiles.append("DISCRIMINATING_TEST")
    if r("information_gain") >= 3 and r("discriminating_power") >= 2:
        profiles.append("HIGH_INFORMATION")
    if (
        candidate.cross_source_composition
        or candidate.external_literature_lineage
        or candidate.candidate_or_unverified_lineage
    ) and r("task_relevance") >= 1 and r("falsifiability") >= 1:
        profiles.append("EXPLORATORY_BRIDGE")
    if candidate.task_relation_mode == "REFRAME" and r("task_relevance") >= 1:
        profiles.append("REFRAME")
    return profiles


def compile_scientific_portfolio_evaluation(
    *,
    pool: ScientificPortfolioCandidatePool,
    draft: ScientificPortfolioEvaluationBatchDraft,
    evaluator_model: str,
) -> ScientificPortfolioEvaluationReport:
    expected = [x.candidate_id for x in pool.candidates]
    actual = [x.candidate_id for x in draft.evaluations]
    if set(actual) != set(expected) or len(actual) != len(expected):
        raise ValueError(
            "evaluation must contain exactly one row for every projected candidate"
        )
    by_id = {x.candidate_id: x for x in pool.candidates}
    eval_by_id = {x.candidate_id: x for x in draft.evaluations}
    compiled: list[ScientificPortfolioCandidateEvaluation] = []
    for candidate_id in expected:
        row = eval_by_id[candidate_id]
        candidate = by_id[candidate_id]
        compiled.append(
            ScientificPortfolioCandidateEvaluation(
                **row.model_dump(mode="json"),
                verification_burden=_verification_burden(candidate),
                eligible_profiles=eligible_profiles(candidate, row),
            )
        )

    profile_counts = Counter(
        profile for row in compiled for profile in row.eligible_profiles
    )
    burden_counts = Counter(row.verification_burden for row in compiled)
    dimensions = (
        "task_relevance",
        "mechanistic_coherence",
        "falsifiability",
        "discriminating_power",
        "operationalizability",
        "information_gain",
    )
    level_counts: dict[str, dict[str, int]] = {}
    for dim in dimensions:
        level_counts[dim] = dict(
            sorted(Counter(getattr(row, dim).level for row in compiled).items())
        )

    provisional = ScientificPortfolioEvaluationReport(
        report_id="pending",
        report_sha256="pending",
        source_pool_id=pool.pool_id,
        source_pool_sha256=pool.pool_sha256,
        evaluator_model=evaluator_model,
        evaluations=compiled,
        evaluation_count=len(compiled),
        eligible_profile_counts=dict(sorted(profile_counts.items())),
        verification_burden_counts=dict(sorted(burden_counts.items())),
        level_counts_by_dimension=level_counts,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"scientific_portfolio_evaluation:{digest[:20]}",
            "report_sha256": digest,
        }
    )


def _profile_vector(
    row: ScientificPortfolioCandidateEvaluation,
    profile: PortfolioProfile,
) -> tuple[int, ...]:
    return tuple(_rank(getattr(row, dim).level) for dim in _PROFILE_DIMENSIONS[profile])


def _dominates(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def _pareto_layers(
    rows: Sequence[ScientificPortfolioCandidateEvaluation],
    profile: PortfolioProfile,
) -> dict[str, int]:
    remaining = {x.candidate_id: x for x in rows}
    layers: dict[str, int] = {}
    layer = 1
    while remaining:
        current_ids = []
        for candidate_id, row in remaining.items():
            vec = _profile_vector(row, profile)
            if not any(
                other_id != candidate_id
                and _dominates(_profile_vector(other, profile), vec)
                for other_id, other in remaining.items()
            ):
                current_ids.append(candidate_id)
        if not current_ids:
            current_ids = sorted(remaining)
        for candidate_id in current_ids:
            layers[candidate_id] = layer
            remaining.pop(candidate_id, None)
        layer += 1
    return layers


def build_scientific_portfolio_selection(
    *,
    pool: ScientificPortfolioCandidatePool,
    evaluation: ScientificPortfolioEvaluationReport,
    max_retained_candidates: int = 8,
    max_retained_per_profile: int = 2,
) -> ScientificPortfolioSelectionReport:
    if max_retained_candidates < 1 or max_retained_per_profile < 1:
        raise ValueError("retention limits must be >= 1")
    if evaluation.source_pool_id != pool.pool_id:
        raise ValueError("evaluation/pool lineage mismatch")
    if evaluation.source_pool_sha256 != pool.pool_sha256:
        raise ValueError("evaluation/pool SHA mismatch")

    candidate_by_id = {x.candidate_id: x for x in pool.candidates}
    eval_by_id = {x.candidate_id: x for x in evaluation.evaluations}
    selected: list[ScientificPortfolioSelectionEntry] = []
    selected_ids: set[str] = set()
    selected_families: set[str] = set()
    selected_profile_counts: Counter[str] = Counter()

    per_profile_order: dict[str, list[tuple[ScientificPortfolioCandidateEvaluation, int]]] = {}
    for profile in PROFILE_ORDER:
        eligible = [x for x in evaluation.evaluations if profile in x.eligible_profiles]
        layers = _pareto_layers(eligible, profile) if eligible else {}
        eligible.sort(
            key=lambda x: (
                layers.get(x.candidate_id, 999),
                tuple(-v for v in _profile_vector(x, profile)),
                x.candidate_id,
            )
        )
        per_profile_order[profile] = [(x, layers[x.candidate_id]) for x in eligible]

    cursor = {profile: 0 for profile in PROFILE_ORDER}
    made_progress = True
    while len(selected) < max_retained_candidates and made_progress:
        made_progress = False
        for profile in PROFILE_ORDER:
            if selected_profile_counts[profile] >= max_retained_per_profile:
                continue
            rows = per_profile_order[profile]
            while cursor[profile] < len(rows):
                row, layer = rows[cursor[profile]]
                cursor[profile] += 1
                candidate = candidate_by_id[row.candidate_id]
                if row.candidate_id in selected_ids:
                    continue
                if candidate.conceptual_family_signature in selected_families:
                    continue
                selected.append(
                    ScientificPortfolioSelectionEntry(
                        candidate_id=row.candidate_id,
                        source_object_id=candidate.source_object_id,
                        origin=candidate.origin,
                        assigned_profile=profile,
                        eligible_profiles=list(row.eligible_profiles),
                        conceptual_family_signature=candidate.conceptual_family_signature,
                        pareto_layer=layer,
                        verification_burden=row.verification_burden,
                        selection_reason_codes=[
                            "PROFILE_ELIGIBLE",
                            "PARETO_LAYER_WITHIN_PROFILE",
                            "CONCEPTUAL_FAMILY_NOT_PREVIOUSLY_RETAINED",
                        ],
                    )
                )
                selected_ids.add(row.candidate_id)
                selected_families.add(candidate.conceptual_family_signature)
                selected_profile_counts[profile] += 1
                made_progress = True
                break
            if len(selected) >= max_retained_candidates:
                break

    retained_ids = [x.candidate_id for x in selected]
    provisional = ScientificPortfolioSelectionReport(
        selection_id="pending",
        selection_sha256="pending",
        source_pool_id=pool.pool_id,
        source_pool_sha256=pool.pool_sha256,
        source_evaluation_report_id=evaluation.report_id,
        source_evaluation_report_sha256=evaluation.report_sha256,
        max_retained_candidates=max_retained_candidates,
        max_retained_per_profile=max_retained_per_profile,
        retained_count=len(selected),
        retained_unique_family_count=len(selected_families),
        retained_count_by_profile=dict(sorted(Counter(x.assigned_profile for x in selected).items())),
        retained_count_by_origin=dict(
            sorted(Counter(candidate_by_id[x.candidate_id].origin for x in selected).items())
        ),
        retained_candidate_ids=retained_ids,
        entries=selected,
        not_retained_candidate_ids=[
            x.candidate_id for x in pool.candidates if x.candidate_id not in selected_ids
        ],
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("selection_id", None)
    payload.pop("selection_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "selection_id": f"scientific_portfolio_selection:{digest[:20]}",
            "selection_sha256": digest,
        }
    )


def evaluation_prompt_payload(
    *,
    pool: ScientificPortfolioCandidatePool,
    task_source: str,
    task_target: str,
) -> ScientificPortfolioEvaluationPromptData:
    rows = []
    for candidate in pool.candidates:
        # Evaluation intentionally receives an origin-blind common semantic
        # view. Lineage/provenance remains in the pool for deterministic
        # verification-burden accounting after evaluation, but it is not shown
        # to the profile evaluator and cannot directly affect scientific scores.
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "scientific_intent": candidate.scientific_intent,
                "conceptual_change_summary": candidate.conceptual_change_summary,
                "core_relations": candidate.core_relations,
            }
        )
    return ScientificPortfolioEvaluationPromptData(
        research_question=pool.research_question,
        task_source=task_source,
        task_target=task_target,
        candidates=rows,
    )


__all__ = [
    "AssessmentLevel",
    "PortfolioProfile",
    "ScientificPortfolioCandidate",
    "ScientificPortfolioCandidatePool",
    "ScientificPortfolioDimensionAssessment",
    "ScientificPortfolioNormalizedSketch",
    "ScientificPortfolioCandidateEvaluationDraft",
    "ScientificPortfolioEvaluationBatchDraft",
    "ScientificPortfolioCandidateEvaluation",
    "ScientificPortfolioEvaluationReport",
    "ScientificPortfolioSelectionEntry",
    "ScientificPortfolioSelectionReport",
    "build_scientific_portfolio_candidate_pool",
    "compile_scientific_portfolio_evaluation",
    "build_scientific_portfolio_selection",
    "evaluation_prompt_payload",
]
