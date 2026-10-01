from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.frontier_exploration_audit import FrontierExplorationAudit
from pipeline_core.discovery.idea_evolution import IdeaEvolutionReport


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IdeaEvolutionSourcePairTransition(StrictModel):
    source_kind_a: str
    source_kind_b: str
    evolution_idea_count: int = Field(ge=1)
    operator_counts: dict[str, int] = Field(default_factory=dict)


class IdeaEvolutionAudit(StrictModel):
    schema_version: Literal["idea-evolution-audit-v1"] = "idea-evolution-audit-v1"

    audit_id: str
    audit_sha256: str
    source_report_id: str
    source_report_sha256: str
    source_population_id: str
    source_exploration_audit_id: str
    source_context_id: str
    source_context_sha256: str
    research_question: str

    baseline_raw_idea_count: int = Field(ge=0)
    baseline_unique_primitive_family_count: int = Field(ge=0)
    baseline_unique_backbone_family_count: int = Field(ge=0)
    baseline_unique_modifier_family_count: int = Field(ge=0)
    baseline_open_world_uncomposed_primitive_family_count: int = Field(ge=0)
    baseline_subordinate_fraction: float = Field(ge=0.0, le=1.0)

    evolution_idea_count: int = Field(ge=0)
    unique_evolution_family_count: int = Field(ge=0)
    duplicate_evolution_family_count: int = Field(ge=0)
    idea_count_by_operator: dict[str, int] = Field(default_factory=dict)
    task_relation_mode_counts: dict[str, int] = Field(default_factory=dict)

    cross_source_bridge_count: int = Field(ge=0)
    cross_source_bridge_family_count: int = Field(ge=0)
    open_world_composed_idea_count: int = Field(ge=0)
    open_world_parent_coverage_count: int = Field(ge=0)
    open_world_parent_coverage_fraction: float = Field(ge=0.0, le=1.0)

    mutated_topology_count: int = Field(ge=0)
    unique_mutated_backbone_family_count: int = Field(ge=0)
    new_mutated_backbone_family_count: int = Field(ge=0)
    mutation_kind_counts: dict[str, int] = Field(default_factory=dict)

    candidate_interpretation_count: int = Field(ge=0)
    candidate_parent_coverage_count: int = Field(ge=0)

    reframe_idea_count: int = Field(ge=0)
    reframe_family_count: int = Field(ge=0)
    reframe_count_by_operator: dict[str, int] = Field(default_factory=dict)

    source_pair_transitions: list[IdeaEvolutionSourcePairTransition] = Field(
        default_factory=list
    )

    cross_source_transition_observed: bool
    new_backbone_transition_observed: bool
    candidate_interpretation_observed: bool
    reframe_observed: bool
    conceptual_transition_observed: bool

    native_llm_calls_attempted: int = Field(ge=0)
    native_llm_calls_succeeded: int = Field(ge=0)

    diagnostic_only: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    candidate_deletion_authority: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    generation_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False
    stage8_input_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class IdeaEvolutionCohortCase(StrictModel):
    case_id: str
    audit_id: str
    source_report_id: str
    research_question: str
    evolution_idea_count: int = Field(ge=0)
    unique_evolution_family_count: int = Field(ge=0)
    cross_source_bridge_count: int = Field(ge=0)
    open_world_parent_coverage_fraction: float = Field(ge=0.0, le=1.0)
    new_mutated_backbone_family_count: int = Field(ge=0)
    candidate_interpretation_count: int = Field(ge=0)
    reframe_idea_count: int = Field(ge=0)
    conceptual_transition_observed: bool


class IdeaEvolutionCohortAudit(StrictModel):
    schema_version: Literal["idea-evolution-cohort-audit-v1"] = (
        "idea-evolution-cohort-audit-v1"
    )

    cohort_id: str
    cohort_sha256: str
    case_count: int = Field(ge=1)
    cases: list[IdeaEvolutionCohortCase] = Field(min_length=1)

    total_evolution_idea_count: int = Field(ge=0)
    case_count_with_cross_source_transition: int = Field(ge=0)
    case_count_with_new_backbone_transition: int = Field(ge=0)
    case_count_with_candidate_interpretation: int = Field(ge=0)
    case_count_with_reframe: int = Field(ge=0)
    case_count_with_any_conceptual_transition: int = Field(ge=0)

    median_evolution_idea_count: float = Field(ge=0.0)
    median_unique_evolution_family_count: float = Field(ge=0.0)
    median_open_world_parent_coverage_fraction: float = Field(ge=0.0, le=1.0)
    median_new_mutated_backbone_family_count: float = Field(ge=0.0)

    diagnostic_only: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    cross_case_winner_selected: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    generation_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False


_TOKEN_CLEAN_RE = re.compile(r"[^a-z0-9α-ω가-힣]+")


def _norm(value: object) -> str:
    text = str(value or "").casefold()
    text = _TOKEN_CLEAN_RE.sub(" ", text)
    return " ".join(text.split())


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


def _backbone_signature(relations: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(_norm(row) for row in relations if _norm(row)))


def build_idea_evolution_audit(
    *,
    exploration_audit: FrontierExplorationAudit,
    evolution_report: IdeaEvolutionReport,
) -> IdeaEvolutionAudit:
    if evolution_report.source_population_id != exploration_audit.population_id:
        raise ValueError("Idea Evolution report population lineage mismatch")
    if evolution_report.source_exploration_audit_id != exploration_audit.audit_id:
        raise ValueError("Idea Evolution report exploration-audit lineage mismatch")
    if evolution_report.source_context_id != exploration_audit.source_context_id:
        raise ValueError("Idea Evolution source context mismatch")
    if evolution_report.source_context_sha256 != exploration_audit.source_context_sha256:
        raise ValueError("Idea Evolution source context SHA mismatch")

    ideas = list(evolution_report.ideas)
    family_counts = Counter(row.conceptual_family_signature for row in ideas)

    bridge = [row for row in ideas if row.operator_id == "CROSS_SOURCE_BRIDGE"]
    mutations = [row for row in ideas if row.operator_id == "BACKBONE_MUTATION"]
    candidate_rows = [
        row for row in ideas if row.operator_id == "CANDIDATE_INTERPRETATION"
    ]
    reframes = [row for row in ideas if row.task_relation_mode == "REFRAME"]

    open_world_pool_count = int(
        exploration_audit.primitive_layer.unique_primitive_family_count_by_source_kind.get(
            "OPEN_WORLD_AXIS", 0
        )
    )
    open_world_parent_ids = {
        lineage.source_object_id
        for row in ideas
        for lineage in row.lineage_refs
        if lineage.lineage_kind == "FRONTIER_IDEA"
        and lineage.source_kind == "OPEN_WORLD_AXIS"
    }
    open_world_parent_coverage = min(
        open_world_pool_count,
        len(open_world_parent_ids),
    )
    open_world_fraction = (
        open_world_parent_coverage / open_world_pool_count
        if open_world_pool_count
        else 0.0
    )

    baseline_backbones = {
        _backbone_signature(row.backbone_relation_texts)
        for row in exploration_audit.topology_layer.backbone_families
    }
    mutation_signatures = {
        _backbone_signature(row.core_relations)
        for row in mutations
        if row.core_relations
    }
    new_mutation_signatures = {
        sig for sig in mutation_signatures if sig and sig not in baseline_backbones
    }

    candidate_parent_ids = {
        parent_id
        for row in candidate_rows
        for parent_id in row.parent_idea_ids
    }

    source_pair_counter: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in ideas:
        kinds = sorted(set(row.parent_source_kinds))
        for left, right in combinations(kinds, 2):
            source_pair_counter[(left, right)][row.operator_id] += 1
    pair_rows = [
        IdeaEvolutionSourcePairTransition(
            source_kind_a=left,
            source_kind_b=right,
            evolution_idea_count=sum(counter.values()),
            operator_counts=dict(sorted(counter.items())),
        )
        for (left, right), counter in sorted(source_pair_counter.items())
    ]

    reframe_counts = Counter(row.operator_id for row in reframes)
    mutation_kind_counts = Counter(
        row.mutation_kind for row in mutations if row.mutation_kind is not None
    )

    cross_observed = bool(bridge)
    backbone_observed = bool(new_mutation_signatures)
    candidate_observed = bool(candidate_rows)
    reframe_observed = bool(reframes)

    provisional = IdeaEvolutionAudit(
        audit_id="pending",
        audit_sha256="pending",
        source_report_id=evolution_report.report_id,
        source_report_sha256=evolution_report.report_sha256,
        source_population_id=evolution_report.source_population_id,
        source_exploration_audit_id=exploration_audit.audit_id,
        source_context_id=evolution_report.source_context_id,
        source_context_sha256=evolution_report.source_context_sha256,
        research_question=evolution_report.research_question,
        baseline_raw_idea_count=exploration_audit.raw_total_idea_count,
        baseline_unique_primitive_family_count=(
            exploration_audit.primitive_layer.unique_primitive_family_count
        ),
        baseline_unique_backbone_family_count=(
            exploration_audit.topology_layer.unique_backbone_family_count
        ),
        baseline_unique_modifier_family_count=(
            exploration_audit.topology_layer.unique_modifier_family_count
        ),
        baseline_open_world_uncomposed_primitive_family_count=(
            exploration_audit.primitive_layer.open_world_uncomposed_primitive_family_count
        ),
        baseline_subordinate_fraction=exploration_audit.subordinate_idea_fraction,
        evolution_idea_count=len(ideas),
        unique_evolution_family_count=len(family_counts),
        duplicate_evolution_family_count=sum(
            count - 1 for count in family_counts.values() if count > 1
        ),
        idea_count_by_operator=dict(sorted(Counter(row.operator_id for row in ideas).items())),
        task_relation_mode_counts=dict(
            sorted(Counter(row.task_relation_mode for row in ideas).items())
        ),
        cross_source_bridge_count=len(bridge),
        cross_source_bridge_family_count=len(
            {row.conceptual_family_signature for row in bridge}
        ),
        open_world_composed_idea_count=sum(
            row.open_world_parent_involved and row.cross_source_composition
            for row in ideas
        ),
        open_world_parent_coverage_count=open_world_parent_coverage,
        open_world_parent_coverage_fraction=float(open_world_fraction),
        mutated_topology_count=len(mutations),
        unique_mutated_backbone_family_count=len(mutation_signatures),
        new_mutated_backbone_family_count=len(new_mutation_signatures),
        mutation_kind_counts=dict(sorted(mutation_kind_counts.items())),
        candidate_interpretation_count=len(candidate_rows),
        candidate_parent_coverage_count=len(candidate_parent_ids),
        reframe_idea_count=len(reframes),
        reframe_family_count=len(
            {row.conceptual_family_signature for row in reframes}
        ),
        reframe_count_by_operator=dict(sorted(reframe_counts.items())),
        source_pair_transitions=pair_rows,
        cross_source_transition_observed=cross_observed,
        new_backbone_transition_observed=backbone_observed,
        candidate_interpretation_observed=candidate_observed,
        reframe_observed=reframe_observed,
        conceptual_transition_observed=(
            cross_observed or backbone_observed or candidate_observed or reframe_observed
        ),
        native_llm_calls_attempted=evolution_report.native_llm_calls_attempted,
        native_llm_calls_succeeded=evolution_report.native_llm_calls_succeeded,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("audit_id", None)
    payload.pop("audit_sha256", None)
    sha = _sha(payload)
    return provisional.model_copy(
        update={
            "audit_id": f"idea_evolution_audit:{sha[:20]}",
            "audit_sha256": sha,
        }
    )


def build_idea_evolution_cohort_audit(
    audits: Sequence[tuple[str, IdeaEvolutionAudit]],
) -> IdeaEvolutionCohortAudit:
    if not audits:
        raise ValueError("Idea Evolution cohort audit requires at least one case")
    case_ids = [str(case_id) for case_id, _ in audits]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Idea Evolution cohort case IDs must be unique")

    cases = [
        IdeaEvolutionCohortCase(
            case_id=case_id,
            audit_id=audit.audit_id,
            source_report_id=audit.source_report_id,
            research_question=audit.research_question,
            evolution_idea_count=audit.evolution_idea_count,
            unique_evolution_family_count=audit.unique_evolution_family_count,
            cross_source_bridge_count=audit.cross_source_bridge_count,
            open_world_parent_coverage_fraction=audit.open_world_parent_coverage_fraction,
            new_mutated_backbone_family_count=audit.new_mutated_backbone_family_count,
            candidate_interpretation_count=audit.candidate_interpretation_count,
            reframe_idea_count=audit.reframe_idea_count,
            conceptual_transition_observed=audit.conceptual_transition_observed,
        )
        for case_id, audit in sorted(audits, key=lambda row: row[0])
    ]

    provisional = IdeaEvolutionCohortAudit(
        cohort_id="pending",
        cohort_sha256="pending",
        case_count=len(cases),
        cases=cases,
        total_evolution_idea_count=sum(row.evolution_idea_count for row in cases),
        case_count_with_cross_source_transition=sum(
            row.cross_source_bridge_count > 0 for row in cases
        ),
        case_count_with_new_backbone_transition=sum(
            row.new_mutated_backbone_family_count > 0 for row in cases
        ),
        case_count_with_candidate_interpretation=sum(
            row.candidate_interpretation_count > 0 for row in cases
        ),
        case_count_with_reframe=sum(row.reframe_idea_count > 0 for row in cases),
        case_count_with_any_conceptual_transition=sum(
            row.conceptual_transition_observed for row in cases
        ),
        median_evolution_idea_count=float(
            statistics.median(row.evolution_idea_count for row in cases)
        ),
        median_unique_evolution_family_count=float(
            statistics.median(row.unique_evolution_family_count for row in cases)
        ),
        median_open_world_parent_coverage_fraction=float(
            statistics.median(row.open_world_parent_coverage_fraction for row in cases)
        ),
        median_new_mutated_backbone_family_count=float(
            statistics.median(row.new_mutated_backbone_family_count for row in cases)
        ),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("cohort_id", None)
    payload.pop("cohort_sha256", None)
    sha = _sha(payload)
    return provisional.model_copy(
        update={
            "cohort_id": f"idea_evolution_cohort:{sha[:20]}",
            "cohort_sha256": sha,
        }
    )


__all__ = [
    "IdeaEvolutionAudit",
    "IdeaEvolutionCohortAudit",
    "IdeaEvolutionCohortCase",
    "IdeaEvolutionSourcePairTransition",
    "build_idea_evolution_audit",
    "build_idea_evolution_cohort_audit",
]
