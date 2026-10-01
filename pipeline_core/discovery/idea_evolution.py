from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.frontier_exploration_audit import (
    FrontierExplorationAudit,
)
from pipeline_core.discovery.frontier_idea_population import (
    FrontierIdea,
    FrontierIdeaPopulation,
)
from pipeline_core.discovery.reframing.proxy_challenge import (
    ProxyChallengeRunReport,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    ScientificReframingShadowReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


NativeEvolutionOperatorId = Literal[
    "CROSS_SOURCE_BRIDGE",
    "BACKBONE_MUTATION",
    "CANDIDATE_INTERPRETATION",
]

IdeaEvolutionOperatorId = Literal[
    "CROSS_SOURCE_BRIDGE",
    "BACKBONE_MUTATION",
    "CANDIDATE_INTERPRETATION",
    "LATENT_VARIABLE",
    "REGIME_BOUNDARY",
    "PROXY_CHALLENGE",
]

EvolutionIdeaForm = Literal[
    "CROSS_SOURCE_BRIDGE",
    "MUTATED_TOPOLOGY",
    "INTERPRETIVE_FORK",
    "SCIENTIFIC_REFRAME",
]

EvolutionTaskRelationMode = Literal[
    "DIRECT",
    "SUBORDINATE",
    "REFRAME",
    "UNKNOWN",
]

EvolutionMutationKind = Literal[
    "MEDIATOR_SUBSTITUTION",
    "MECHANISM_INSERTION",
    "PROXY_TARGET_SEPARATION",
    "CONDITION_INVERSION",
    "CAUSAL_DIRECTION_CHALLENGE",
    "MULTI_MECHANISM_COMPETITION",
]

EvolutionLineageKind = Literal[
    "FRONTIER_IDEA",
    "SCIENTIFIC_REFRAME",
    "PROXY_CHALLENGE",
]


def _norm(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[‐‑‒–—−-]+", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _sha256_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _dedupe_nonblank(values: Sequence[object]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _parent_source_kinds(parent_ideas: Sequence[FrontierIdea]) -> list[str]:
    return sorted({row.source_kind for row in parent_ideas})


def _inherited_external_work_ids(parent_ideas: Sequence[FrontierIdea]) -> list[str]:
    return sorted({
        work_id
        for idea in parent_ideas
        for lineage in idea.source_lineage
        for work_id in lineage.external_work_ids
        if str(work_id).strip()
    })


def _candidate_or_unverified(idea: FrontierIdea) -> bool:
    return any(
        lineage.candidate_or_unverified_lineage
        for lineage in idea.source_lineage
    )


def _idea_summary(idea: FrontierIdea) -> dict[str, Any]:
    return {
        "idea_id": idea.idea_id,
        "idea_form": idea.idea_form,
        "source_kind": idea.source_kind,
        "task_relation_mode": idea.task_relation_mode,
        "rendered_scientific_intent": idea.rendered_scientific_intent,
        "relation_signature": (
            idea.relation_signature.model_dump(mode="json")
            if idea.relation_signature is not None
            else None
        ),
        "topology_signature": (
            idea.topology_signature.model_dump(mode="json")
            if idea.topology_signature is not None
            else None
        ),
        "candidate_or_unverified_lineage": _candidate_or_unverified(idea),
        "external_literature_lineage": any(
            lineage.external_literature_lineage
            for lineage in idea.source_lineage
        ),
    }


class EvolutionLineageRef(StrictModel):
    lineage_kind: EvolutionLineageKind
    source_object_id: str = Field(min_length=1)
    source_artifact: str = Field(min_length=1)
    source_artifact_sha256: str = Field(min_length=64, max_length=64)
    source_kind: str = Field(min_length=1)


class IdeaEvolutionDraft(StrictModel):
    local_id: str = Field(min_length=1)
    operator_id: NativeEvolutionOperatorId
    parent_idea_ids: list[str] = Field(min_length=1, max_length=6)
    title: str = Field(min_length=1)
    scientific_intent: str = Field(min_length=1)
    conceptual_change_summary: str = Field(min_length=1)
    core_relations: list[str] = Field(min_length=1, max_length=4)
    transformed_question: str | None = None
    mutation_kind: EvolutionMutationKind | None = None
    challenged_assumption: str | None = None
    alternative_explanations: list[str] = Field(default_factory=list, max_length=2)
    differential_prediction: str = Field(min_length=1)
    falsification_condition: str = Field(min_length=1)
    discriminating_observation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_operator_shape(self) -> "IdeaEvolutionDraft":
        if len(self.parent_idea_ids) != len(set(self.parent_idea_ids)):
            raise ValueError("parent_idea_ids must be unique")
        if self.operator_id == "BACKBONE_MUTATION":
            if self.mutation_kind is None:
                raise ValueError("BACKBONE_MUTATION requires mutation_kind")
        elif self.mutation_kind is not None:
            raise ValueError("mutation_kind is exclusive to BACKBONE_MUTATION")

        if self.operator_id == "CANDIDATE_INTERPRETATION":
            if not str(self.challenged_assumption or "").strip():
                raise ValueError(
                    "CANDIDATE_INTERPRETATION requires challenged_assumption"
                )
            if len(self.alternative_explanations) != 2:
                raise ValueError(
                    "CANDIDATE_INTERPRETATION requires exactly two alternative_explanations"
                )
            if (
                _norm(self.alternative_explanations[0])
                == _norm(self.alternative_explanations[1])
            ):
                raise ValueError("alternative explanations must differ")
        elif self.alternative_explanations:
            raise ValueError(
                "alternative_explanations are exclusive to CANDIDATE_INTERPRETATION"
            )

        if self.operator_id != "CANDIDATE_INTERPRETATION" and str(
            self.challenged_assumption or ""
        ).strip():
            raise ValueError(
                "challenged_assumption is exclusive to CANDIDATE_INTERPRETATION"
            )
        if str(self.transformed_question or "").strip():
            raise ValueError(
                "native Frontier evolution remains task-subordinate; "
                "transformed_question is reserved for imported scientific reframes"
            )
        return self


class IdeaEvolutionBatchDraft(StrictModel):
    schema_version: Literal[
        "idea-evolution-batch-draft-v1"
    ] = "idea-evolution-batch-draft-v1"
    operator_id: NativeEvolutionOperatorId
    candidates: list[IdeaEvolutionDraft] = Field(default_factory=list)
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def validate_batch(self) -> "IdeaEvolutionBatchDraft":
        if any(row.operator_id != self.operator_id for row in self.candidates):
            raise ValueError("candidate operator_id must match batch operator_id")
        if self.candidates and self.abstention_reason:
            raise ValueError("non-empty batch must not carry abstention_reason")
        if not self.candidates and not str(self.abstention_reason or "").strip():
            raise ValueError("empty batch requires abstention_reason")
        ids = [row.local_id for row in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate local_id values must be unique")
        return self


class IdeaEvolutionIdea(StrictModel):
    schema_version: Literal[
        "idea-evolution-idea-v1"
    ] = "idea-evolution-idea-v1"

    evolution_id: str = Field(min_length=1)
    operator_id: IdeaEvolutionOperatorId
    idea_form: EvolutionIdeaForm

    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)

    parent_idea_ids: list[str] = Field(default_factory=list)
    parent_source_kinds: list[str] = Field(default_factory=list)
    lineage_refs: list[EvolutionLineageRef] = Field(min_length=1)

    title: str = Field(min_length=1)
    scientific_intent: str = Field(min_length=1)
    conceptual_change_summary: str = Field(min_length=1)
    core_relations: list[str] = Field(default_factory=list)
    transformed_question: str | None = None
    mutation_kind: EvolutionMutationKind | None = None
    challenged_assumption: str | None = None
    alternative_explanations: list[str] = Field(default_factory=list)

    differential_prediction: str = Field(min_length=1)
    falsification_condition: str = Field(min_length=1)
    discriminating_observation: str = Field(min_length=1)

    task_relation_mode: EvolutionTaskRelationMode
    task_relation_classification_authority: Literal[False] = False
    conceptual_family_signature: str = Field(min_length=1)

    inherited_external_work_ids: list[str] = Field(default_factory=list)
    inherited_external_literature_lineage: bool = False
    inherited_candidate_or_unverified_lineage: bool = False

    cross_source_composition: bool = False
    open_world_parent_involved: bool = False
    candidate_parent_involved: bool = False

    requires_verification: Literal[True] = True
    epistemic_status: Literal[
        "INSPIRATION_ONLY"
    ] = "INSPIRATION_ONLY"
    shadow_only: Literal[True] = True
    positive_premise_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_lineage_shape(self) -> "IdeaEvolutionIdea":
        native = self.operator_id in {
            "CROSS_SOURCE_BRIDGE",
            "BACKBONE_MUTATION",
            "CANDIDATE_INTERPRETATION",
        }
        if native and not self.parent_idea_ids:
            raise ValueError("native evolution idea requires Frontier parent ideas")
        if native and any(
            row.lineage_kind != "FRONTIER_IDEA"
            for row in self.lineage_refs
        ):
            raise ValueError("native evolution lineage must reference Frontier ideas")
        if not native and self.parent_idea_ids:
            raise ValueError("imported reframes must not invent Frontier parent lineage")
        if self.task_relation_mode == "REFRAME" and native:
            raise ValueError(
                "native Frontier evolution does not create REFRAME authority; "
                "use the existing scientific reframing subsystem"
            )
        return self


class IdeaEvolutionOperatorPlan(StrictModel):
    operator_id: NativeEvolutionOperatorId
    enabled: bool
    parent_pool_idea_ids: list[str] = Field(default_factory=list)
    max_output_count: int = Field(ge=1)
    reasons: list[str] = Field(default_factory=list)


class IdeaEvolutionPlan(StrictModel):
    schema_version: Literal[
        "idea-evolution-plan-v1"
    ] = "idea-evolution-plan-v1"

    plan_id: str
    plan_sha256: str
    source_population_id: str
    source_population_sha256: str
    source_exploration_audit_id: str
    source_exploration_audit_sha256: str
    source_context_id: str
    source_context_sha256: str
    research_question: str
    task_source: str
    task_target: str

    backbone_representative_idea_ids: list[str] = Field(default_factory=list)
    open_world_primitive_idea_ids: list[str] = Field(default_factory=list)
    kg_primitive_idea_ids: list[str] = Field(default_factory=list)
    candidate_topology_idea_ids: list[str] = Field(default_factory=list)
    operator_plans: list[IdeaEvolutionOperatorPlan]

    deterministic_parent_selection: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    candidate_deletion_authority: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False
    stage8_input_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_operator_uniqueness(self) -> "IdeaEvolutionPlan":
        operators = [row.operator_id for row in self.operator_plans]
        if len(operators) != len(set(operators)):
            raise ValueError("operator plans must be unique")
        return self


class IdeaEvolutionRunRecord(StrictModel):
    operator_id: NativeEvolutionOperatorId
    decision: Literal[
        "generated",
        "abstained",
        "rejected_invalid_draft",
        "generation_failed",
        "skipped_no_inputs",
    ]
    parent_pool_idea_ids: list[str] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)
    rejected_candidate_count: int = Field(default=0, ge=0)
    compile_issues: list[str] = Field(default_factory=list)
    abstention_reason: str | None = None
    generation_error_type: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None


class IdeaEvolutionReport(StrictModel):
    schema_version: Literal[
        "idea-evolution-shadow-report-v1"
    ] = "idea-evolution-shadow-report-v1"

    report_id: str
    report_sha256: str
    source_population_id: str
    source_population_sha256: str
    source_exploration_audit_id: str
    source_exploration_audit_sha256: str
    source_context_id: str
    source_context_sha256: str
    research_question: str
    task_source: str
    task_target: str

    plan_id: str
    native_runs: list[IdeaEvolutionRunRecord]
    ideas: list[IdeaEvolutionIdea]

    idea_count: int = Field(ge=0)
    idea_count_by_operator: dict[str, int] = Field(default_factory=dict)
    idea_count_by_form: dict[str, int] = Field(default_factory=dict)
    task_relation_mode_counts: dict[str, int] = Field(default_factory=dict)

    native_llm_calls_attempted: int = Field(ge=0)
    native_llm_calls_succeeded: int = Field(ge=0)
    imported_scientific_reframe_count: int = Field(ge=0)
    imported_proxy_challenge_count: int = Field(ge=0)

    cross_source_composition_count: int = Field(ge=0)
    open_world_composed_idea_count: int = Field(ge=0)
    candidate_continuation_count: int = Field(ge=0)
    reframe_idea_count: int = Field(ge=0)

    frontier_population_mutated: Literal[False] = False
    external_literature_as_positive_premise: Literal[False] = False
    new_retrieval_calls: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    n10_run: Literal[False] = False
    shadow_only: Literal[True] = True
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    generation_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False
    stage8_input_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "IdeaEvolutionReport":
        if self.idea_count != len(self.ideas):
            raise ValueError("idea_count mismatch")
        ids = [row.evolution_id for row in self.ideas]
        if len(ids) != len(set(ids)):
            raise ValueError("evolution IDs must be unique")
        return self


def _frontier_lineage_ref(
    *,
    idea: FrontierIdea,
    source_artifact: str,
    source_artifact_sha256: str,
) -> EvolutionLineageRef:
    return EvolutionLineageRef(
        lineage_kind="FRONTIER_IDEA",
        source_object_id=idea.idea_id,
        source_artifact=source_artifact,
        source_artifact_sha256=source_artifact_sha256,
        source_kind=idea.source_kind,
    )


def _family_signature(
    *,
    operator_id: str,
    core_relations: Sequence[str],
    transformed_question: str | None,
    challenged_assumption: str | None,
    alternative_explanations: Sequence[str],
) -> str:
    payload = {
        "operator_id": operator_id,
        "core_relations": sorted(_norm(x) for x in core_relations if _norm(x)),
        "transformed_question": _norm(transformed_question),
        "challenged_assumption": _norm(challenged_assumption),
        "alternative_explanations": sorted(
            _norm(x) for x in alternative_explanations if _norm(x)
        ),
    }
    return _sha256_json(payload)


def _plan_payload_without_identity(plan: IdeaEvolutionPlan) -> dict[str, Any]:
    payload = plan.model_dump(mode="json")
    payload.pop("plan_id", None)
    payload.pop("plan_sha256", None)
    return payload


def build_idea_evolution_plan(
    *,
    population: FrontierIdeaPopulation,
    exploration_audit: FrontierExplorationAudit,
    max_cross_source_outputs: int = 6,
    max_backbone_mutation_outputs: int = 6,
    max_candidate_interpretation_outputs: int = 4,
    max_candidate_parent_pool: int = 8,
) -> IdeaEvolutionPlan:
    if exploration_audit.population_id != population.population_id:
        raise ValueError("exploration audit population_id mismatch")
    if exploration_audit.population_sha256 != population.population_sha256:
        raise ValueError("exploration audit population_sha256 mismatch")

    by_id = {row.idea_id: row for row in population.ideas}

    open_world = [
        row.idea_id
        for row in population.ideas
        if row.source_kind == "OPEN_WORLD_AXIS"
        and row.relation_signature is not None
    ]
    kg = [
        row.idea_id
        for row in population.ideas
        if row.source_kind == "KG_AXIS"
        and row.relation_signature is not None
    ]

    backbone_representatives: list[str] = []
    for family in exploration_audit.topology_layer.backbone_families:
        if not family.topology_idea_ids:
            continue
        idea_id = family.topology_idea_ids[0]
        if idea_id not in by_id:
            raise ValueError(
                "exploration audit references unknown topology idea: " + idea_id
            )
        backbone_representatives.append(idea_id)

    candidate_topologies: list[str] = []
    seen_candidate_parent_keys: set[str] = set()
    for row in population.ideas:
        if row.topology_signature is None or not _candidate_or_unverified(row):
            continue
        signature = row.topology_signature
        candidate_key = (
            str(signature.modifier_candidate_unit_id or "").strip()
            or str(signature.modifier_component_id or "").strip()
            or row.idea_id
        )
        if candidate_key in seen_candidate_parent_keys:
            continue
        seen_candidate_parent_keys.add(candidate_key)
        candidate_topologies.append(row.idea_id)
        if len(candidate_topologies) >= max_candidate_parent_pool:
            break

    cross_pool = _dedupe_nonblank([
        *open_world,
        *kg,
        *backbone_representatives,
    ])
    mutation_pool = _dedupe_nonblank([
        *backbone_representatives,
        *kg,
        *open_world,
    ])

    operator_plans = [
        IdeaEvolutionOperatorPlan(
            operator_id="CROSS_SOURCE_BRIDGE",
            enabled=bool(open_world and (kg or backbone_representatives)),
            parent_pool_idea_ids=cross_pool,
            max_output_count=max_cross_source_outputs,
            reasons=(
                [
                    "At least one OPEN_WORLD_AXIS primitive and one non-open-world Frontier idea are available.",
                    "Cross-source composition is exploration-only and does not promote external literature into evidence.",
                ]
                if open_world and (kg or backbone_representatives)
                else [
                    "Cross-source evolution requires both OPEN_WORLD_AXIS and non-open-world Frontier parents."
                ]
            ),
        ),
        IdeaEvolutionOperatorPlan(
            operator_id="BACKBONE_MUTATION",
            enabled=bool(backbone_representatives),
            parent_pool_idea_ids=mutation_pool,
            max_output_count=max_backbone_mutation_outputs,
            reasons=(
                [
                    "One representative from each observed backbone family is retained without scientific ranking.",
                    "Primitive axes may be cited as inspiration material, but the mutated backbone must differ structurally from its parent backbone.",
                ]
                if backbone_representatives
                else ["No observed topology backbone family is available for mutation."]
            ),
        ),
        IdeaEvolutionOperatorPlan(
            operator_id="CANDIDATE_INTERPRETATION",
            enabled=bool(candidate_topologies),
            parent_pool_idea_ids=candidate_topologies,
            max_output_count=max_candidate_interpretation_outputs,
            reasons=(
                [
                    "Candidate/unverified topology lineage exists and may be reframed into explicit competing interpretations without gaining evidence authority."
                ]
                if candidate_topologies
                else [
                    "No candidate/unverified topology lineage is available for interpretive continuation."
                ]
            ),
        ),
    ]

    provisional = IdeaEvolutionPlan(
        plan_id="pending",
        plan_sha256="pending",
        source_population_id=population.population_id,
        source_population_sha256=population.population_sha256,
        source_exploration_audit_id=exploration_audit.audit_id,
        source_exploration_audit_sha256=exploration_audit.audit_sha256,
        source_context_id=population.source_context_id,
        source_context_sha256=population.source_context_sha256,
        research_question=population.research_question,
        task_source=population.task_source,
        task_target=population.task_target,
        backbone_representative_idea_ids=backbone_representatives,
        open_world_primitive_idea_ids=open_world,
        kg_primitive_idea_ids=kg,
        candidate_topology_idea_ids=candidate_topologies,
        operator_plans=operator_plans,
    )
    payload = _plan_payload_without_identity(provisional)
    sha = _sha256_json(payload)
    return provisional.model_copy(
        update={
            "plan_id": f"idea_evolution_plan:{sha[:20]}",
            "plan_sha256": sha,
        }
    )


def compile_native_evolution_draft(
    *,
    draft: IdeaEvolutionDraft,
    operator_plan: IdeaEvolutionOperatorPlan,
    population: FrontierIdeaPopulation,
    population_artifact: str,
    population_artifact_sha256: str,
) -> IdeaEvolutionIdea:
    if draft.operator_id != operator_plan.operator_id:
        raise ValueError("draft operator does not match operator plan")
    if not operator_plan.enabled:
        raise ValueError("cannot compile a disabled operator plan")

    by_id = {row.idea_id: row for row in population.ideas}
    allowed = set(operator_plan.parent_pool_idea_ids)
    unknown = [row for row in draft.parent_idea_ids if row not in by_id]
    if unknown:
        raise ValueError("draft references unknown parent ideas: " + repr(unknown))
    outside = [row for row in draft.parent_idea_ids if row not in allowed]
    if outside:
        raise ValueError("draft references parents outside frozen operator pool: " + repr(outside))

    parent_ideas = [by_id[row] for row in draft.parent_idea_ids]
    parent_kinds = _parent_source_kinds(parent_ideas)

    if draft.operator_id == "CROSS_SOURCE_BRIDGE":
        if len(parent_ideas) < 2:
            raise ValueError("CROSS_SOURCE_BRIDGE requires at least two parents")
        if len(parent_kinds) < 2:
            raise ValueError("CROSS_SOURCE_BRIDGE requires distinct source kinds")
        if "OPEN_WORLD_AXIS" not in parent_kinds:
            raise ValueError("CROSS_SOURCE_BRIDGE requires an OPEN_WORLD_AXIS parent")
        if set(parent_kinds) == {"OPEN_WORLD_AXIS"}:
            raise ValueError("CROSS_SOURCE_BRIDGE requires a non-open-world parent")
        if not _inherited_external_work_ids(parent_ideas):
            raise ValueError(
                "CROSS_SOURCE_BRIDGE requires preserved external-literature lineage"
            )
        form: EvolutionIdeaForm = "CROSS_SOURCE_BRIDGE"
    elif draft.operator_id == "BACKBONE_MUTATION":
        topology_parents = [
            row for row in parent_ideas if row.topology_signature is not None
        ]
        if not topology_parents:
            raise ValueError("BACKBONE_MUTATION requires a topology parent")
        proposed = tuple(sorted(_norm(row) for row in draft.core_relations))
        parent_backbones = {
            tuple(sorted(_norm(row) for row in idea.topology_signature.backbone_relation_texts))
            for idea in topology_parents
            if idea.topology_signature is not None
        }
        if proposed in parent_backbones:
            raise ValueError(
                "BACKBONE_MUTATION core_relations are unchanged from a parent backbone"
            )
        form = "MUTATED_TOPOLOGY"
    else:
        if not any(_candidate_or_unverified(row) for row in parent_ideas):
            raise ValueError(
                "CANDIDATE_INTERPRETATION requires candidate/unverified parent lineage"
            )
        form = "INTERPRETIVE_FORK"

    external_work_ids = _inherited_external_work_ids(parent_ideas)
    candidate_parent = any(_candidate_or_unverified(row) for row in parent_ideas)
    open_world_parent = "OPEN_WORLD_AXIS" in parent_kinds
    cross_source = len(parent_kinds) > 1

    family_signature = _family_signature(
        operator_id=draft.operator_id,
        core_relations=draft.core_relations,
        transformed_question=draft.transformed_question,
        challenged_assumption=draft.challenged_assumption,
        alternative_explanations=draft.alternative_explanations,
    )

    evolution_id = _stable_id(
        "idea_evolution",
        draft.operator_id,
        *draft.parent_idea_ids,
        family_signature,
    )

    return IdeaEvolutionIdea(
        evolution_id=evolution_id,
        operator_id=draft.operator_id,
        idea_form=form,
        source_context_id=population.source_context_id,
        source_context_sha256=population.source_context_sha256,
        parent_idea_ids=list(draft.parent_idea_ids),
        parent_source_kinds=parent_kinds,
        lineage_refs=[
            _frontier_lineage_ref(
                idea=idea,
                source_artifact=population_artifact,
                source_artifact_sha256=population_artifact_sha256,
            )
            for idea in parent_ideas
        ],
        title=draft.title,
        scientific_intent=draft.scientific_intent,
        conceptual_change_summary=draft.conceptual_change_summary,
        core_relations=_dedupe_nonblank(draft.core_relations),
        transformed_question=draft.transformed_question,
        mutation_kind=draft.mutation_kind,
        challenged_assumption=draft.challenged_assumption,
        alternative_explanations=_dedupe_nonblank(draft.alternative_explanations),
        differential_prediction=draft.differential_prediction,
        falsification_condition=draft.falsification_condition,
        discriminating_observation=draft.discriminating_observation,
        task_relation_mode="UNKNOWN",
        conceptual_family_signature=family_signature,
        inherited_external_work_ids=external_work_ids,
        inherited_external_literature_lineage=bool(external_work_ids),
        inherited_candidate_or_unverified_lineage=candidate_parent,
        cross_source_composition=cross_source,
        open_world_parent_involved=open_world_parent,
        candidate_parent_involved=candidate_parent,
    )


def _validate_import_context(
    *,
    population: FrontierIdeaPopulation,
    source_context_id: str,
    source_context_sha256: str,
    label: str,
) -> None:
    if source_context_id != population.source_context_id:
        raise ValueError(f"{label} source_context_id mismatch")
    if source_context_sha256 != population.source_context_sha256:
        raise ValueError(f"{label} source_context_sha256 mismatch")


def import_scientific_reframe_ideas(
    *,
    report: ScientificReframingShadowReport,
    population: FrontierIdeaPopulation,
    source_artifact: str,
    source_artifact_sha256: str,
) -> list[IdeaEvolutionIdea]:
    _validate_import_context(
        population=population,
        source_context_id=report.source_context_id,
        source_context_sha256=report.source_context_sha256,
        label="scientific reframe report",
    )
    ideas: list[IdeaEvolutionIdea] = []
    for candidate in report.candidates:
        predictions = list(candidate.differential_predictions)
        falsifiers = list(candidate.falsifiers)
        prediction = (
            predictions[0].discriminating_outcome
            if predictions
            else "A discriminating outcome remains to be specified."
        )
        falsifier = (
            falsifiers[0].falsifying_outcome
            if falsifiers
            else "The proposed reframe is falsified if its alternative model has no discriminating consequence."
        )
        transformed_question = (
            "What observations distinguish the baseline model '"
            + candidate.baseline_model.summary
            + "' from the alternative model '"
            + candidate.alternative_model.summary
            + "' for the original task?"
        )
        core_relations = _dedupe_nonblank([
            *candidate.proposed_constructs,
            *candidate.latent_constructs,
            *candidate.boundary_variables,
        ])
        family_signature = _family_signature(
            operator_id=candidate.operator_id,
            core_relations=core_relations,
            transformed_question=transformed_question,
            challenged_assumption=candidate.challenged_assumption,
            alternative_explanations=[
                candidate.baseline_model.summary,
                candidate.alternative_model.summary,
            ],
        )
        ideas.append(
            IdeaEvolutionIdea(
                evolution_id=_stable_id(
                    "idea_evolution_import",
                    candidate.reframe_id,
                    family_signature,
                ),
                operator_id=candidate.operator_id,
                idea_form="SCIENTIFIC_REFRAME",
                source_context_id=population.source_context_id,
                source_context_sha256=population.source_context_sha256,
                lineage_refs=[
                    EvolutionLineageRef(
                        lineage_kind="SCIENTIFIC_REFRAME",
                        source_object_id=candidate.reframe_id,
                        source_artifact=source_artifact,
                        source_artifact_sha256=source_artifact_sha256,
                        source_kind=candidate.operator_id,
                    )
                ],
                title=candidate.title,
                scientific_intent=(
                    "Challenge the assumption that "
                    + candidate.challenged_assumption
                    + ". Compare baseline: "
                    + candidate.baseline_model.summary
                    + " Alternative: "
                    + candidate.alternative_model.summary
                ),
                conceptual_change_summary=candidate.challenged_assumption,
                core_relations=core_relations,
                transformed_question=transformed_question,
                challenged_assumption=candidate.challenged_assumption,
                alternative_explanations=[
                    candidate.baseline_model.summary,
                    candidate.alternative_model.summary,
                ],
                differential_prediction=prediction,
                falsification_condition=falsifier,
                discriminating_observation=candidate.discriminating_test.test_design,
                task_relation_mode="REFRAME",
                conceptual_family_signature=family_signature,
            )
        )
    return ideas


def import_proxy_challenge_ideas(
    *,
    report: ProxyChallengeRunReport,
    population: FrontierIdeaPopulation,
    source_artifact: str,
    source_artifact_sha256: str,
) -> list[IdeaEvolutionIdea]:
    _validate_import_context(
        population=population,
        source_context_id=report.source_context_id,
        source_context_sha256=report.source_context_sha256,
        label="proxy challenge report",
    )
    ideas: list[IdeaEvolutionIdea] = []
    for candidate in report.candidates:
        prediction = candidate.differential_predictions[0].discriminating_outcome
        falsifier = candidate.falsifiers[0].falsifying_outcome
        transformed_question = (
            "Does the observable '"
            + candidate.challenged_observable
            + "' remain an adequate proxy for '"
            + candidate.target_construct
            + "' across the original task conditions?"
        )
        core_relations = [
            f"{candidate.challenged_observable} --PROXY_FOR?--> {candidate.target_construct}"
        ]
        alternatives = [
            candidate.baseline_model.summary,
            candidate.alternative_model.summary,
        ]
        family_signature = _family_signature(
            operator_id="PROXY_CHALLENGE",
            core_relations=core_relations,
            transformed_question=transformed_question,
            challenged_assumption=candidate.challenged_proxy_assumption,
            alternative_explanations=alternatives,
        )
        ideas.append(
            IdeaEvolutionIdea(
                evolution_id=_stable_id(
                    "idea_evolution_import",
                    candidate.candidate_id,
                    family_signature,
                ),
                operator_id="PROXY_CHALLENGE",
                idea_form="SCIENTIFIC_REFRAME",
                source_context_id=population.source_context_id,
                source_context_sha256=population.source_context_sha256,
                lineage_refs=[
                    EvolutionLineageRef(
                        lineage_kind="PROXY_CHALLENGE",
                        source_object_id=candidate.candidate_id,
                        source_artifact=source_artifact,
                        source_artifact_sha256=source_artifact_sha256,
                        source_kind="PROXY_CHALLENGE",
                    )
                ],
                title=candidate.title,
                scientific_intent=(
                    "Challenge proxy interchangeability: "
                    + candidate.challenged_proxy_assumption
                    + " Baseline: "
                    + candidate.baseline_model.summary
                    + " Alternative: "
                    + candidate.alternative_model.summary
                ),
                conceptual_change_summary=candidate.challenged_proxy_assumption,
                core_relations=core_relations,
                transformed_question=transformed_question,
                challenged_assumption=candidate.challenged_proxy_assumption,
                alternative_explanations=alternatives,
                differential_prediction=prediction,
                falsification_condition=falsifier,
                discriminating_observation=candidate.discriminating_test.test_design,
                task_relation_mode="REFRAME",
                conceptual_family_signature=family_signature,
            )
        )
    return ideas


def build_idea_evolution_report(
    *,
    population: FrontierIdeaPopulation,
    exploration_audit: FrontierExplorationAudit,
    plan: IdeaEvolutionPlan,
    native_runs: Sequence[IdeaEvolutionRunRecord],
    native_ideas: Sequence[IdeaEvolutionIdea],
    imported_reframe_ideas: Sequence[IdeaEvolutionIdea] = (),
    imported_proxy_ideas: Sequence[IdeaEvolutionIdea] = (),
) -> IdeaEvolutionReport:
    if plan.source_population_id != population.population_id:
        raise ValueError("plan/population mismatch")
    if plan.source_exploration_audit_id != exploration_audit.audit_id:
        raise ValueError("plan/exploration audit mismatch")

    ideas = [
        *native_ideas,
        *imported_reframe_ideas,
        *imported_proxy_ideas,
    ]
    # Preserve source/operator order but exact-dedupe repeated IDs fail closed.
    ids = [row.evolution_id for row in ideas]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate evolution IDs")

    by_operator = Counter(row.operator_id for row in ideas)
    by_form = Counter(row.idea_form for row in ideas)
    by_task = Counter(row.task_relation_mode for row in ideas)

    provisional = IdeaEvolutionReport(
        report_id="pending",
        report_sha256="pending",
        source_population_id=population.population_id,
        source_population_sha256=population.population_sha256,
        source_exploration_audit_id=exploration_audit.audit_id,
        source_exploration_audit_sha256=exploration_audit.audit_sha256,
        source_context_id=population.source_context_id,
        source_context_sha256=population.source_context_sha256,
        research_question=population.research_question,
        task_source=population.task_source,
        task_target=population.task_target,
        plan_id=plan.plan_id,
        native_runs=list(native_runs),
        ideas=ideas,
        idea_count=len(ideas),
        idea_count_by_operator=dict(sorted(by_operator.items())),
        idea_count_by_form=dict(sorted(by_form.items())),
        task_relation_mode_counts=dict(sorted(by_task.items())),
        native_llm_calls_attempted=sum(
            row.decision != "skipped_no_inputs"
            for row in native_runs
        ),
        native_llm_calls_succeeded=sum(
            row.decision in {
                "generated",
                "abstained",
                "rejected_invalid_draft",
            }
            for row in native_runs
        ),
        imported_scientific_reframe_count=len(imported_reframe_ideas),
        imported_proxy_challenge_count=len(imported_proxy_ideas),
        cross_source_composition_count=sum(
            row.cross_source_composition for row in ideas
        ),
        open_world_composed_idea_count=sum(
            row.open_world_parent_involved and row.cross_source_composition
            for row in ideas
        ),
        candidate_continuation_count=sum(
            row.operator_id == "CANDIDATE_INTERPRETATION"
            for row in ideas
        ),
        reframe_idea_count=sum(
            row.task_relation_mode == "REFRAME"
            for row in ideas
        ),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    sha = _sha256_json(payload)
    return provisional.model_copy(
        update={
            "report_id": f"idea_evolution_report:{sha[:20]}",
            "report_sha256": sha,
        }
    )


def operator_plan_by_id(
    plan: IdeaEvolutionPlan,
) -> dict[NativeEvolutionOperatorId, IdeaEvolutionOperatorPlan]:
    return {
        row.operator_id: row
        for row in plan.operator_plans
    }


def frontier_idea_payloads(
    *,
    population: FrontierIdeaPopulation,
    idea_ids: Sequence[str],
) -> list[dict[str, Any]]:
    by_id = {row.idea_id: row for row in population.ideas}
    unknown = [idea_id for idea_id in idea_ids if idea_id not in by_id]
    if unknown:
        raise ValueError("unknown Frontier idea IDs: " + repr(unknown))
    return [_idea_summary(by_id[idea_id]) for idea_id in idea_ids]


__all__ = [
    "EvolutionIdeaForm",
    "EvolutionLineageRef",
    "EvolutionMutationKind",
    "EvolutionTaskRelationMode",
    "IdeaEvolutionBatchDraft",
    "IdeaEvolutionDraft",
    "IdeaEvolutionIdea",
    "IdeaEvolutionOperatorId",
    "IdeaEvolutionOperatorPlan",
    "IdeaEvolutionPlan",
    "IdeaEvolutionReport",
    "IdeaEvolutionRunRecord",
    "NativeEvolutionOperatorId",
    "build_idea_evolution_plan",
    "build_idea_evolution_report",
    "compile_native_evolution_draft",
    "frontier_idea_payloads",
    "import_proxy_challenge_ideas",
    "import_scientific_reframe_ideas",
    "operator_plan_by_id",
]
