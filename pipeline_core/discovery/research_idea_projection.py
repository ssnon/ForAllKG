from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Mapping, Sequence

from pipeline_core.discovery.frontier_idea_population import (
    FrontierIdea,
    FrontierIdeaPopulation,
)
from pipeline_core.discovery.idea_evolution import (
    IdeaEvolutionIdea,
    IdeaEvolutionReport,
)
from pipeline_core.discovery.research_idea_contracts import (
    ResearchIdeaKernel,
    ResearchIdeaNode,
)


_NATIVE_EVOLUTION_OPERATORS = {
    "CROSS_SOURCE_BRIDGE",
    "BACKBONE_MUTATION",
    "CANDIDATE_INTERPRETATION",
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
    raw = _canonical_json(parts)
    return f"{prefix}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _dedupe_nonblank(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _kernel_with_fallback(
    *,
    canonical_intent: str,
    core: Sequence[object],
    scope: Sequence[object] = (),
    contrast: Sequence[object] = (),
    question: str | None = None,
) -> tuple[ResearchIdeaKernel, list[str]]:
    diagnostics: list[str] = []
    core_values = _dedupe_nonblank(core)
    if not core_values:
        core_values = [str(canonical_intent).strip()]
        diagnostics.append("KERNEL_CORE_FALLBACK_TO_INTENT")
    return (
        ResearchIdeaKernel(
            canonical_intent=str(canonical_intent).strip(),
            core_scientific_commitments=core_values,
            scope_commitments=_dedupe_nonblank(scope),
            contrastive_commitments=_dedupe_nonblank(contrast),
            question_commitment=(
                str(question).strip()
                if question is not None and str(question).strip()
                else None
            ),
        ),
        diagnostics,
    )


def _frontier_kernel(
    idea: FrontierIdea,
) -> tuple[ResearchIdeaKernel, list[str]]:
    scope: list[str] = []
    contrast: list[str] = []
    core: list[str] = []

    if idea.relation_signature is not None:
        row = idea.relation_signature
        core = [f"{row.subject} --{row.relation}--> {row.object}"]

    elif idea.topology_signature is not None:
        row = idea.topology_signature
        core = list(row.backbone_relation_texts)
        if str(row.modifier_text or "").strip():
            scope.append(f"modifier: {row.modifier_text}")
        if str(row.modifier_anchor_role or "").strip():
            scope.append(f"modifier_anchor_role: {row.modifier_anchor_role}")
        if str(row.modifier_anchor_text or "").strip():
            scope.append(f"modifier_anchor_text: {row.modifier_anchor_text}")
        if str(row.modifier_relation_text or "").strip():
            scope.append(f"modifier_relation: {row.modifier_relation_text}")

    elif idea.tension_signature is not None:
        row = idea.tension_signature
        core = list(row.basis_relation_texts)
        contrast = [
            f"tension_type: {row.tension_type}",
            row.tension_statement,
        ]

    elif idea.competing_explanation_signature is not None:
        row = idea.competing_explanation_signature
        core = list(row.basis_relation_texts)
        contrast = [
            f"explanation_type: {row.explanation_type}",
            row.explanation_statement,
            f"discriminator: {row.discriminator_requirement}",
        ]

    return _kernel_with_fallback(
        canonical_intent=idea.rendered_scientific_intent,
        core=core,
        scope=scope,
        contrast=contrast,
    )


def _evolution_kernel(
    idea: IdeaEvolutionIdea,
) -> tuple[ResearchIdeaKernel, list[str]]:
    contrast = _dedupe_nonblank(
        [
            idea.challenged_assumption,
            *idea.alternative_explanations,
        ]
    )
    return _kernel_with_fallback(
        canonical_intent=idea.scientific_intent,
        core=idea.core_relations,
        contrast=contrast,
        question=idea.transformed_question,
    )


def project_frontier_idea(
    idea: FrontierIdea,
    *,
    generation_index: int = 0,
) -> ResearchIdeaNode:
    kernel, diagnostics = _frontier_kernel(idea)
    source_artifacts = _dedupe_nonblank(
        row.source_artifact
        for row in idea.source_lineage
    )
    return ResearchIdeaNode(
        idea_id=_stable_id("research_idea", "FRONTIER", idea.idea_id),
        generation_index=generation_index,
        parent_idea_ids=[],
        source_context_id=idea.source_context_id,
        source_context_sha256=idea.source_context_sha256,
        origin_kind="FRONTIER",
        source_object_id=idea.idea_id,
        source_parent_object_ids=[],
        source_kind=idea.source_kind,
        operator_id=None,
        source_artifact_refs=source_artifacts,
        kernel=kernel,
        kernel_sha256=_sha(kernel),
        task_relation_mode=idea.task_relation_mode,
        projection_diagnostic_codes=diagnostics,
    )


def project_evolution_idea(
    idea: IdeaEvolutionIdea,
    *,
    parent_research_id_by_source_object_id: Mapping[str, str] | None = None,
    generation_index: int = 1,
) -> ResearchIdeaNode:
    kernel, diagnostics = _evolution_kernel(idea)
    source_parent_ids = _dedupe_nonblank(idea.parent_idea_ids)
    parent_map = dict(parent_research_id_by_source_object_id or {})
    native = idea.operator_id in _NATIVE_EVOLUTION_OPERATORS

    if native:
        missing = [source_id for source_id in source_parent_ids if source_id not in parent_map]
        if missing:
            raise ValueError(
                "native evolution projection is missing ResearchIdea parent mapping: "
                + repr(missing)
            )
        parent_ids = [parent_map[source_id] for source_id in source_parent_ids]
        origin_kind = "EVOLUTION"
    else:
        if source_parent_ids:
            raise ValueError(
                "existing imported reframe must not invent Frontier parent lineage"
            )
        parent_ids = []
        origin_kind = "IMPORTED_REFRAME"

    source_artifacts = _dedupe_nonblank(
        row.source_artifact
        for row in idea.lineage_refs
    )

    return ResearchIdeaNode(
        idea_id=_stable_id("research_idea", origin_kind, idea.evolution_id),
        generation_index=generation_index,
        parent_idea_ids=parent_ids,
        source_context_id=idea.source_context_id,
        source_context_sha256=idea.source_context_sha256,
        origin_kind=origin_kind,
        source_object_id=idea.evolution_id,
        source_parent_object_ids=source_parent_ids,
        source_kind=None,
        operator_id=idea.operator_id,
        source_artifact_refs=source_artifacts,
        kernel=kernel,
        kernel_sha256=_sha(kernel),
        task_relation_mode=idea.task_relation_mode,
        differential_prediction=idea.differential_prediction,
        falsification_condition=idea.falsification_condition,
        discriminating_observation=idea.discriminating_observation,
        projection_diagnostic_codes=diagnostics,
    )


def project_current_idea_artifacts(
    *,
    population: FrontierIdeaPopulation,
    evolution_report: IdeaEvolutionReport,
) -> list[ResearchIdeaNode]:
    """Project the current one-shot idea artifacts without changing authority.

    Projection is one source object -> one ResearchIdeaNode. No semantic merge,
    family collapse, candidate deletion, or lineage invention is performed.
    """

    if evolution_report.source_population_id != population.population_id:
        raise ValueError("Idea Evolution/Frontier population lineage mismatch")
    if evolution_report.source_context_id != population.source_context_id:
        raise ValueError("Idea Evolution/Frontier source context mismatch")

    frontier_nodes = [
        project_frontier_idea(idea, generation_index=0)
        for idea in population.ideas
    ]
    parent_map = {
        node.source_object_id: node.idea_id
        for node in frontier_nodes
    }
    evolution_nodes = [
        project_evolution_idea(
            idea,
            parent_research_id_by_source_object_id=parent_map,
            generation_index=1,
        )
        for idea in evolution_report.ideas
    ]
    nodes = [*frontier_nodes, *evolution_nodes]
    ids = [node.idea_id for node in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("projected ResearchIdea IDs must be unique")
    return nodes


__all__ = [
    "project_current_idea_artifacts",
    "project_evolution_idea",
    "project_frontier_idea",
]
