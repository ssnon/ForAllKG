from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.frontier_idea_population import FrontierIdeaPopulation
from pipeline_core.discovery.idea_evolution import IdeaEvolutionReport
from pipeline_core.discovery.research_idea_contracts import ResearchIdeaNode
from pipeline_core.discovery.research_idea_epistemic_generational_evolution import (
    EpistemicG4ExecutionReport,
)
from pipeline_core.discovery.research_idea_projection import (
    project_current_idea_artifacts,
)
from pipeline_core.discovery.scientific_portfolio_selection import (
    ScientificPortfolioCandidatePool,
    ScientificPortfolioSelectionReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical(parts).encode('utf-8')).hexdigest()[:20]}"


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


class E2ESeedSelectionLink(StrictModel):
    candidate_id: str = Field(min_length=1)
    source_object_id: str = Field(min_length=1)
    candidate_origin: str = Field(min_length=1)
    research_idea_id: str = Field(min_length=1)
    research_idea_origin_kind: str = Field(min_length=1)
    original_generation_index: int = Field(ge=0)


class ResearchIdeaE2ESeedReport(StrictModel):
    schema_version: Literal[
        "sis-v3-2-e2e-research-idea-seed-v1"
    ] = "sis-v3-2-e2e-research-idea-seed-v1"

    seed_id: str
    seed_sha256: str

    source_population_id: str
    source_population_sha256: str
    source_evolution_report_id: str
    source_evolution_report_sha256: str
    source_pool_id: str
    source_pool_sha256: str
    source_selection_id: str
    source_selection_sha256: str
    source_context_id: str
    source_context_sha256: str

    sis_search_epoch: Literal[0] = 0
    compatibility_execution_generation_index: Literal[2] = 2
    selected_candidate_count: int = Field(ge=0)
    selected_candidate_ids: list[str] = Field(default_factory=list)
    selected_source_object_ids: list[str] = Field(default_factory=list)
    selected_research_idea_ids: list[str] = Field(default_factory=list)
    selected_count_by_candidate_origin: dict[str, int] = Field(default_factory=dict)
    links: list[E2ESeedSelectionLink] = Field(default_factory=list)
    research_ideas: list[ResearchIdeaNode] = Field(default_factory=list)

    p0_identity_frozen: Literal[True] = True
    p0_selected_from_scientific_portfolio: Literal[True] = True
    scientific_portfolio_selection_is_not_truth_authority: Literal[True] = True
    original_research_idea_generation_indices_preserved: Literal[True] = True
    compatibility_execution_generation_does_not_rewrite_idea_genealogy: Literal[True] = True
    positive_premise_authority_created: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "ResearchIdeaE2ESeedReport":
        count = len(self.research_ideas)
        if self.selected_candidate_count != count:
            raise ValueError("selected_candidate_count mismatch")
        if len(self.links) != count:
            raise ValueError("seed link count mismatch")
        if len(self.selected_candidate_ids) != count:
            raise ValueError("selected candidate ID count mismatch")
        if len(self.selected_source_object_ids) != count:
            raise ValueError("selected source object ID count mismatch")
        if len(self.selected_research_idea_ids) != count:
            raise ValueError("selected ResearchIdea ID count mismatch")
        for values, label in (
            (self.selected_candidate_ids, "selected_candidate_ids"),
            (self.selected_source_object_ids, "selected_source_object_ids"),
            (self.selected_research_idea_ids, "selected_research_idea_ids"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        if self.selected_research_idea_ids != [row.idea_id for row in self.research_ideas]:
            raise ValueError("selected ResearchIdea order does not match research_ideas")
        return self


def select_retained_research_ideas(
    *,
    projected_nodes: Sequence[ResearchIdeaNode],
    pool_candidates: Sequence[Any],
    selection_entries: Sequence[Any],
) -> tuple[list[ResearchIdeaNode], list[E2ESeedSelectionLink]]:
    """Map retained Scientific Portfolio candidates back to canonical ResearchIdeaNodes.

    The mapping is deterministic and lineage-only. It does not rerank, merge, repair,
    or reinterpret scientific content.
    """

    node_by_source: dict[str, ResearchIdeaNode] = {}
    for node in projected_nodes:
        if node.source_object_id in node_by_source:
            raise ValueError(
                "ResearchIdea projection contains duplicate source_object_id: "
                + node.source_object_id
            )
        node_by_source[node.source_object_id] = node

    candidate_by_id = {
        str(_get(row, "candidate_id")): row
        for row in pool_candidates
    }

    selected: list[ResearchIdeaNode] = []
    links: list[E2ESeedSelectionLink] = []
    seen_source_ids: set[str] = set()
    for entry in selection_entries:
        candidate_id = str(_get(entry, "candidate_id") or "")
        source_object_id = str(_get(entry, "source_object_id") or "")
        origin = str(_get(entry, "origin") or "")
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(
                f"Scientific Portfolio selection references unknown candidate: {candidate_id}"
            )
        candidate_source_object_id = str(_get(candidate, "source_object_id") or "")
        candidate_origin = str(_get(candidate, "origin") or "")
        if source_object_id != candidate_source_object_id:
            raise ValueError(
                "selection/candidate source_object_id mismatch for " + candidate_id
            )
        if origin and candidate_origin and origin != candidate_origin:
            raise ValueError("selection/candidate origin mismatch for " + candidate_id)
        if source_object_id in seen_source_ids:
            raise ValueError(
                "Scientific Portfolio retained the same source object more than once: "
                + source_object_id
            )
        node = node_by_source.get(source_object_id)
        if node is None:
            raise ValueError(
                "No canonical ResearchIdea projection for retained source object: "
                + source_object_id
            )
        if candidate_origin == "FRONTIER" and node.origin_kind != "FRONTIER":
            raise ValueError("FRONTIER candidate mapped to non-FRONTIER ResearchIdea")
        if candidate_origin == "EVOLUTION" and node.origin_kind not in {
            "EVOLUTION",
            "IMPORTED_REFRAME",
        }:
            raise ValueError("EVOLUTION candidate mapped to incompatible ResearchIdea")

        seen_source_ids.add(source_object_id)
        selected.append(node)
        links.append(
            E2ESeedSelectionLink(
                candidate_id=candidate_id,
                source_object_id=source_object_id,
                candidate_origin=candidate_origin,
                research_idea_id=node.idea_id,
                research_idea_origin_kind=node.origin_kind,
                original_generation_index=node.generation_index,
            )
        )

    return selected, links


def build_e2e_research_idea_seed(
    *,
    population: FrontierIdeaPopulation,
    evolution_report: IdeaEvolutionReport,
    pool: ScientificPortfolioCandidatePool,
    selection: ScientificPortfolioSelectionReport,
) -> ResearchIdeaE2ESeedReport:
    if evolution_report.source_population_id != population.population_id:
        raise ValueError("Idea Evolution/Frontier population lineage mismatch")
    if evolution_report.source_context_id != population.source_context_id:
        raise ValueError("Idea Evolution/Frontier source context mismatch")
    if pool.source_population_id != population.population_id:
        raise ValueError("Scientific Portfolio pool/Frontier lineage mismatch")
    if pool.source_population_sha256 != population.population_sha256:
        raise ValueError("Scientific Portfolio pool/Frontier SHA mismatch")
    if pool.source_evolution_report_id != evolution_report.report_id:
        raise ValueError("Scientific Portfolio pool/Idea Evolution lineage mismatch")
    if pool.source_evolution_report_sha256 != evolution_report.report_sha256:
        raise ValueError("Scientific Portfolio pool/Idea Evolution SHA mismatch")
    if selection.source_pool_id != pool.pool_id:
        raise ValueError("Scientific Portfolio selection/pool lineage mismatch")
    if selection.source_pool_sha256 != pool.pool_sha256:
        raise ValueError("Scientific Portfolio selection/pool SHA mismatch")
    if pool.source_context_id != population.source_context_id:
        raise ValueError("Scientific Portfolio pool/context lineage mismatch")
    if pool.source_context_sha256 != population.source_context_sha256:
        raise ValueError("Scientific Portfolio pool/context SHA mismatch")

    projected = project_current_idea_artifacts(
        population=population,
        evolution_report=evolution_report,
    )
    selected, links = select_retained_research_ideas(
        projected_nodes=projected,
        pool_candidates=pool.candidates,
        selection_entries=selection.entries,
    )
    if not selected:
        raise ValueError("Scientific Portfolio retained zero ResearchIdeas; SIS P0 is empty")
    if len(selected) != selection.retained_count:
        raise ValueError("retained Scientific Portfolio count / SIS P0 count mismatch")

    provisional = ResearchIdeaE2ESeedReport(
        seed_id="pending",
        seed_sha256="pending",
        source_population_id=population.population_id,
        source_population_sha256=population.population_sha256,
        source_evolution_report_id=evolution_report.report_id,
        source_evolution_report_sha256=evolution_report.report_sha256,
        source_pool_id=pool.pool_id,
        source_pool_sha256=pool.pool_sha256,
        source_selection_id=selection.selection_id,
        source_selection_sha256=selection.selection_sha256,
        source_context_id=population.source_context_id,
        source_context_sha256=population.source_context_sha256,
        selected_candidate_count=len(selected),
        selected_candidate_ids=[row.candidate_id for row in links],
        selected_source_object_ids=[row.source_object_id for row in links],
        selected_research_idea_ids=[row.idea_id for row in selected],
        selected_count_by_candidate_origin=dict(
            sorted(Counter(row.candidate_origin for row in links).items())
        ),
        links=links,
        research_ideas=selected,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("seed_id", None)
    payload.pop("seed_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "seed_id": f"sis_v3_2_e2e_seed:{digest[:20]}",
            "seed_sha256": digest,
        }
    )


def build_e2e_seed_execution(
    seed: ResearchIdeaE2ESeedReport,
) -> EpistemicG4ExecutionReport:
    """Wrap a frozen E2E P0 in the existing SIS-v3.x execution contract.

    Generation index 2 is a compatibility search-epoch index required by the existing
    EpistemicG4ExecutionReport. ResearchIdeaNode generation/genealogy is preserved
    verbatim; no idea is rewritten to pretend it was born in generation 2.
    """

    idea_ids = list(seed.selected_research_idea_ids)
    provisional = EpistemicG4ExecutionReport(
        report_id="pending",
        report_sha256="pending",
        source_parallel_report_id=seed.seed_id,
        source_plan_id=seed.source_selection_id,
        generation_index=seed.compatibility_execution_generation_index,
        tasks=[],
        run_records=[],
        offspring_nodes=[],
        semantic_records=[],
        g4_population_nodes=[row.model_dump(mode="json") for row in seed.research_ideas],
        raw_offspring_count=0,
        g4_population_count=len(seed.research_ideas),
        genuine_child_count=0,
        indeterminate_probe_count=0,
        same_idea_refinement_count=0,
        exact_kernel_duplicate_suppressed_count=0,
        identity_relation_counts={},
        disposition_counts={},
        generated_count_by_channel={},
        generated_count_by_operator={},
        genuine_child_from_non_grounded_feedback_count=0,
        genuine_child_from_speculative_feedback_count=0,
        genuine_child_from_evidence_seeking_feedback_count=0,
        llm_call_count=0,
        input_tokens=0,
        output_tokens=0,
        semantic_retry_count=0,
        offspring_generation_executed=False,
        carried_forward_idea_ids=idea_ids,
        carried_forward_count=len(idea_ids),
        replaced_parent_idea_ids=[],
        replaced_parent_count=0,
        population_composed_with_persistence=False,
        population_growth_budget=0,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"sis_v3_2_e2e_seed_execution:{digest[:20]}",
            "report_sha256": digest,
        }
    )


__all__ = [
    "E2ESeedSelectionLink",
    "ResearchIdeaE2ESeedReport",
    "select_retained_research_ideas",
    "build_e2e_research_idea_seed",
    "build_e2e_seed_execution",
]
